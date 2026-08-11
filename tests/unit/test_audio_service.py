#!/usr/bin/env python3
"""Tests unitarios del servicio central de audio (services.audio_service)."""

import io
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from config.settings import settings
from services.audio_service import (
    SPEECH_SAMPLE_RATE,
    AudioService,
    AudioValidationError,
)

pytestmark = pytest.mark.skipif(
    not hasattr(np, "sin"), reason="numpy requerido"
)

SR = 24000
DURATION = 0.25
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _sine(sr=SR, seconds=DURATION, amplitude=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype("float32")


def _wav_bytes(wav, sr=SR):
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format="wav")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def service():
    config = SimpleNamespace(
        paths=SimpleNamespace(audios_dir="."),
        audio=SimpleNamespace(normalization_dbfs=-1.0),
    )
    return AudioService(config, None)


def test_load_wav_returns_mono_float32(service):
    wav = _sine()
    audio, sr = service.load(_wav_bytes(wav))
    assert sr == SR
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == len(wav)


def test_load_resamples_to_target_sr(service):
    audio, sr = service.load(_wav_bytes(_sine()), target_sr=SPEECH_SAMPLE_RATE)
    assert sr == SPEECH_SAMPLE_RATE
    expected = int(SR * DURATION * SPEECH_SAMPLE_RATE / SR)
    assert abs(audio.shape[0] - expected) < 2


def test_load_mixes_stereo_to_mono(service):
    stereo = np.stack([_sine(), _sine() * 0.5], axis=1).astype("float32")
    audio, _ = service.load(_wav_bytes(stereo))
    assert audio.ndim == 1
    assert audio.shape[0] == stereo.shape[0]


def test_load_path_and_file_like(service, tmp_path):
    wav = _sine()
    path = tmp_path / "ref.wav"
    sf.write(str(path), wav, SR)
    audio, sr = service.load(str(path))
    assert sr == SR and audio.shape[0] == len(wav)
    with open(path, "rb") as f:
        audio2, _ = service.load(f)
    assert np.allclose(audio, audio2)


def test_load_invalid_audio_raises(service):
    with pytest.raises(AudioValidationError):
        service.load(b"esto no es audio")


@pytest.mark.parametrize("fmt", ["wav", "flac", "ogg"])
def test_convert_roundtrip_soundfile_formats(service, fmt):
    wav = _sine()
    data = service.convert(wav, SR, fmt)
    audio, sr = service.load(data)
    assert sr == SR
    assert audio.shape[0] == len(wav)


@pytest.mark.parametrize("fmt", ["mp3", "m4a"])
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg no disponible")
def test_convert_roundtrip_ffmpeg_formats(service, fmt):
    wav = _sine()
    data = service.convert(wav, SR, fmt)
    audio, sr = service.load(data, target_sr=SR)
    assert sr == SR
    # Codificación con pérdida: solo se comprueba que sea decodificable.
    assert audio.shape[0] > 0


def test_convert_unsupported_format_raises(service):
    with pytest.raises(AudioValidationError):
        service.convert(_sine(), SR, "oggopus")


def test_convert_mono_to_stereo(service):
    data = service.convert(_sine(), SR, "wav", channels=2)
    audio, _ = service.load(data, mono=False)
    assert audio.ndim == 2 and audio.shape[1] == 2


def test_normalize_quiet_audio(service):
    wav = _sine(amplitude=0.01)
    out = service.normalize(wav)
    assert float(np.max(np.abs(out))) == pytest.approx(10 ** (-1.0 / 20.0), abs=1e-4)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 1.0


def test_normalize_silence_unchanged(service):
    silent = np.zeros(1000, dtype="float32")
    out = service.normalize(silent)
    assert np.allclose(out, silent)


def test_normalize_custom_dbfs(service):
    wav = _sine(amplitude=0.01)
    out = service.normalize(wav, dbfs=-6.0)
    assert float(np.max(np.abs(out))) == pytest.approx(10 ** (-6.0 / 20.0), abs=1e-4)
    # 0 dBFS = pico a escala completa (1.0)
    out = service.normalize(wav, dbfs=0.0)
    assert float(np.max(np.abs(out))) == pytest.approx(1.0, abs=1e-4)


def test_normalize_uses_config_default(service, monkeypatch):
    # El valor editable (runtime) es la fuente del dBFS por defecto
    monkeypatch.setattr(settings.runtime, "normalization_dbfs", -3.0)
    svc = AudioService(service._config, None)
    out = svc.normalize(_sine(amplitude=0.01))
    assert float(np.max(np.abs(out))) == pytest.approx(10 ** (-3.0 / 20.0), abs=1e-4)


def test_get_duration(service):
    assert service.get_duration(_wav_bytes(_sine())) == pytest.approx(DURATION, abs=0.01)


def test_validate_ok(service):
    info = service.validate(_wav_bytes(_sine()), filename="voz.wav", formats=AudioService.FORMATS)
    assert info.sample_rate == SR
    assert info.duration == pytest.approx(DURATION, abs=0.01)
    assert info.size_bytes > 0
    assert info.channels == 1
    # Sin decode=True no se decodifica: info es solo cabecera.
    assert info.samples is None


def test_validate_decode_returns_audio(service):
    """decode=True decodifica una sola vez y expone el array para reutilizar."""
    wav = _sine()
    info = service.validate(
        _wav_bytes(wav),
        filename="voz.wav",
        formats=AudioService.FORMATS,
        decode=True,
    )
    assert info.samples is not None
    assert info.samples.shape[0] == int(SR * DURATION)
    assert info.sample_rate == SR


def test_validate_decode_prepare_reuses_audio(service):
    """El array de validate() se puede post-procesar sin volver a decodificar."""
    info = service.validate(
        _wav_bytes(_sine()),
        filename="voz.wav",
        formats=AudioService.FORMATS,
        decode=True,
    )
    wav, sr = service.prepare(info.samples, info.sample_rate, target_sr=8000)
    assert sr == 8000
    assert wav.shape[0] == int(8000 * DURATION)


def test_validate_empty_raises(service):
    with pytest.raises(AudioValidationError):
        service.validate(b"", filename="a.wav")


def test_validate_max_bytes(service):
    wav = _sine(seconds=1.0)
    with pytest.raises(AudioValidationError):
        service.validate(_wav_bytes(wav), max_bytes=1000)


def test_validate_max_duration(service):
    wav = _sine(seconds=1.0)
    with pytest.raises(AudioValidationError):
        service.validate(_wav_bytes(wav), max_duration=0.5)


def test_validate_unallowed_extension(service):
    with pytest.raises(AudioValidationError):
        service.validate(_wav_bytes(_sine()), filename="voz.exe", formats=AudioService.FORMATS)


def test_validate_not_audio_raises(service):
    with pytest.raises(AudioValidationError):
        service.validate(b"data de relleno", filename="a.wav")


def test_validate_mime_matches(service):
    info = service.validate(
        _wav_bytes(_sine()),
        filename="clip.wav",
        content_type="audio/wav",
    )
    assert info.format == "wav"


def test_validate_mime_mismatch_with_extension(service):
    with pytest.raises(AudioValidationError, match="MIME"):
        service.validate(
            _wav_bytes(_sine()),
            filename="clip.wav",
            content_type="audio/mpeg",
        )


def test_validate_unsupported_mime_raises(service):
    with pytest.raises(AudioValidationError, match="MIME"):
        service.validate(
            _wav_bytes(_sine()),
            filename="clip.wav",
            content_type="image/png",
        )


def test_validate_octet_stream_is_lenient(service):
    info = service.validate(
        _wav_bytes(_sine()),
        filename="clip.wav",
        content_type="application/octet-stream",
    )
    assert info.sample_rate == SR


def test_validate_content_mismatch_with_extension(service):
    # Bytes de un WAV con nombre .mp3: el contenido real contradice la extensión.
    with pytest.raises(AudioValidationError, match="contenido real"):
        service.validate(_wav_bytes(_sine()), filename="falso.mp3")


def test_validate_sample_rate_range(service):
    wav = _sine()
    with pytest.raises(AudioValidationError, match="Sample rate"):
        service.validate(_wav_bytes(wav), min_sample_rate=SR + 1)
    with pytest.raises(AudioValidationError, match="Sample rate"):
        service.validate(_wav_bytes(wav), max_sample_rate=SR - 1)
    service.validate(_wav_bytes(wav), min_sample_rate=SR, max_sample_rate=SR)


def test_validate_channels_limit(service):
    stereo = np.stack([_sine(), _sine()], axis=1).astype("float32")
    with pytest.raises(AudioValidationError, match="canales"):
        service.validate(_wav_bytes(stereo), max_channels=1)
    service.validate(_wav_bytes(stereo), max_channels=2)


def test_validate_truncated_wav_rejected_with_decode(service):
    data = _wav_bytes(_sine())
    truncated = data[:44]  # solo cabecera, sin datos
    # La cabecera es válida pero el contenido está truncado: solo decode=True lo detecta.
    with pytest.raises(AudioValidationError):
        service.validate(truncated, filename="rot.wav", decode=True)


def test_validate_garbage_rejected(service):
    with pytest.raises(AudioValidationError):
        service.validate(b"\x00\x01\x02\x03 no audio", filename="x.wav", decode=True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg no disponible")
def test_validate_m4a_upload(service):
    m4a = service.convert(_sine(), SR, "m4a")
    info = service.validate(m4a, filename="clip.m4a", formats=AudioService.FORMATS)
    assert info.duration == pytest.approx(DURATION, abs=0.05)


def test_encode_pcm(service):
    wav = _sine()
    pcm = service.encode_pcm(wav, SR)
    assert len(pcm) == len(wav) * 2
    assert pcm[:2] != pcm[2:4]  # no es silencio


def _touch(path, age_seconds):
    import os
    import time
    old = time.time() - age_seconds
    os.utime(path, (old, old))


def test_cleanup_old_removes_only_expired(service, tmp_path, monkeypatch):
    config = SimpleNamespace(paths=SimpleNamespace(audios_dir=str(tmp_path)))
    svc = AudioService(config, None)
    fresh = tmp_path / "tts_fresh.wav"
    expired = tmp_path / "tts_expired.wav"
    sub = tmp_path / "subdir"
    sub.mkdir()
    for p in (fresh, expired):
        p.write_bytes(b"RIFFxxxx")
    (sub / "tts_anidado.wav").write_bytes(b"RIFFxxxx")
    _touch(str(expired), 3 * 3600)
    _touch(str(fresh), 1 * 3600)

    removed = svc.cleanup_old(2 * 3600)
    assert removed == 1
    assert expired.exists() is False
    assert fresh.exists() is True
    assert (sub / "tts_anidado.wav").exists() is True


def test_cleanup_old_ttl_zero_is_noop(service, tmp_path):
    config = SimpleNamespace(paths=SimpleNamespace(audios_dir=str(tmp_path)))
    svc = AudioService(config, None)
    (tmp_path / "a.wav").write_bytes(b"x")
    assert svc.cleanup_old(0) == 0
    assert (tmp_path / "a.wav").exists()


def test_save_skips_when_save_audios_disabled(service, tmp_path, monkeypatch):
    """Por defecto (save_audios=false) el audio NO se persiste: se devuelve por HTTP."""
    from services import config_service
    monkeypatch.setattr(config_service.settings.runtime, "save_audios", False)
    config = SimpleNamespace(paths=SimpleNamespace(audios_dir=str(tmp_path)))
    svc = AudioService(config, None)
    assert svc.save(_sine(), SR, "tts") == ""
    assert list(tmp_path.iterdir()) == []


def test_save_writes_when_save_audios_enabled(service, tmp_path, monkeypatch):
    import os
    from services import config_service
    monkeypatch.setattr(config_service.settings.runtime, "save_audios", True)
    config = SimpleNamespace(paths=SimpleNamespace(audios_dir=str(tmp_path)))
    svc = AudioService(config, None)
    path = svc.save(_sine(), SR, "tts")
    assert path.startswith(str(tmp_path))
    assert os.path.exists(path)
    audio, sr = svc.load(path)
    assert sr == SR and audio.shape[0] == int(SR * DURATION)


def test_save_names_never_collide_same_second(service, tmp_path, monkeypatch):
    """Dos guardados en el mismo segundo no pueden machacarse (sin confiar
    solo en el timestamp de 1 s de precisión)."""
    import os
    from services import config_service
    monkeypatch.setattr(config_service.settings.runtime, "save_audios", True)
    config = SimpleNamespace(paths=SimpleNamespace(audios_dir=str(tmp_path)))
    svc = AudioService(config, None)

    # Streaming: mismo request, distintos fragmentos -> request_id + chunk_index.
    p1 = svc.save(_sine(), SR, "tts_stream", request_id="req_8f3a12", chunk_index=0)
    p2 = svc.save(_sine(), SR, "tts_stream", request_id="req_8f3a12", chunk_index=1)
    assert p1 != p2
    assert "req_8f3a12_0" in p1 and "req_8f3a12_1" in p2
    assert os.path.exists(p1) and os.path.exists(p2)

    # Peticiones distintas (mismo prefijo, sin id) -> sufijo aleatorio único.
    p3 = svc.save(_sine(), SR, "tts")
    p4 = svc.save(_sine(), SR, "tts")
    assert p3 != p4
    assert os.path.exists(p3) and os.path.exists(p4)

    # Solo el id sanitizado entra en el nombre (nada de rutas/caracteres raros).
    p5 = svc.save(_sine(), SR, "tts", request_id="../../etc/passwd\x00A")
    assert "/../" not in p5
    assert str(tmp_path / os.path.basename(p5)) == p5
    assert os.path.exists(p5)

