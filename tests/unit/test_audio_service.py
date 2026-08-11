#!/usr/bin/env python3
"""Tests unitarios del servicio central de audio (services.audio_service)."""

import io
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

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
    config = SimpleNamespace(paths=SimpleNamespace(audios_dir="."))
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


def test_get_duration(service):
    assert service.get_duration(_wav_bytes(_sine())) == pytest.approx(DURATION, abs=0.01)


def test_validate_ok(service):
    info = service.validate(_wav_bytes(_sine()), filename="voz.wav", formats=AudioService.FORMATS)
    assert info["sample_rate"] == SR
    assert info["duration"] == pytest.approx(DURATION, abs=0.01)
    assert info["size_bytes"] > 0
    assert info["channels"] == 1


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
    assert info["format"] == "wav"


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
    assert info["sample_rate"] == SR


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
    assert info["duration"] == pytest.approx(DURATION, abs=0.05)


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

