#!/usr/bin/env python3
"""Tests unitarios de validación de entradas."""

import pytest
from fastapi import HTTPException

from security.validation import (
    validate_text,
    require_text,
    validate_voice_name,
    validate_model_id,
    reject_path_traversal,
)


def test_validate_text_empty_raises():
    with pytest.raises(HTTPException):
        validate_text("", 1000)
    with pytest.raises(HTTPException):
        validate_text("   ", 1000)


def test_validate_text_too_long_raises():
    with pytest.raises(HTTPException):
        validate_text("a" * 1001, 1000)


def test_validate_text_ok():
    validate_text("Hola", 1000)


def test_require_text():
    with pytest.raises(HTTPException):
        require_text(None)
    with pytest.raises(HTTPException):
        require_text("")
    require_text("Hola")


def test_validate_voice_name():
    with pytest.raises(HTTPException):
        validate_voice_name("")
    with pytest.raises(HTTPException):
        validate_voice_name("../evil")
    validate_voice_name("annab_2")


def test_validate_model_id():
    with pytest.raises(HTTPException):
        validate_model_id("")
    validate_model_id("model-x")


@pytest.mark.parametrize("evil", [
    "../x",
    "../../etc/passwd",
    "/etc/passwd",
    "/home/user/voices/myvoice/reference.wav",
    "C:\\Windows\\System32",
    "C:/Windows",
    "a/b",
    "a\\b",
    "..",
    ".",
    "/",
    "\\",
    "voice\x00_1",
    "",
])
def test_reject_path_traversal(evil):
    with pytest.raises(HTTPException, match="ruta"):
        reject_path_traversal(evil, "voice_id")


def test_reject_path_traversal_allows_safe_values():
    for safe in ("voice_7f32a1", "Narrador", "maria_v1", "voz-2", "abc_123"):
        reject_path_traversal(safe, "voice_id")


def test_validate_config_update_dtype():
    from security.validation import validate_config_update

    class FakeConfigService:
        VALID_DTYPES = ("auto", "bfloat16", "float16", "float32")

        def validate_dtype(self, dtype):
            return dtype in self.VALID_DTYPES

        def validate_device(self, device):
            return device in ("auto", "cpu")

    cs = FakeConfigService()
    with pytest.raises(HTTPException):
        validate_config_update({"dtype": "float64"}, cs)
    with pytest.raises(HTTPException):
        validate_config_update({"unload_tts_for_whisper": "true"}, cs)
    with pytest.raises(HTTPException):
        validate_config_update({"unload_whisper_for_tts": 1}, cs)
    validate_config_update({"dtype": "bfloat16", "unload_tts_for_whisper": True}, cs)


def test_validate_config_update_new_fields():
    """Los nuevos ajustes editables se validan con rangos razonables."""
    from security.validation import validate_config_update

    class FakeConfigService:
        VALID_DTYPES = ("auto", "bfloat16", "float16", "float32")

        def validate_dtype(self, dtype):
            return dtype in self.VALID_DTYPES

        def validate_device(self, device):
            return device in ("auto", "cpu")

    cs = FakeConfigService()

    # Valores inválidos: se rechazan
    invalid = [
        {"chunking": "word"},
        {"normalization_dbfs": 5},
        {"normalization_dbfs": -100},
        {"max_parallel_inference": 0},
        {"max_parallel_inference": 99},
        {"max_text_characters": 0},
        {"max_estimated_audio_duration_seconds": -5},
        {"max_reference_audio_mb": 0},
        {"max_reference_duration_seconds": None},
        {"max_voice_audio_bytes_mb": -1},
        {"max_voice_audio_duration_seconds": 0},
        {"max_transcribe_audio_bytes_mb": 0},
        {"max_transcribe_duration_seconds": 0},
        {"generated_audio_ttl_hours": 0},
        {"min_sample_rate": 500},
        {"max_sample_rate": 500000},
        {"min_sample_rate": 48000, "max_sample_rate": 16000},
        {"max_channels": 0},
        {"max_channels": 9},
        {"normalize_reference_audio": "si"},
    ]
    for changes in invalid:
        with pytest.raises(HTTPException, match="debe|inválido|menor|entre"):
            validate_config_update(changes, cs)

    # Valores válidos: pasan
    validate_config_update(
        {
            "chunking": "paragraph",
            "normalization_dbfs": -1.5,
            "max_parallel_inference": 2,
            "max_text_characters": 20000,
            "max_estimated_audio_duration_seconds": 60,
            "max_reference_audio_mb": 30,
            "max_reference_duration_seconds": 90,
            "max_voice_audio_bytes_mb": 75,
            "max_voice_audio_duration_seconds": 150,
            "max_transcribe_audio_bytes_mb": 200,
            "max_transcribe_duration_seconds": 900,
            "generated_audio_ttl_hours": 48,
            "min_sample_rate": 8000,
            "max_sample_rate": 96000,
            "max_channels": 2,
            "normalize_reference_audio": True,
        },
        cs,
    )


def test_validate_config_update_chunking_none():
    """El modo 'none' (sin división) es válido en la configuración runtime."""
    from fastapi import HTTPException
    from security.validation import validate_config_update

    class FakeConfigService:
        VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
        VALID_DTYPES = ("auto", "bfloat16", "float16", "float32")

        def validate_dtype(self, dtype):
            return dtype in self.VALID_DTYPES

        def validate_device(self, device):
            return device in ("auto", "cpu")

    cs = FakeConfigService()
    validate_config_update({"chunking": "none"}, cs)
    with pytest.raises(HTTPException):
        validate_config_update({"chunking": "frases"}, cs)


def test_validate_config_update_whisper_model():
    """whisper_model se valida contra la whitelist del downloader (fuente
    única): los modelos conocidos pasan, los desconocidos se rechazan."""
    from fastapi import HTTPException
    from security.validation import validate_config_update
    from services.model_downloader import SUPPORTED_MODELS

    class FakeConfigService:
        def get_runtime_config(self):
            return {}

    cs = FakeConfigService()
    whisper_names = [m["name"] for m in SUPPORTED_MODELS if m["kind"] == "whisper"]
    assert whisper_names, "el downloader debe declarar modelos Whisper"

    # Todos los modelos Whisper soportados pasan
    for name in whisper_names:
        validate_config_update({"whisper_model": name}, cs)

    # Desconocidos / vacíos / no-string: se rechazan
    for bad in ("whisper-huge", "Qwen3-TTS-12Hz-1.7B-Base", "", "   ", 123, None):
        with pytest.raises(HTTPException, match="whisper_model inválido"):
            validate_config_update({"whisper_model": bad}, cs)


def test_validate_config_update_partial_sample_rate_range():
    """Una actualización parcial también valida el rango contra el valor
    vigente (no solo cuando min y max vienen juntos)."""
    from fastapi import HTTPException
    from security.validation import validate_config_update

    class FakeConfigService:
        def get_runtime_config(self):
            return {"min_sample_rate": 96000, "max_sample_rate": 96000}

    cs = FakeConfigService()
    # max actualizado a un valor menor que el min vigente: se rechaza
    with pytest.raises(HTTPException, match="menor"):
        validate_config_update({"max_sample_rate": 8000}, cs)
    # min actualizado a un valor mayor que el max vigente: se rechaza
    with pytest.raises(HTTPException, match="menor"):
        validate_config_update({"min_sample_rate": 192000}, cs)
    # coherentes con el vigente: pasan
    validate_config_update({"max_sample_rate": 192000}, cs)
    validate_config_update({"min_sample_rate": 4000}, cs)


# -- Contención de rutas (reference_audio) y referencias de voz -------------


def test_is_safe_voice_ref():
    from security.validation import is_safe_voice_ref

    assert is_safe_voice_ref("voice_7f32a1")
    assert is_safe_voice_ref("Serena")
    assert not is_safe_voice_ref("")
    assert not is_safe_voice_ref(".")
    assert not is_safe_voice_ref("..")
    assert not is_safe_voice_ref("/etc/passwd")
    assert not is_safe_voice_ref("a/b")
    assert not is_safe_voice_ref("a\\b")
    assert not is_safe_voice_ref("a..b")
    assert not is_safe_voice_ref("C:\\x")
    assert not is_safe_voice_ref("voice\x00_1")


def test_resolve_contained_path_inside_root(tmp_path):
    from security.validation import resolve_contained_path

    root = tmp_path / "proj"
    inner = root / "audios"
    inner.mkdir(parents=True)
    f = inner / "ref.wav"
    f.write_bytes(b"x")

    # Ruta absoluta dentro de root
    assert resolve_contained_path(str(f), str(root)) == str(f)
    # Ruta relativa (se resuelve contra root)
    assert resolve_contained_path("audios/ref.wav", str(root)) == str(f)


def test_resolve_contained_path_rejects_outside(tmp_path):
    from security.validation import resolve_contained_path

    root = tmp_path / "proj"
    root.mkdir()

    # Ruta absoluta externa
    with pytest.raises(ValueError, match="dentro del directorio"):
        resolve_contained_path("/etc/hostname", str(root))
    # Traversal con ..
    with pytest.raises(ValueError):
        resolve_contained_path("../../etc/hostname", str(root))
    # Vacío / NUL
    with pytest.raises(ValueError):
        resolve_contained_path("", str(root))
    with pytest.raises(ValueError):
        resolve_contained_path("ref\x00.wav", str(root))


def test_resolve_contained_path_rejects_symlink_escape(tmp_path):
    """Un symlink dentro de root que apunta fuera no permite escapar."""
    import os
    from security.validation import resolve_contained_path

    root = tmp_path / "proj"
    root.mkdir()
    os.symlink("/etc", str(root / "escape"))

    with pytest.raises(ValueError):
        resolve_contained_path(str(root / "escape" / "hostname"), str(root))
