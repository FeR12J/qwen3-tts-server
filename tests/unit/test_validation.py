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
