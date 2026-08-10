#!/usr/bin/env python3
"""Tests unitarios de validación de entradas."""

import pytest
from fastapi import HTTPException

from security.validation import (
    validate_text,
    require_text,
    validate_voice_name,
    validate_model_id,
    validate_audio_size,
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


def test_validate_audio_size():
    with pytest.raises(HTTPException):
        validate_audio_size(b"", 1024)
    with pytest.raises(HTTPException):
        validate_audio_size(b"x" * 1025, 1024)
    validate_audio_size(b"x" * 1024, 1024)
