#!/usr/bin/env python3
"""Validación de entradas de la API."""

import re

from fastapi import HTTPException

VOICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


def validate_text(text: str, max_chars: int):
    """Validar texto para TTS."""
    if not text or not text.strip():
        raise HTTPException(400, "Texto vacío")
    if len(text) > max_chars:
        raise HTTPException(400, f"Texto demasiado largo ({len(text)} chars). Máximo: {max_chars}")


def require_text(text: str):
    """Exigir un texto no vacío (endpoints OpenWebUI)."""
    if not text or not text.strip():
        raise HTTPException(400, "Campo 'text' o 'input' requerido")


def validate_voice_name(voice_name: str):
    """Validar que el nombre de la voz sea seguro (sin path traversal)."""
    if not voice_name:
        raise HTTPException(400, "voice_name vacío")
    if not VOICE_NAME_PATTERN.match(voice_name):
        raise HTTPException(
            400,
            "El nombre de la voz solo puede contener letras, números, guiones y guiones bajos",
        )


def validate_model_id(model_id: str):
    """Validar que el identificador de modelo no esté vacío."""
    if not model_id:
        raise HTTPException(400, "model_id vacío")


def validate_wav_upload(upload) -> None:
    """Validar que el archivo subido sea un WAV."""
    if not upload.filename or not upload.filename.lower().endswith(".wav"):
        raise HTTPException(400, "El archivo debe ser un WAV (.wav)")


def validate_audio_size(data: bytes, max_bytes: int, allow_empty: bool = False):
    """Validar el tamaño del audio subido."""
    if not allow_empty and len(data) == 0:
        raise HTTPException(400, "El archivo de audio está vacío")
    if len(data) > max_bytes:
        raise HTTPException(
            400,
            f"El archivo excede {max_bytes // (1024 * 1024)} MB",
        )


def validate_config_update(changes: dict, config_service):
    """Validar los campos de configuración en tiempo de ejecución."""
    if "max_text_chars" in changes and (changes["max_text_chars"] is None or changes["max_text_chars"] <= 0):
        raise HTTPException(400, "max_text_chars debe ser mayor que 0")
    if "playback_wait_timeout" in changes and (changes["playback_wait_timeout"] is None or changes["playback_wait_timeout"] <= 0):
        raise HTTPException(400, "playback_wait_timeout debe ser mayor que 0")
    if "log_level" in changes and changes["log_level"] not in config_service.VALID_LOG_LEVELS:
        raise HTTPException(400, f"log_level inválido. Válidos: {', '.join(config_service.VALID_LOG_LEVELS)}")
    if "device" in changes and not config_service.validate_device(changes["device"]):
        raise HTTPException(400, "device inválido. Válidos: auto, cpu o cuda:N con N dentro del rango de GPUs")
    if "dtype" in changes and not config_service.validate_dtype(changes["dtype"]):
        raise HTTPException(
            400,
            f"dtype inválido. Válidos: {', '.join(config_service.VALID_DTYPES)}",
        )
    for flag in ("unload_tts_for_whisper", "unload_whisper_for_tts"):
        if flag in changes and not isinstance(changes[flag], bool):
            raise HTTPException(400, f"{flag} debe ser true o false")
