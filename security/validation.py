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


def reject_path_traversal(value: str, field: str = "Valor") -> None:
    """Rechazar valores con forma de ruta (path traversal) con un 400 claro.

    Bloquea rutas absolutas (``/etc/...``, ``C:\\``), separadores, ``..``,
    ``.`` y NUL, de modo que un id/nombre de voz nunca pueda usarse para
    construir rutas en el servidor.
    """
    value = str(value or "")
    if (value.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
            or "/" in value
            or "\\" in value
            or ".." in value
            or "\x00" in value
            or value in ("", ".", "..")):
        raise HTTPException(
            400,
            f"{field} no puede ser una ruta de archivo (se rechaza '{value}'). "
            "Use solo un id o nombre de voz.",
        )


def validate_model_id(model_id: str):
    """Validar que el identificador de modelo no esté vacío."""
    if not model_id:
        raise HTTPException(400, "model_id vacío")


def validate_config_update(changes: dict, config_service):
    """Validar los campos de configuración en tiempo de ejecución."""
    if "max_text_chars" in changes and (changes["max_text_chars"] is None or changes["max_text_chars"] <= 0):
        raise HTTPException(400, "max_text_chars debe ser mayor que 0")
    if "playback_wait_timeout" in changes and (changes["playback_wait_timeout"] is None or changes["playback_wait_timeout"] <= 0):
        raise HTTPException(400, "playback_wait_timeout debe ser mayor que 0")
    if "log_level" in changes and changes["log_level"] not in config_service.VALID_LOG_LEVELS:
        raise HTTPException(400, f"log_level inválido. Válidos: {', '.join(config_service.VALID_LOG_LEVELS)}")
    if "device" in changes and not config_service.validate_device(changes["device"]):
        raise HTTPException(400, "device inválido. Válidos: auto, cpu, cuda o cuda:N con N dentro del rango de GPUs")
    if "dtype" in changes and not config_service.validate_dtype(changes["dtype"]):
        raise HTTPException(
            400,
            f"dtype inválido. Válidos: {', '.join(config_service.VALID_DTYPES)}",
        )
    if "chunking" in changes and changes["chunking"] not in ("sentence", "paragraph"):
        raise HTTPException(400, "chunking inválido. Válidos: sentence, paragraph")
    if "normalization_dbfs" in changes and (
        changes["normalization_dbfs"] is None
        or not isinstance(changes["normalization_dbfs"], (int, float))
        or changes["normalization_dbfs"] > 0
        or changes["normalization_dbfs"] < -60
    ):
        raise HTTPException(400, "normalization_dbfs debe ser un valor en dBFS entre -60 y 0")
    if "max_parallel_inference" in changes and (
        changes["max_parallel_inference"] is None
        or not 1 <= changes["max_parallel_inference"] <= 16
    ):
        raise HTTPException(400, "max_parallel_inference debe estar entre 1 y 16")
    for field in (
        "max_text_characters", "max_estimated_audio_duration_seconds",
        "max_reference_audio_mb", "max_reference_duration_seconds",
        "max_voice_audio_bytes_mb", "max_voice_audio_duration_seconds",
        "max_transcribe_audio_bytes_mb", "max_transcribe_duration_seconds",
    ):
        if field in changes and (changes[field] is None or changes[field] <= 0):
            raise HTTPException(400, f"{field} debe ser mayor que 0")
    if "generated_audio_ttl_hours" in changes and (
        changes["generated_audio_ttl_hours"] is None
        or changes["generated_audio_ttl_hours"] <= 0
    ):
        raise HTTPException(400, "generated_audio_ttl_hours debe ser mayor que 0")
    if "min_sample_rate" in changes and (
        changes["min_sample_rate"] is None
        or not 1000 <= changes["min_sample_rate"] <= 384000
    ):
        raise HTTPException(400, "min_sample_rate debe estar entre 1000 y 384000 Hz")
    if "max_sample_rate" in changes:
        if changes["max_sample_rate"] is None or not 1000 <= changes["max_sample_rate"] <= 384000:
            raise HTTPException(400, "max_sample_rate debe estar entre 1000 y 384000 Hz")
        if ("min_sample_rate" in changes and changes["min_sample_rate"] > changes["max_sample_rate"]):
            raise HTTPException(400, "max_sample_rate no puede ser menor que min_sample_rate")
    if "max_channels" in changes and (
        changes["max_channels"] is None or not 1 <= changes["max_channels"] <= 8
    ):
        raise HTTPException(400, "max_channels debe estar entre 1 y 8")
    for flag in ("unload_tts_for_whisper", "unload_whisper_for_tts",
                 "streaming_enabled", "save_audios", "normalize_reference_audio"):
        if flag in changes and not isinstance(changes[flag], bool):
            raise HTTPException(400, f"{flag} debe ser true o false")
