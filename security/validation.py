#!/usr/bin/env python3
"""Validación de entradas de la API."""

import os
import re

from fastapi import HTTPException

VOICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


def is_safe_voice_ref(value: str) -> bool:
    """True si el valor es un id/nombre de voz usable (no una ruta).

    Única regla anti path-traversal para referencias de voz: la usan las
    rutas (vía reject_path_traversal), el VoiceManager y el TTSService, para
    que no haya cuatro variantes distintas de la misma regla.
    """
    v = str(value or "")
    if not v or v in (".", "..") or "\x00" in v:
        return False
    if "/" in v or "\\" in v or ".." in v:
        return False
    if v.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", v):
        return False
    return True


def resolve_contained_path(raw: str, root: str, field: str = "reference_audio") -> str:
    """Resolver una ruta de servidor garantizando que queda dentro de ``root``.

    Resuelve la ruta real (incluidos enlaces simbólicos) y lanza ValueError
    si el valor intenta escapar del directorio (ruta absoluta externa,
    ``..``, symlink fuera de root...). Devuelve la ruta absoluta resuelta.

    Es la contención de ``reference_audio`` (TTS): el cliente puede indicar
    un archivo del servidor, pero solo dentro del directorio del proyecto.
    """
    raw = str(raw or "")
    if not raw or "\x00" in raw:
        raise ValueError(f"{field} debe ser una ruta de archivo válida")
    base = os.path.realpath(root)
    candidate = raw if os.path.isabs(raw) else os.path.join(base, raw)
    path = os.path.realpath(candidate)
    if path != base and not path.startswith(base + os.sep):
        raise ValueError(
            f"{field} debe estar dentro del directorio del proyecto "
            f"({base}): no se permiten rutas externas"
        )
    return path


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
    if not is_safe_voice_ref(value):
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
    if "whisper_timestamps" in changes and changes["whisper_timestamps"] not in (
        "off", "segment", "word"
    ):
        raise HTTPException(400, "whisper_timestamps inválido. Válidos: off, segment, word")
    if "whisper_model" in changes:
        model = changes["whisper_model"]
        # Fuente única de los modelos instalables: la whitelist del downloader.
        from services.model_downloader import SUPPORTED_MODELS
        whisper_models = [m["name"] for m in SUPPORTED_MODELS if m["kind"] == "whisper"]
        if not isinstance(model, str) or not model.strip() or model not in whisper_models:
            raise HTTPException(
                400,
                "whisper_model inválido. Disponibles: " + ", ".join(whisper_models),
            )
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
    if "queue_max_size" in changes and (
        changes["queue_max_size"] is None
        or not 1 <= changes["queue_max_size"] <= 100
    ):
        raise HTTPException(400, "queue_max_size debe estar entre 1 y 100")
    if "port" in changes and (
        changes["port"] is None or not 1 <= changes["port"] <= 65535
    ):
        raise HTTPException(400, "port debe estar entre 1 y 65535")
    if "cors_enabled" in changes and not isinstance(changes["cors_enabled"], bool):
        raise HTTPException(400, "cors_enabled debe ser true o false")
    if "cors_allow_wildcard" in changes and not isinstance(
        changes["cors_allow_wildcard"], bool
    ):
        raise HTTPException(400, "cors_allow_wildcard debe ser true o false")
    if "cors_origins" in changes:
        origins = changes["cors_origins"]
        if not isinstance(origins, list) or not all(
            isinstance(o, str) and o.strip() for o in origins
        ):
            raise HTTPException(400, "cors_origins debe ser una lista de orígenes (URLs)")
        if any("*" in o for o in origins):
            raise HTTPException(400, "cors_origins no puede contener '*'")
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
    # Rango min/max: se valida contra el valor vigente (no solo cuando ambos
    # vienen en la misma actualización, o una actualización parcial podría
    # persistir un rango invertido y rechazar todos los audios).
    if "min_sample_rate" in changes and (
        changes["min_sample_rate"] is None
        or not 1000 <= changes["min_sample_rate"] <= 384000
    ):
        raise HTTPException(400, "min_sample_rate debe estar entre 1000 y 384000 Hz")
    if "max_sample_rate" in changes and (
        changes["max_sample_rate"] is None
        or not 1000 <= changes["max_sample_rate"] <= 384000
    ):
        raise HTTPException(400, "max_sample_rate debe estar entre 1000 y 384000 Hz")
    if "min_sample_rate" in changes or "max_sample_rate" in changes:
        eff_min = changes.get("min_sample_rate")
        eff_max = changes.get("max_sample_rate")
        if eff_min is None or eff_max is None:
            # Falta uno de los dos: se completa con el valor vigente.
            getter = getattr(config_service, "get_runtime_config", None)
            current = getter() if callable(getter) else {}
            if eff_min is None:
                eff_min = current.get("min_sample_rate")
            if eff_max is None:
                eff_max = current.get("max_sample_rate")
        if eff_min is not None and eff_max is not None and eff_min > eff_max:
            raise HTTPException(
                400, "max_sample_rate no puede ser menor que min_sample_rate"
            )
    if "max_channels" in changes and (
        changes["max_channels"] is None or not 1 <= changes["max_channels"] <= 8
    ):
        raise HTTPException(400, "max_channels debe estar entre 1 y 8")
    for flag in ("unload_tts_for_whisper", "unload_whisper_for_tts",
                 "streaming_enabled", "save_audios", "normalize_reference_audio",
                 "queue_enabled", "log_input_text", "flash_attn"):
        if flag in changes and not isinstance(changes[flag], bool):
            raise HTTPException(400, f"{flag} debe ser true o false")
