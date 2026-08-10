#!/usr/bin/env python3
"""Configuración en tiempo de ejecución persistente del servidor TTS.

La fuente de verdad única es ``config.settings.settings`` (Settings pydantic);
este módulo solo gestiona la parte editable (settings.runtime) y su
persistencia en disco.
"""

import logging

from config.settings import settings, RuntimeSettings
from storage.config_storage import load_runtime_file, save_runtime_file

logger = logging.getLogger("tts")

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VALID_DTYPES = ("auto", "bfloat16", "float16", "float32")

RUNTIME_FIELDS = tuple(RuntimeSettings.model_fields)


def load_runtime_config():
    """Cargar configuración persistida desde disco (si existe)."""
    settings.runtime = RuntimeSettings()
    data = load_runtime_file()
    valid = {
        key: value
        for key, value in data.items()
        if key in RUNTIME_FIELDS and value is not None
    }
    if valid:
        settings.runtime = settings.runtime.model_copy(update=valid)


def save_runtime_config():
    """Persistir la configuración en disco."""
    save_runtime_file(settings.runtime.model_dump())


def get_runtime_config() -> dict:
    """Devolver la configuración en tiempo de ejecución."""
    return settings.runtime.model_dump()


def get_runtime() -> RuntimeSettings:
    """Devolver la configuración en tiempo de ejecución tipada."""
    return settings.runtime


def update_runtime_config(changes: dict) -> dict:
    """Actualizar claves válidas de la configuración en tiempo de ejecución."""
    valid = {
        key: value
        for key, value in changes.items()
        if key in RUNTIME_FIELDS and value is not None
    }
    if valid:
        settings.runtime = settings.runtime.model_copy(update=valid)
    save_runtime_config()
    return get_runtime_config()


def apply_log_level():
    """Aplicar el nivel de logging configurado al logger principal."""
    level = settings.runtime.log_level
    if level not in VALID_LOG_LEVELS:
        level = "INFO"
    logging.getLogger("tts").setLevel(level)


def resolve_device() -> str:
    """Resolver el dispositivo a usar según la configuración y lo disponible.

    Devuelve "cuda:N" o "cpu". Si la GPU solicitada no existe, usa cuda:0.
    """
    import torch

    requested = settings.runtime.device
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested != "auto":
            logger.warning(f"Dispositivo '{requested}' solicitado pero no hay CUDA, usando cpu")
        return "cpu"
    if requested == "auto":
        return "cuda:0"
    if requested.startswith("cuda:"):
        try:
            idx = int(requested.split(":", 1)[1])
            if 0 <= idx < torch.cuda.device_count():
                return requested
        except ValueError:
            pass
        logger.warning(f"GPU '{requested}' no disponible, usando cuda:0")
        return "cuda:0"
    logger.warning(f"Dispositivo '{requested}' inválido, usando cuda:0")
    return "cuda:0"


def validate_device(device: str) -> bool:
    """Comprobar que el valor de device es válido y existe (si es cuda:N)."""
    if device in ("auto", "cpu"):
        return True
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
            import torch
            return 0 <= idx < torch.cuda.device_count()
        except (ValueError, ImportError):
            return False
    return False


def resolve_dtype() -> str:
    """Resolver el dtype configurado ("auto" -> segun GPU)."""
    requested = settings.runtime.dtype
    if requested in ("bfloat16", "float16", "float32"):
        return requested
    if resolve_device().startswith("cuda:"):
        import torch
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def validate_dtype(dtype: str) -> bool:
    """Comprobar que el valor de dtype es válido."""
    return dtype in VALID_DTYPES
