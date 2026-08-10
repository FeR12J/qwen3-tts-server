#!/usr/bin/env python3
"""Configuración en tiempo de ejecución persistente del servidor TTS."""

import logging

from config.defaults import DEFAULTS
from storage.config_storage import load_runtime_file, save_runtime_file

logger = logging.getLogger("tts")

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

_runtime = dict(DEFAULTS)


def load_runtime_config():
    """Cargar configuración persistida desde disco (si existe)."""
    global _runtime
    _runtime = dict(DEFAULTS)
    data = load_runtime_file()
    for key in DEFAULTS:
        if key in data and data[key] is not None:
            _runtime[key] = data[key]


def save_runtime_config():
    """Persistir la configuración en disco."""
    save_runtime_file(_runtime)


def get_runtime_config() -> dict:
    """Devolver la configuración en tiempo de ejecución."""
    return dict(_runtime)


def update_runtime_config(changes: dict) -> dict:
    """Actualizar claves válidas de la configuración en tiempo de ejecución."""
    for key, value in changes.items():
        if key in DEFAULTS and value is not None:
            _runtime[key] = value
    save_runtime_config()
    return get_runtime_config()


def apply_log_level():
    """Aplicar el nivel de logging configurado al logger principal."""
    level = _runtime.get("log_level", "INFO")
    if level not in VALID_LOG_LEVELS:
        level = "INFO"
    logging.getLogger("tts").setLevel(level)


def resolve_device() -> str:
    """Resolver el dispositivo a usar según la configuración y lo disponible.

    Devuelve "cuda:N" o "cpu". Si la GPU solicitada no existe, usa cuda:0.
    """
    import torch

    requested = _runtime.get("device", "auto")
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
