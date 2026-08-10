#!/usr/bin/env python3
"""Utilidades de GPU y memoria."""

import logging

import torch

logger = logging.getLogger("tts")


def _resolve_device() -> str:
    """Import perezoso para evitar el ciclo: utils -> services -> utils."""
    from services.config_service import resolve_device
    return resolve_device()


def get_vram_available():
    """Obtener VRAM disponible en GB de la GPU seleccionada en la configuración."""
    device = _resolve_device()
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
            if torch.cuda.is_available() and idx < torch.cuda.device_count():
                return round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1)
        except (ValueError, IndexError, RuntimeError):
            pass
    return 0.0


def get_dtype():
    """Determinar dtype óptimo para el dispositivo configurado."""
    if _resolve_device().startswith("cuda:"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def list_devices() -> dict:
    """Listar GPUs disponibles (para el panel web)."""
    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1e9, 1)
                })
            except Exception:
                devices.append({"index": i, "name": "GPU desconocida", "vram_gb": None})
    return {
        "cuda_available": cuda_available,
        "count": torch.cuda.device_count() if cuda_available else 0,
        "devices": devices,
    }
