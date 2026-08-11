#!/usr/bin/env python3
"""Rutas de estado del sistema (nivel PUBLIC).

/health y /ready son rápidas y no cargan modelos (solo consultan estado).
"""

import importlib.metadata
import logging

import torch

from fastapi import FastAPI

from config.settings import VERSION, settings
from services.config_service import resolve_device
from services import whisper_service

logger = logging.getLogger("tts")

SERVER_NAME = "qwen3-tts-server"


def _pkg_version(pkg: str):
    """Versión de un paquete instalado, o None si no está disponible."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_status() -> dict:
    """Estado de la GPU en uso (memoria en MB)."""
    try:
        device = resolve_device()
        if not (torch.cuda.is_available() and device.startswith("cuda:")):
            return {
                "available": False,
                "name": None,
                "total_vram_mb": 0,
                "used_vram_mb": 0,
                "free_vram_mb": 0,
            }
        idx = int(device.split(":", 1)[1])
        props = torch.cuda.get_device_properties(idx)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        return {
            "available": True,
            "name": props.name,
            "total_vram_mb": total_b // (1024 * 1024),
            "used_vram_mb": (total_b - free_b) // (1024 * 1024),
            "free_vram_mb": free_b // (1024 * 1024),
        }
    except Exception as e:
        logger.debug(f"No se pudo leer el estado de la GPU: {e}")
        return {
            "available": False,
            "name": None,
            "total_vram_mb": 0,
            "used_vram_mb": 0,
            "free_vram_mb": 0,
        }


def create_system_routes(app: FastAPI, ctx):
    """Rutas de estado y salud del servidor."""

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        """Métricas sencillas del servidor (JSON; Prometheus opcional)."""
        return ctx.metrics.get_metrics()

    @app.get("/ready")
    async def ready():
        return {
            "ready": True,
            "tts_model_loaded": ctx.models.is_loaded(),
        }

    @app.get("/system/status")
    async def system_status():
        tts = await ctx.models.get_model_status()
        return {
            "gpu": _gpu_status(),
            "tts": {
                "state": tts["state"],
                "model": tts["model_id"],
            },
            "whisper": {
                "state": "loaded" if whisper_service.is_loaded() else "unloaded",
            },
        }

    @app.get("/version")
    async def version():
        return {
            "server": SERVER_NAME,
            "version": VERSION,
            "qwen_tts": _pkg_version("qwen-tts"),
            "torch": getattr(torch, "__version__", None),
            "transformers": _pkg_version("transformers"),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        }

    @app.get("/")
    async def root():
        active = await ctx.models.get_active_model()
        return {
            "status": "ok",
            "current_model": active.model_id if active else None,
            "clone_active": ctx.voices.clone_active,
            "vram_available_gb": ctx.metrics.vram_available_gb(),
            "device": resolve_device(),
        }
