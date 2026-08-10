#!/usr/bin/env python3
"""Rutas de estado del sistema (nivel PUBLIC)."""

from fastapi import FastAPI

from config.settings import VERSION
from services.config_service import resolve_device


def create_system_routes(app: FastAPI, ctx):
    """Rutas de estado y salud del servidor."""

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/version")
    async def version():
        return {"version": VERSION}

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
