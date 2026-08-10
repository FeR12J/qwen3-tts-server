#!/usr/bin/env python3
"""Permisos de acceso a los servicios del servidor."""

from fastapi import HTTPException

NOT_LOADED_MSG = "No hay modelo cargado. Usa /model/load primero."


async def require_model_loaded(model_manager):
    """Exigir que haya un modelo activo cargado."""
    if await model_manager.get_active_model() is None:
        raise HTTPException(400, NOT_LOADED_MSG)


async def ensure_voice_cloning_supported(model_manager):
    """Exigir que el modelo activo soporte voice cloning (tipo base/unknown)."""
    info = await model_manager.get_active_model()
    if info is None:
        raise HTTPException(400, NOT_LOADED_MSG)
    if info.model_type in ("base", "unknown"):
        return
    raise HTTPException(
        400,
        f"El modelo actual ({info.model_id}) no soporta voice cloning "
        f"(tipo: {info.model_type})",
    )
