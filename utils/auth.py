#!/usr/bin/env python3
"""Autenticación por clave API."""

from fastapi import Request, HTTPException

from services import apikey_service


def _extract_key(request: Request):
    """Extraer la clave API de la cabecera X-API-Key o Authorization: Bearer."""
    key = request.headers.get("x-api-key")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    return key or None


async def require_api_key(request: Request):
    """Dependencia FastAPI: exige clave API válida si están activadas."""
    if not apikey_service.global_enabled():
        return
    key = _extract_key(request)
    if not key or not apikey_service.verify_key(key):
        raise HTTPException(401, "API key inválida o no proporcionada")
