#!/usr/bin/env python3
"""Autenticación y niveles de protección de la API.

Tres niveles:
  - PUBLIC: /health, /version, estado y listados (sin autenticación).
  - PROTECTED (require_api_key): /tts/*, /transcribe. Exige clave válida
    solo si la exigencia global de claves está activada.
  - ADMIN (require_admin): /model/*, /voice/*, /apikeys/*, /config/*.
    Exige siempre una clave API válida, esté o no activada la exigencia
    global, para que las operaciones administrativas no queden accesibles
    sin autenticación. Modo bootstrap: si aún no existe ninguna clave,
    permite el acceso (para poder crear la primera clave).
"""

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
    """Dependencia FastAPI (PROTECTED): exige clave válida si están activadas."""
    if not apikey_service.global_enabled():
        return
    key = _extract_key(request)
    if not key or not apikey_service.verify_key(key):
        raise HTTPException(401, "API key inválida o no proporcionada")


async def require_admin(request: Request):
    """Dependencia FastAPI (ADMIN): exige siempre una clave API válida.

    Bootstrap: si todavía no existe ninguna clave, se permite el acceso
    para poder crear la primera clave desde el panel.
    """
    if apikey_service.count_keys() == 0:
        return
    key = _extract_key(request)
    if not key or not apikey_service.verify_key(key):
        raise HTTPException(401, "API key de administración inválida o no proporcionada")
