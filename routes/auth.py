#!/usr/bin/env python3
"""Rutas de administración de claves API."""

import logging

from fastapi import FastAPI, HTTPException

from schemas.system import ApiKeyCreate
from services import apikey_service

logger = logging.getLogger("tts")


def create_auth_routes(app: FastAPI, ctx):
    """Rutas de gestión de claves API (panel de administración)."""

    @app.get("/webui/api/apikeys")
    async def get_api_keys():
        return {
            "enabled": apikey_service.global_enabled(),
            "keys": apikey_service.list_keys(),
        }

    @app.post("/webui/api/apikeys")
    async def create_api_key(body: ApiKeyCreate):
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "El nombre de la clave no puede estar vacío")
        try:
            return apikey_service.create_key(name)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/webui/api/apikeys/{key_id}")
    async def delete_api_key(key_id: str):
        try:
            apikey_service.delete_key(key_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {"status": "ok"}

    @app.post("/webui/api/apikeys/{key_id}/toggle")
    async def toggle_api_key(key_id: str):
        try:
            return apikey_service.toggle_key(key_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
