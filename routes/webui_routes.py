#!/usr/bin/env python3
"""Panel web de administración y documentación de la API."""

import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch

from config.settings import BASE_DIR
from services import config_service, apikey_service

logger = logging.getLogger("tts")

WEBUI_DIR = os.path.join(BASE_DIR, "webui")


class ConfigUpdate(BaseModel):
    """Campos de configuración en tiempo de ejecución actualizables."""
    max_text_chars: Optional[int] = None
    playback_wait_timeout: Optional[int] = None
    def_language: Optional[str] = None
    def_voice: Optional[str] = None
    def_instruct: Optional[str] = None
    log_level: Optional[str] = None
    log_requests: Optional[bool] = None
    api_keys_enabled: Optional[bool] = None
    device: Optional[str] = None


class ApiKeyCreate(BaseModel):
    name: str


def create_webui_routes(app: FastAPI, model_registry, current_model_id_var, clone_prompt_var):
    """Crear las rutas del panel web y de administración."""

    @app.get("/webui")
    async def webui_panel():
        return FileResponse(os.path.join(WEBUI_DIR, "panel.html"))

    @app.get("/webui/docs")
    async def webui_docs():
        return FileResponse(os.path.join(WEBUI_DIR, "docs.html"))

    @app.get("/webui/api/config")
    async def get_config():
        return config_service.get_runtime_config()

    @app.post("/webui/api/config")
    async def set_config(body: ConfigUpdate):
        changes = body.model_dump(exclude_none=True)

        if "max_text_chars" in changes and changes["max_text_chars"] is not None and changes["max_text_chars"] <= 0:
            raise HTTPException(400, "max_text_chars debe ser mayor que 0")
        if "playback_wait_timeout" in changes and changes["playback_wait_timeout"] is not None and changes["playback_wait_timeout"] <= 0:
            raise HTTPException(400, "playback_wait_timeout debe ser mayor que 0")
        if "log_level" in changes and changes["log_level"] not in config_service.VALID_LOG_LEVELS:
            raise HTTPException(400, f"log_level inválido. Válidos: {', '.join(config_service.VALID_LOG_LEVELS)}")
        if "device" in changes and not config_service.validate_device(changes["device"]):
            raise HTTPException(400, "device inválido. Válidos: auto, cpu o cuda:N con N dentro del rango de GPUs")

        rc = config_service.update_runtime_config(changes)
        config_service.apply_log_level()
        logger.info("Configuración en tiempo de ejecución actualizada desde el panel")
        return rc

    @app.get("/webui/api/devices")
    async def get_devices():
        """Listar GPUs disponibles y el dispositivo actual."""
        devices = []
        cuda_available = torch.cuda.is_available()
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
            "current": config_service.get_runtime_config().get("device", "auto"),
            "resolved": config_service.resolve_device()
        }

    @app.get("/webui/api/apikeys")
    async def get_api_keys():
        return {
            "enabled": apikey_service.global_enabled(),
            "keys": apikey_service.list_keys()
        }

    @app.post("/webui/api/apikeys")
    async def create_api_key(body: ApiKeyCreate):
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "El nombre de la clave no puede estar vacío")
        return apikey_service.create_key(name)

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
