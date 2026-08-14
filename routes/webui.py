#!/usr/bin/env python3
"""Panel web de administración y documentación de la API."""

import os
import logging

from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse

from config.settings import settings
from security.auth import require_admin
from schemas.system import ConfigUpdate
from security.validation import validate_config_update
from services import config_service, whisper_service
from utils.gpu import list_devices

logger = logging.getLogger("tts")

WEBUI_DIR = settings.paths.webui_dir


def create_webui_routes(app: FastAPI, ctx):
    """Rutas del panel web y de administración."""

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(
            os.path.join(WEBUI_DIR, "favicon.svg"),
            media_type="image/svg+xml",
        )

    @app.get("/webui")
    async def webui_panel():
        return FileResponse(os.path.join(WEBUI_DIR, "panel.html"))

    @app.get("/webui/docs")
    async def webui_docs():
        return FileResponse(os.path.join(WEBUI_DIR, "docs.html"))

    @app.get("/webui/api/config", dependencies=[Depends(require_admin)])
    async def get_config():
        return config_service.get_runtime_config()

    @app.post("/webui/api/config", dependencies=[Depends(require_admin)])
    async def set_config(body: ConfigUpdate):
        changes = body.model_dump(exclude_none=True)
        validate_config_update(changes, config_service)

        rc = config_service.update_runtime_config(changes)
        config_service.apply_log_level()

        # Si cambió el modelo Whisper y está cargado, descargarlo: libera la
        # VRAM al instante y la próxima transcripción cargará el modelo
        # elegido (bajo model_lock para no cortar una transcripción activa).
        if "whisper_model" in changes and whisper_service.is_loaded():
            async with ctx.queue.model_lock():
                await whisper_service.unload()
            logger.info(
                f"Modelo Whisper cambiado a '{changes['whisper_model']}': "
                "el anterior se descargó, se recargará en la próxima transcripción"
            )

        logger.info("Configuración en tiempo de ejecución actualizada desde el panel")
        return rc

    @app.get("/webui/api/devices")
    async def get_devices():
        """Listar GPUs disponibles y el dispositivo actual."""
        devices = list_devices()
        devices["current"] = config_service.get_runtime_config().get("device", "auto")
        devices["resolved"] = config_service.resolve_device()
        return devices
