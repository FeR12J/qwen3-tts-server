#!/usr/bin/env python3
"""Rutas de gestión de modelos."""

import logging
import traceback

from fastapi import FastAPI, Depends, HTTPException

from config.settings import settings
from security.auth import require_admin
from security.validation import validate_model_id
from services.gpu_management import prepare_for_tts
from services.model_manager import GPUOutOfMemoryError
from services import whisper_service
from schemas.models import LoadModelRequest

logger = logging.getLogger("tts")


def create_models_routes(app: FastAPI, ctx):
    """Rutas de carga/descarga y listado de modelos."""

    @app.post("/model/load", dependencies=[Depends(require_admin)])
    async def load_model_endpoint(req_body: LoadModelRequest):
        async with ctx.queue.model_lock():
            # Resetear voice cloning al cargar nuevo modelo
            ctx.voices.unload_voice()

            model_id = req_body.model_id.strip()
            validate_model_id(model_id)

            try:
                # Liberar VRAM que ocupe Whisper antes de cargar el modelo TTS
                await prepare_for_tts(
                    ctx.models, ctx.voices, whisper_service, restore_model=False
                )
                info = await ctx.models.load_model(model_id)
                return {
                    "status": "ok",
                    "loaded_model": info.model_id,
                    "model_type": info.model_type,
                    "vram_available_gb": ctx.metrics.vram_available_gb(),
                }

            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            except HTTPException:
                raise
            except GPUOutOfMemoryError:
                raise
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error cargando modelo: {str(e)}")

    @app.post("/model/unload", dependencies=[Depends(require_admin)])
    async def unload_model_endpoint():
        async with ctx.queue.model_lock():
            info = await ctx.models.get_active_model()
            if info is None:
                return {"status": "ok", "message": "No hay modelo cargado"}

            await ctx.models.unload_model(info.model_id)
            ctx.voices.unload_voice()

            logger.info(f"Modelo descargado: {info.model_id}")
            return {
                "status": "ok",
                "unloaded_model": info.model_id,
                "message": f"Modelo '{info.model_id}' descargado y VRAM liberada",
            }

    @app.get("/model/status")
    async def model_status():
        """Estado del modelo activo (states: unloaded, loading, ready, generating, unloading, error)."""
        return await ctx.models.get_model_status()

    @app.get("/models")
    @app.get("/tts/audio/models")
    async def list_models():
        try:
            active = await ctx.models.get_active_model()
            return {
                "available_models": ctx.models.list_local_models(),
                "current_model": active.model_id if active else None,
                "models_dir": settings.paths.models_dir,
            }
        except Exception as e:
            logger.error(f"Error listando modelos: {e}")
            raise HTTPException(500, f"Error leyendo directorio de modelos: {str(e)}")
