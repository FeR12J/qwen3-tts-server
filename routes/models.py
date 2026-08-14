#!/usr/bin/env python3
"""Rutas de gestión de modelos."""

import logging
import traceback

from fastapi import FastAPI, Depends, HTTPException

from config.settings import settings
from security.auth import require_admin
from security.validation import validate_model_id
from services import model_downloader, whisper_service
from services.errors import APIError, ModelLoadingError, ModelNotLoadedError
from services.gpu_management import prepare_for_tts
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
                    ctx.models, ctx.voices, whisper_service,
                    restore_model=False, queue=ctx.queue,
                )
                info = await ctx.models.load_model(model_id)
                return {
                    "status": "ok",
                    "loaded_model": info.model_id,
                    "model_type": info.model_type,
                    "vram_available_gb": ctx.metrics.vram_available_gb(),
                }

            except FileNotFoundError as e:
                raise APIError("MODEL_NOT_FOUND", str(e), 404)
            except ValueError as e:
                raise APIError("INVALID_MODEL_ID", str(e), 400)
            except (HTTPException, APIError):
                raise
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
                logger.debug(traceback.format_exc())
                raise ModelLoadingError(f"Error cargando modelo: {str(e)}")

    @app.post("/model/unload", dependencies=[Depends(require_admin)])
    async def unload_model_endpoint(req_body: LoadModelRequest | None = None):
        model_id = req_body.model_id.strip() if req_body is not None else None
        async with ctx.queue.model_lock():
            if model_id is None:
                info = await ctx.models.get_active_model()
                if info is None:
                    return {"status": "ok", "message": "No hay modelo cargado"}
                model_id = info.model_id
            else:
                validate_model_id(model_id)
                if not ctx.models.is_loaded_model(model_id):
                    return {"status": "ok", "message": f"El modelo '{model_id}' no estaba cargado"}

            active = await ctx.models.get_active_model()
            was_active = active is not None and active.model_id == model_id

            await ctx.models.unload_model(model_id)
            if was_active:
                ctx.voices.unload_voice()

            logger.info(f"Modelo descargado: {model_id}")
            return {
                "status": "ok",
                "unloaded_model": model_id,
                "message": f"Modelo '{model_id}' descargado y VRAM liberada",
            }

    @app.post("/model/activate", dependencies=[Depends(require_admin)])
    async def activate_model_endpoint(req_body: LoadModelRequest):
        """Activar un modelo ya cargado en memoria (sin recargar ni resetear
        la voz clonada). Solo disponible si el modelo está READY."""
        model_id = req_body.model_id.strip()
        validate_model_id(model_id)
        async with ctx.queue.model_lock():
            if not ctx.models.is_loaded_model(model_id):
                raise ModelNotLoadedError(
                    f"El modelo '{model_id}' no está cargado en memoria: usa /model/load"
                )
            info = await ctx.models.switch_model(model_id)
            logger.info(f"Modelo activado: {info.model_id}")
            return {
                "status": "ok",
                "loaded_model": info.model_id,
                "model_type": info.model_type,
                "message": f"Modelo '{info.model_id}' activado",
            }

    @app.get("/models/status")
    @app.get("/tts/audio/models/status")
    async def models_status():
        """Estado de cada modelo local (para la tabla de gestión del panel)."""
        return {"models": ctx.models.list_models_status()}

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

    # -- Descarga de modelos soportados ------------------------------------

    @app.get("/models/download/status")
    async def models_download_status():
        """Estado de los modelos soportados (instalado / descargando / error)."""
        return {"models": model_downloader.list_status()}

    @app.post("/models/download", dependencies=[Depends(require_admin)])
    async def models_download(req_body: LoadModelRequest):
        """Descargar un modelo soportado (whitelist) a models/.

        Solo acepta los nombres de modelo de la lista establecida; nunca un
        repo_id arbitrario. La descarga corre en segundo plano (estado en
        /models/download/status) y solo una a la vez.
        """
        model_id = req_body.model_id.strip()
        validate_model_id(model_id)
        try:
            return await model_downloader.start_download(model_id)
        except APIError:
            raise
        except Exception as e:
            logger.error(f"Error iniciando descarga de '{model_id}': {e}")
            logger.debug(traceback.format_exc())
            raise APIError("DOWNLOAD_FAILED", f"Error iniciando la descarga: {str(e)}", 500)
