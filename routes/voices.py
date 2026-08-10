#!/usr/bin/env python3
"""Rutas de gestión de voces."""

import logging
import traceback

from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile

from security.auth import require_api_key
from security.permissions import require_model_loaded, ensure_voice_cloning_supported
from security.validation import (
    validate_voice_name,
    validate_wav_upload,
    validate_audio_size,
)
from schemas.voices import LoadVoiceRequest
from services.gpu_management import prepare_for_tts
from services.model_manager import GPUOutOfMemoryError
from services import whisper_service

logger = logging.getLogger("tts")

# Tamaño máximo del audio subido para crear voces (50 MB)
MAX_VOICE_AUDIO_BYTES = 50 * 1024 * 1024


def create_voices_routes(app: FastAPI, ctx):
    """Rutas de carga, creación, descarga y listado de voces."""

    @app.post("/voice/load", dependencies=[Depends(require_api_key)])
    async def load_voice(req_body: LoadVoiceRequest):
        async with ctx.queue.inference_lock():
            await prepare_for_tts(ctx.models, ctx.voices, whisper_service)
            await require_model_loaded(ctx.models)
            voice_name = req_body.voice_name.strip()
            validate_voice_name(voice_name)
            await ensure_voice_cloning_supported(ctx.models)

            try:
                await ctx.voices.load_voice(voice_name)
                active = await ctx.models.get_active_model()
                return {
                    "status": "ok",
                    "voice": voice_name,
                    "model": active.model_id if active else None,
                    "message": f"Voz '{voice_name}' lista para usar",
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
                logger.error(f"Error creando voz clonada: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error creando voz clonada: {str(e)}")

    @app.post("/voice/create", dependencies=[Depends(require_api_key)])
    async def create_voice(
        voice_name: str = Form(...),
        text: str = Form(...),
        audio: UploadFile = File(...),
    ):
        """Crear una voz subiendo un WAV y su transcripción. La guarda y la clona."""
        async with ctx.queue.inference_lock():
            await prepare_for_tts(ctx.models, ctx.voices, whisper_service)
            await require_model_loaded(ctx.models)

            voice_name = voice_name.strip()
            validate_voice_name(voice_name)
            if not text.strip():
                raise HTTPException(400, "La transcripción no puede estar vacía")
            validate_wav_upload(audio)
            await ensure_voice_cloning_supported(ctx.models)

            data = await audio.read()
            validate_audio_size(data, MAX_VOICE_AUDIO_BYTES)

            try:
                await ctx.voices.create_voice(voice_name, text.strip(), data)
                active = await ctx.models.get_active_model()
                return {
                    "status": "ok",
                    "voice": voice_name,
                    "model": active.model_id if active else None,
                    "message": f"Voz '{voice_name}' creada, guardada y aplicada",
                }

            except HTTPException:
                raise
            except GPUOutOfMemoryError:
                raise
            except Exception as e:
                logger.error(f"Error creando voz clonada: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error creando voz clonada: {str(e)}")

    @app.post("/voice/unload", dependencies=[Depends(require_api_key)])
    async def unload_voice():
        async with ctx.queue.inference_lock():
            if not ctx.voices.unload_voice():
                return {
                    "status": "ok",
                    "message": "Voice cloning ya estaba desactivado",
                }
            return {
                "status": "ok",
                "message": "Voice cloning desactivado",
            }

    @app.get("/voices")
    @app.get("/tts/audio/voices")
    async def list_voices():
        try:
            voices = ctx.voices.list_voices()
            return {
                "available_voices": [v["name"] for v in voices],
                "clone_active": ctx.voices.clone_active,
                "voices_detail": voices,
            }
        except Exception as e:
            logger.error(f"Error listando voces: {e}")
            raise HTTPException(500, f"Error leyendo directorio de voces: {str(e)}")
