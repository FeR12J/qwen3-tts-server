#!/usr/bin/env python3
"""Rutas de transcripción de audio con Whisper."""

import logging
import traceback
from typing import Optional

from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile

from security.auth import require_api_key
from services import whisper_service
from services.config_service import get_limits
from services.gpu_management import prepare_for_whisper
from services.model_manager import GPUOutOfMemoryError

logger = logging.getLogger("tts")


def create_whisper_routes(app: FastAPI, ctx):
    """Rutas del servicio de transcripción."""

    @app.get("/transcribe/status")
    async def transcribe_status():
        return {"status": "ok", **whisper_service.status()}

    @app.post("/transcribe", dependencies=[Depends(require_api_key)])
    async def transcribe_endpoint(
        audio: Optional[UploadFile] = File(None),
        language: Optional[str] = Form(None),
    ):
        # Regla arquitectónica: validación SIEMPRE antes de adquirir el
        # inference_lock (la GPU nunca se bloquea validando input).
        if audio is None or not audio.filename:
            raise HTTPException(400, "Archivo de audio requerido (campo 'audio')")

        data = await audio.read()
        sl = get_limits()
        try:
            ctx.audio.validate(
                data,
                max_bytes=sl.max_transcribe_audio_bytes,
                max_duration=sl.max_transcribe_duration_seconds,
                filename=audio.filename,
                content_type=audio.content_type,
                min_sample_rate=sl.min_sample_rate,
                max_sample_rate=sl.max_sample_rate,
                max_channels=sl.max_channels,
                decode=True,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        async with ctx.queue.inference_lock():
            # Liberar VRAM del modelo TTS antes de cargar Whisper (si la config lo exige)
            await prepare_for_whisper(ctx.models, ctx.voices, whisper_service)

            try:
                result = await whisper_service.transcribe(data, language)
                logger.info(f"Transcripción completada ({result['language']}, {result['duration_seconds']}s)")
                return {"status": "ok", **result}
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            except HTTPException:
                raise
            except GPUOutOfMemoryError:
                raise
            except Exception as e:
                logger.error(f"Error en transcripción: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error transcribiendo audio: {str(e)}")

    @app.post("/transcribe/unload", dependencies=[Depends(require_api_key)])
    async def transcribe_unload():
        async with ctx.queue.model_lock():
            if not whisper_service.is_loaded():
                return {"status": "ok", "message": "El modelo de transcripción ya estaba descargado"}
            await whisper_service.unload()
            return {"status": "ok", "message": "Modelo de transcripción descargado y VRAM liberada"}
