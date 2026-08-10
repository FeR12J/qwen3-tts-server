#!/usr/bin/env python3
"""Rutas de transcripción de audio con Whisper."""

import asyncio
import logging
import traceback
from typing import Optional

from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile

from config.settings import CONFIG
from security.auth import require_api_key
from security.validation import validate_audio_size
from services import whisper_service

logger = logging.getLogger("tts")

# Tamaño máximo del audio a transcribir (100 MB)
MAX_TRANSCRIBE_AUDIO_BYTES = 100 * 1024 * 1024


def create_whisper_routes(app: FastAPI, ctx):
    """Rutas del servicio de transcripción."""

    @app.get("/transcribe/status")
    async def transcribe_status():
        return {
            "status": "ok",
            "model_loaded": whisper_service.is_loaded(),
            "model": CONFIG.get("whisper_model", "whisper-large-v3"),
            "device": whisper_service.get_device(),
        }

    @app.post("/transcribe", dependencies=[Depends(require_api_key)])
    async def transcribe_endpoint(
        audio: Optional[UploadFile] = File(None),
        language: Optional[str] = Form(None),
    ):
        async with ctx.queue.infer():
            if audio is None or not audio.filename:
                raise HTTPException(400, "Archivo de audio requerido (campo 'audio')")

            data = await audio.read()
            validate_audio_size(data, MAX_TRANSCRIBE_AUDIO_BYTES)

            try:
                result = await asyncio.to_thread(whisper_service.transcribe, data, language)
                logger.info(f"Transcripción completada ({result['language']}, {result['duration_seconds']}s)")
                return {"status": "ok", **result}
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error en transcripción: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error transcribiendo audio: {str(e)}")

    @app.post("/transcribe/unload", dependencies=[Depends(require_api_key)])
    async def transcribe_unload():
        async with ctx.queue.infer():
            if not whisper_service.is_loaded():
                return {"status": "ok", "message": "El modelo de transcripción ya estaba descargado"}
            await asyncio.to_thread(whisper_service.unload)
            return {"status": "ok", "message": "Modelo de transcripción descargado y VRAM liberada"}
