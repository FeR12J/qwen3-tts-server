#!/usr/bin/env python3
"""Rutas HTTP del servidor TTS."""

import logging
import traceback

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import Response

from security.auth import require_api_key
from security.permissions import require_model_loaded
from security.validation import validate_text, require_text
from services.config_service import get_runtime_config
from services.model_manager import GPUOutOfMemoryError
from schemas.tts import TTSRequest, TTSRequestOpenWebUI

logger = logging.getLogger("tts")


def create_tts_routes(app: FastAPI, ctx):
    """Rutas de generación TTS (no contienen lógica de inferencia)."""

    @app.post("/tts", dependencies=[Depends(require_api_key)])
    async def tts_endpoint(req_body: TTSRequest, req: Request):
        async with ctx.queue.inference_lock():
            rc = get_runtime_config()
            validate_text(req_body.text, rc["max_text_chars"])
            if rc.get("log_requests", True):
                ctx.metrics.log_request(req, req_body.text)
            await require_model_loaded(ctx.models)

            try:
                audio_bytes, _sr = await ctx.tts.generate(
                    text=req_body.text,
                    language=req_body.language,
                    speaker=req_body.speaker,
                    instruct=req_body.instruct,
                )
                return Response(audio_bytes, media_type="audio/wav")

            except HTTPException:
                raise
            except GPUOutOfMemoryError:
                raise
            except Exception as e:
                logger.error(f"Error en generación TTS: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error generando audio: {str(e)}")

    @app.post("/tts/play", dependencies=[Depends(require_api_key)])
    async def tts_play_endpoint(req_body: TTSRequest, req: Request):
        """Generar TTS y reproducirlo directamente en este equipo, esperando a que la reproducción anterior termine."""
        async with ctx.queue.inference_lock():
            rc = get_runtime_config()
            validate_text(req_body.text, rc["max_text_chars"])
            if rc.get("log_requests", True):
                ctx.metrics.log_request(req, req_body.text)
            await require_model_loaded(ctx.models)

            try:
                audio_bytes, sr = await ctx.tts.generate(
                    text=req_body.text,
                    language=req_body.language,
                    speaker=req_body.speaker,
                    instruct=req_body.instruct,
                )

                info = await ctx.audio.play(audio_bytes, sr, rc["playback_wait_timeout"])
                return {
                    "status": "ok",
                    "message": "Audio generado y reproduciéndose en este equipo",
                    **info,
                }

            except HTTPException:
                raise
            except GPUOutOfMemoryError:
                raise
            except Exception as e:
                logger.error(f"Error en reproducción TTS: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error generando o reproduciendo audio: {str(e)}")

    @app.post("/tts/audio/speech", dependencies=[Depends(require_api_key)])
    async def openwebui_tts(req_body: TTSRequestOpenWebUI, req: Request):
        async with ctx.queue.inference_lock():
            rc = get_runtime_config()
            text = req_body.text or req_body.input
            require_text(text)
            validate_text(text, rc["max_text_chars"])
            if rc.get("log_requests", True):
                ctx.metrics.log_request(req, text)
            await require_model_loaded(ctx.models)

            try:
                audio_bytes, _sr = await ctx.tts.generate(
                    text=text,
                    language=req_body.language,
                    speaker=req_body.speaker,
                    instruct=req_body.instruct,
                )
                logger.info("Generación completada (OpenWebUI)")
                return Response(audio_bytes, media_type="audio/wav")

            except HTTPException:
                raise
            except GPUOutOfMemoryError:
                raise
            except Exception as e:
                logger.error(f"Error en generación OpenWebUI: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error generando audio: {str(e)}")
