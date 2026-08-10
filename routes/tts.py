#!/usr/bin/env python3
"""Rutas HTTP del servidor TTS.

Todos los endpoints terminan en tts_service.synthesize(request):
aquí no hay lógica de generación, solo formato de respuesta.
"""

import logging
import traceback

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import Response

from security.auth import require_api_key
from services.config_service import get_runtime_config
from services.model_manager import GPUOutOfMemoryError
from services.tts_service import TTSValidationError
from schemas.tts import TTSRequest

logger = logging.getLogger("tts")


def _media_type(output_format: str) -> str:
    return "audio/wav" if output_format == "wav" else "audio/pcm"


def create_tts_routes(app: FastAPI, ctx):
    """Rutas de generación TTS (no contienen lógica de inferencia)."""

    @app.post("/tts/audio/speech", dependencies=[Depends(require_api_key)])
    @app.post("/tts", dependencies=[Depends(require_api_key)])
    async def tts_endpoint(req_body: TTSRequest, req: Request):
        """Generar audio (estándar y compatible OpenWebUI).

        Acepta `text` (o `input` para OpenWebUI) y todos los campos de
        TTSRequest. Devuelve WAV (o PCM si output_format="pcm").
        """
        try:
            result = await ctx.tts.synthesize(req_body, http_request=req)
        except (HTTPException, GPUOutOfMemoryError):
            raise
        except Exception as e:
            logger.error(f"Error en generación TTS: {e}")
            logger.debug(traceback.format_exc())
            raise HTTPException(500, f"Error generando audio: {str(e)}")
        return Response(result.audio, media_type=_media_type(req_body.output_format))

    @app.post("/tts/play", dependencies=[Depends(require_api_key)])
    async def tts_play_endpoint(req_body: TTSRequest, req: Request):
        """Generar TTS y reproducirlo directamente en este equipo, esperando
        a que la reproducción anterior termine."""
        try:
            result = await ctx.tts.synthesize(req_body, http_request=req)
        except (HTTPException, GPUOutOfMemoryError):
            raise
        except Exception as e:
            logger.error(f"Error en reproducción TTS: {e}")
            logger.debug(traceback.format_exc())
            raise HTTPException(500, f"Error generando o reproduciendo audio: {str(e)}")

        rc = get_runtime_config()
        info = await ctx.audio.play(result.audio, result.sample_rate, rc["playback_wait_timeout"])
        return {
            "status": "ok",
            "message": "Audio generado y reproduciéndose en este equipo",
            "model": result.model_id,
            **info,
        }
