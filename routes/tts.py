#!/usr/bin/env python3
"""Rutas HTTP del servidor TTS.

Todos los endpoints terminan en tts_service.synthesize(request) o en el
generador tts_service.stream_synthesize(): aquí no hay lógica de generación,
solo formato de respuesta.
"""

import logging
import traceback

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from security.auth import require_api_key
from services.audio_service import TTS_SAMPLE_RATE
from services.config_service import get_runtime_config
from services.model_manager import GPUOutOfMemoryError
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

    @app.post("/tts/stream", dependencies=[Depends(require_api_key)])
    async def tts_stream_endpoint(req_body: TTSRequest, req: Request):
        """Generar TTS en streaming real: audio por frases en cuanto terminan.

        - output_format="wav": WAV streaming (cabecera con tamaño indeterminado
          + chunks PCM). Reproducible con `curl | ffplay -` o `curl > archivo`.
        - output_format="pcm": PCM 16-bit LE; la frecuencia se indica en la
          cabecera X-Audio-Rate.

        El primer byte llega cuando acaba la primera frase, no al terminar de
        generar el texto completo. La validación (400) ocurre antes de abrir
        el stream, en tts_service.stream_plan(). Se puede desactivar desde la
        configuración en tiempo de ejecución (streaming_enabled).
        """
        if not get_runtime_config().get("streaming_enabled", True):
            raise HTTPException(
                404,
                detail={"error": {
                    "code": "STREAMING_DISABLED",
                    "message": "El streaming TTS está desactivado en la configuración "
                               "del servidor (streaming_enabled).",
                }},
            )
        try:
            plan = await ctx.tts.stream_plan(req_body)
        except (HTTPException, GPUOutOfMemoryError):
            raise
        except Exception as e:
            logger.error(f"Error preparando streaming: {e}")
            logger.debug(traceback.format_exc())
            raise HTTPException(500, f"Error preparando streaming: {str(e)}")

        async def gen():
            first = True
            try:
                async for chunk in ctx.tts.stream_synthesize(req_body, plan, http_request=req):
                    if req_body.output_format == "wav":
                        if first:
                            yield ctx.audio.wav_stream_header(chunk.sample_rate) + chunk.audio
                            first = False
                        else:
                            yield chunk.audio
                    else:
                        yield chunk.audio
            except Exception as e:
                logger.error(f"Error durante streaming: {e}")
                logger.debug(traceback.format_exc())
                raise

        if req_body.output_format == "wav":
            return StreamingResponse(gen(), media_type="audio/wav")
        return StreamingResponse(
            gen(),
            media_type="audio/L16",
            headers={
                "X-Audio-Rate": str(TTS_SAMPLE_RATE),
                "X-Audio-Channels": "1",
                "X-Audio-Bits": "16",
            },
        )

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
