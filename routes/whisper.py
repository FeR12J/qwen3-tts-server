#!/usr/bin/env python3
"""Rutas de transcripción de audio con Whisper."""

import logging
import traceback
from typing import List, Optional

from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from security.auth import require_api_key
from services import whisper_service
from services.config_service import get_limits
from services.errors import APIError, InvalidAudioError
from services.gpu_management import prepare_for_whisper

logger = logging.getLogger("tts")

VALID_RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")
VALID_GRANULARITIES = ("segment", "word")


def _srt_time(seconds: float) -> str:
    """Formato de tiempo SRT: HH:MM:SS,mmm."""
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    """Formato de tiempo VTT: HH:MM:SS.mmm."""
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _subtitle_text(segments: list, fmt: str) -> str:
    """Convertir segmentos [{start, end, text}] a subtítulos SRT o VTT."""
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
    else:
        lines = []
    for i, seg in enumerate(segments, start=1):
        if fmt == "vtt":
            lines.append(f"{_vtt_time(seg['start'])} --> {_vtt_time(seg['end'])}")
        else:
            lines.append(str(i))
            lines.append(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def create_whisper_routes(app: FastAPI, ctx):
    """Rutas del servicio de transcripción."""

    @app.get("/transcribe/status")
    async def transcribe_status():
        return {"status": "ok", **whisper_service.status()}

    @app.post("/transcribe/load", dependencies=[Depends(require_api_key)])
    async def transcribe_load():
        """Cargar el modelo Whisper de forma explícita (sin transcribir).

        Libera primero la VRAM del modelo TTS si la configuración lo exige
        (unload_tts_for_whisper), igual que la carga lazy de /transcribe.
        """
        async with ctx.queue.model_lock():
            await prepare_for_whisper(
                ctx.models, ctx.voices, whisper_service, queue=ctx.queue
            )
            try:
                await whisper_service.load()
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except (HTTPException, APIError):
                raise
            except Exception as e:
                logger.error(f"Error cargando modelo de transcripción: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(
                    500, f"Error cargando modelo de transcripción: {str(e)}"
                )
        return {"status": "ok", **whisper_service.status()}

    @app.post("/transcribe", dependencies=[Depends(require_api_key)])
    async def transcribe_endpoint(
        audio: Optional[UploadFile] = File(None),
        language: Optional[str] = Form(None),
        timestamps: Optional[str] = Form(None),
    ):
        # Regla arquitectónica: validación SIEMPRE antes de adquirir el
        # inference_lock (la GPU nunca se bloquea validando input).
        if audio is None or not audio.filename:
            raise InvalidAudioError("Archivo de audio requerido (campo 'audio')")

        if timestamps is not None and timestamps not in whisper_service.VALID_TIMESTAMP_MODES:
            raise HTTPException(
                400,
                "timestamps inválido. Válidos: "
                + ", ".join(whisper_service.VALID_TIMESTAMP_MODES)
                + " (vacío = ajuste configurado)",
            )

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
            raise InvalidAudioError(str(e))

        async with ctx.queue.inference_lock():
            # Liberar VRAM del modelo TTS antes de cargar Whisper (si la config lo exige)
            await prepare_for_whisper(ctx.models, ctx.voices, whisper_service,
                                      queue=ctx.queue)

            try:
                result = await whisper_service.transcribe(
                    data, language, timestamps=timestamps
                )
                logger.info(f"Transcripción completada ({result['language']}, {result['duration_seconds']}s)")
                return {"status": "ok", **result}
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            except (HTTPException, APIError):
                raise
            except Exception as e:
                logger.error(f"Error en transcripción: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error transcribiendo audio: {str(e)}")

    @app.post("/tts/audio/transcriptions", dependencies=[Depends(require_api_key)])
    async def openai_transcriptions_endpoint(
        file: Optional[UploadFile] = File(None),
        model: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        timestamp_granularities: Optional[List[str]] = Form(None),
    ):
        """Transcripción compatible con la API de OpenAI (/audio/transcriptions).

        Multipart: `file` (obligatorio), `model` (se ignora: se usa el modelo
        configurado, p.ej. desde el panel), `language`, `response_format`
        (json, text, srt, vtt, verbose_json) y `timestamp_granularities`
        (segment, word). Usado por OpenWebUI con base URL
        http://host:8001/tts -> POST /tts/audio/transcriptions.
        """
        if file is None or not file.filename:
            raise InvalidAudioError("Archivo de audio requerido (campo 'file')")

        fmt = (response_format or "json").strip().lower()
        if fmt not in VALID_RESPONSE_FORMATS:
            raise HTTPException(
                400,
                "response_format inválido. Válidos: " + ", ".join(VALID_RESPONSE_FORMATS),
            )

        granularities = timestamp_granularities or []
        for g in granularities:
            if g not in VALID_GRANULARITIES:
                raise HTTPException(
                    400,
                    "timestamp_granularities inválido. Válidos: "
                    + ", ".join(VALID_GRANULARITIES),
                )

        if model:
            configured = whisper_service.status()["model"]
            if model != configured:
                logger.info(
                    f"OpenAI transcription: el cliente pidió '{model}' y se usa "
                    f"el configurado '{configured}'"
                )

        data = await file.read()
        sl = get_limits()
        try:
            ctx.audio.validate(
                data,
                max_bytes=sl.max_transcribe_audio_bytes,
                max_duration=sl.max_transcribe_duration_seconds,
                filename=file.filename,
                content_type=file.content_type,
                min_sample_rate=sl.min_sample_rate,
                max_sample_rate=sl.max_sample_rate,
                max_channels=sl.max_channels,
                decode=True,
            )
        except ValueError as e:
            raise InvalidAudioError(str(e))

        # Modo de marcas de tiempo según la respuesta pedida:
        #   - granularidades con "word"       -> palabras
        #   - segment/verbose_json/srt/vtt    -> segmentos
        #   - json/text sin granularidades    -> solo texto
        if "word" in granularities:
            mode = "word"
        elif "segment" in granularities or fmt in ("srt", "vtt", "verbose_json"):
            mode = "segment"
        else:
            mode = "off"

        async with ctx.queue.inference_lock():
            await prepare_for_whisper(
                ctx.models, ctx.voices, whisper_service, queue=ctx.queue
            )
            try:
                result = await whisper_service.transcribe(
                    data, language, timestamps=mode
                )
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            except (HTTPException, APIError):
                raise
            except Exception as e:
                logger.error(f"Error en transcripción: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error transcribiendo audio: {str(e)}")

        text = result["text"]
        if fmt == "text":
            return PlainTextResponse(text)
        if fmt == "srt":
            return PlainTextResponse(_subtitle_text(result.get("segments", []), "srt"))
        if fmt == "vtt":
            return PlainTextResponse(_subtitle_text(result.get("segments", []), "vtt"))
        if fmt == "verbose_json":
            body = {
                "text": text,
                "language": result["language"],
                "duration": result["duration_seconds"],
                "segments": result.get("segments", []),
            }
            if mode == "word":
                body["words"] = result.get("words", [])
            return body
        return {"text": text}

    @app.post("/transcribe/unload", dependencies=[Depends(require_api_key)])
    async def transcribe_unload():
        async with ctx.queue.model_lock():
            if not whisper_service.is_loaded():
                return {"status": "ok", "message": "El modelo de transcripción ya estaba descargado"}
            await whisper_service.unload()
            return {"status": "ok", "message": "Modelo de transcripción descargado y VRAM liberada"}
