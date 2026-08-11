#!/usr/bin/env python3
"""Rutas de gestión de voces (CRUD) y voice cloning."""

import logging
import traceback
from typing import Optional

from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile

from security.auth import require_admin
from security.permissions import require_model_loaded, ensure_voice_cloning_supported
from security.validation import (
    reject_path_traversal,
    validate_voice_name,
)
from schemas.voices import LoadVoiceRequest
from services.audio_service import SPEECH_SAMPLE_RATE, AudioService
from services.config_service import get_limits, get_runtime
from services.gpu_management import prepare_for_tts
from services.model_manager import GPUOutOfMemoryError
from services import whisper_service

logger = logging.getLogger("tts")


def create_voices_routes(app: FastAPI, ctx):
    """Rutas de gestión de voces: /voice/load, /voice/create, /voice/unload,
    /voices (list), /voices/{id} (get/update/delete)."""

    def _validation_error(e) -> HTTPException:
        """Traducir errores de validación/audio a 400."""
        return HTTPException(400, str(e))

    async def _read_and_validate_audio(audio: UploadFile) -> bytes:
        """Leer, validar y canonicalizar el audio subido (WAV 16 kHz mono
        normalizado). Lanza HTTPException 400 si no es válido."""
        data = await audio.read()
        sl = get_limits()
        try:
            info = ctx.audio.validate(
                data,
                max_bytes=sl.max_voice_audio_bytes,
                max_duration=sl.max_voice_audio_duration_seconds,
                filename=audio.filename,
                content_type=audio.content_type,
                formats=AudioService.FORMATS,
                min_sample_rate=sl.min_sample_rate,
                max_sample_rate=sl.max_sample_rate,
                max_channels=sl.max_channels,
                decode=True,
            )
        except ValueError as e:
            raise _validation_error(e)
        # Reutilizar el numpy array de validate(): evita decodificar dos veces.
        wav, sr = ctx.audio.prepare(info.samples, info.sample_rate, target_sr=SPEECH_SAMPLE_RATE)
        # Normalización del pico a normalization_dbfs (configurable; puede
        # desactivarse con normalize_reference_audio para preservar la
        # dinámica original del archivo).
        if get_runtime().normalize_reference_audio:
            wav = ctx.audio.normalize(wav)
        return ctx.audio.convert(wav, sr, "wav")

    @app.post("/voice/load", dependencies=[Depends(require_admin)])
    async def load_voice(req_body: LoadVoiceRequest):
        # Validación sin GPU: no mantener ocupado el inference_lock.
        voice_name = req_body.voice_name.strip()
        reject_path_traversal(voice_name, "voice_name")
        validate_voice_name(voice_name)

        async with ctx.queue.inference_lock():
            await prepare_for_tts(ctx.models, ctx.voices, whisper_service)
            await require_model_loaded(ctx.models)
            await ensure_voice_cloning_supported(ctx.models)

            try:
                resolved = await ctx.voices.load_voice(voice_name)
                active = await ctx.models.get_active_model()
                return {
                    "status": "ok",
                    "voice": resolved,
                    "model": active.model_id if active else None,
                    "message": f"Voz '{resolved}' lista para usar",
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

    @app.post("/voice/create", dependencies=[Depends(require_admin)])
    async def create_voice(
        voice_name: str = Form(...),
        text: str = Form(...),
        audio: UploadFile = File(...),
        language: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
    ):
        """Crear una voz subiendo un audio (wav, mp3, flac, ogg, m4a), su
        transcripción y metadata (name, language, description).

        El id lo genera el servidor (``voice_<hex>``); el nombre es solo
        metadata. El audio se normaliza y guarda como reference.wav (16 kHz
        mono), la transcripción como reference.txt y los metadatos en
        metadata.json. La voz se aplica como clon activo.
        """
        # Validaciones sin GPU: no mantener ocupado el inference_lock
        # mientras se valida/decodifica el audio subido.
        voice_name = voice_name.strip()
        reject_path_traversal(voice_name, "voice_name")
        validate_voice_name(voice_name)
        if not text.strip():
            raise HTTPException(400, "La transcripción no puede estar vacía")
        wav_bytes = await _read_and_validate_audio(audio)

        # Sección crítica (GPU): cargar modelo y guardar/aplicar la voz.
        async with ctx.queue.inference_lock():
            await prepare_for_tts(ctx.models, ctx.voices, whisper_service)
            await require_model_loaded(ctx.models)
            await ensure_voice_cloning_supported(ctx.models)

            try:
                created_id = await ctx.voices.create(
                    name=voice_name,
                    text=text.strip(),
                    audio_bytes=wav_bytes,
                    language=language,
                    description=description,
                )
                active = await ctx.models.get_active_model()
                return {
                    "status": "ok",
                    "voice": created_id,
                    "voice_metadata": ctx.voices.get(created_id),
                    "model": active.model_id if active else None,
                    "message": f"Voz '{created_id}' creada, guardada y aplicada",
                }

            except HTTPException:
                raise
            except GPUOutOfMemoryError:
                raise
            except ValueError as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                logger.error(f"Error creando voz clonada: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error creando voz clonada: {str(e)}")

    @app.get("/voices/{voice_id}", dependencies=[Depends(require_admin)])
    async def get_voice(voice_id: str):
        """Metadata de una voz (id, name, language, description, ...)."""
        reject_path_traversal(voice_id, "voice_id")
        try:
            return {"status": "ok", "voice": ctx.voices.get(voice_id)}
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            logger.error(f"Error obteniendo voz '{voice_id}': {e}")
            raise HTTPException(500, f"Error obteniendo voz: {str(e)}")

    @app.patch("/voices/{voice_id}", dependencies=[Depends(require_admin)])
    async def update_voice(
        voice_id: str,
        name: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        text: Optional[str] = Form(None),
        audio: Optional[UploadFile] = File(None),
    ):
        """Actualizar metadata y/o archivos de referencia de una voz.

        Solo se actualizan los campos enviados. Si se cambia el audio o la
        transcripción y la voz es el clon activo, se regenera el prompt.
        """
        # Validaciones sin GPU: no mantener ocupado el inference_lock.
        reject_path_traversal(voice_id, "voice_id")
        if name is not None:
            name = name.strip()
            reject_path_traversal(name, "name")
            validate_voice_name(name)
        if text is not None and not text.strip():
            raise HTTPException(400, "La transcripción no puede estar vacía")
        wav_bytes = None
        if audio is not None:
            wav_bytes = await _read_and_validate_audio(audio)

        # Sección crítica (GPU): cargar modelo y actualizar/aplicar la voz.
        async with ctx.queue.inference_lock():
            if audio is not None:
                await require_model_loaded(ctx.models)
                await ensure_voice_cloning_supported(ctx.models)

            try:
                updated = await ctx.voices.update(
                    voice_id,
                    name=name,
                    language=language,
                    description=description,
                    text=text.strip() if text else None,
                    audio_bytes=wav_bytes,
                )
                return {"status": "ok", "voice": updated}
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            except HTTPException:
                raise
            except GPUOutOfMemoryError:
                raise
            except Exception as e:
                logger.error(f"Error actualizando voz '{voice_id}': {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error actualizando voz: {str(e)}")

    @app.delete("/voices/{voice_id}", dependencies=[Depends(require_admin)])
    async def delete_voice(voice_id: str):
        """Eliminar una voz (directorio completo)."""
        reject_path_traversal(voice_id, "voice_id")
        try:
            existed = ctx.voices.delete(voice_id)
        except Exception as e:
            logger.error(f"Error eliminando voz '{voice_id}': {e}")
            raise HTTPException(500, f"Error eliminando voz: {str(e)}")
        if not existed:
            raise HTTPException(404, f"Voz '{voice_id}' no encontrada")
        return {
            "status": "ok",
            "message": f"Voz '{voice_id}' eliminada",
            "clone_active": ctx.voices.clone_active,
        }

    @app.post("/voice/unload", dependencies=[Depends(require_admin)])
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
            voices = ctx.voices.list()
            return {
                "available_voices": [v["name"] for v in voices],
                "available_voice_ids": [v["id"] for v in voices],
                "clone_active": ctx.voices.clone_active,
                "clone_voice_id": ctx.voices.active_voice_id,
                "voices_detail": voices,
            }
        except Exception as e:
            logger.error(f"Error listando voces: {e}")
            raise HTTPException(500, f"Error leyendo directorio de voces: {str(e)}")
