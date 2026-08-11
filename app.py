#!/usr/bin/env python3
"""Aplicación principal del servidor TTS."""

import os
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from utils.logging import setup_logging
from utils.gpu import get_vram_available
from services.config_service import load_runtime_config, apply_log_level
from services.queue_service import QueueService
from services.model_manager import ModelManager, GPUOutOfMemoryError, GPU_OOM_MESSAGE
from services.voice_manager import VoiceManager
from services.audio_service import AudioService
from services.whisper_service import configure as configure_whisper
from services.metrics_service import MetricsService
from services.tts_service import TTSService, TTSValidationError
from routes import (
    create_tts_routes,
    create_models_routes,
    create_voices_routes,
    create_system_routes,
    create_whisper_routes,
    create_auth_routes,
    create_webui_routes,
)

setup_logging()
logger = logging.getLogger("tts")


@dataclass
class AppContext:
    """Contenedor de dependencias de la aplicación."""
    config: dict
    queue: QueueService
    models: ModelManager
    voices: VoiceManager
    audio: AudioService
    metrics: MetricsService
    tts: TTSService


def build_context() -> AppContext:
    """Construir e interconectar todos los servicios."""
    queue = QueueService(settings.queue.max_parallel_inference)
    models = ModelManager()
    voices = VoiceManager(models)
    audio = AudioService(settings, queue)
    metrics = MetricsService(settings)
    tts = TTSService(settings, queue, models, voices, audio, metrics)
    configure_whisper(audio)
    return AppContext(
        config=settings,
        queue=queue,
        models=models,
        voices=voices,
        audio=audio,
        metrics=metrics,
        tts=tts,
    )


def register_routes(app: FastAPI, ctx: AppContext):
    """Registrar todas las rutas HTTP."""
    create_tts_routes(app, ctx)
    create_models_routes(app, ctx)
    create_voices_routes(app, ctx)
    create_system_routes(app, ctx)
    create_whisper_routes(app, ctx)
    create_auth_routes(app, ctx)
    create_webui_routes(app, ctx)


# Intervalo de la limpieza periódica de audios generados (1 hora)
AUDIO_CLEANUP_INTERVAL_SECONDS = 60 * 60


async def audio_cleanup_loop(ctx: AppContext):
    """Limpieza automática periódica del directorio de audios.

    Elimina los audios generados con más de storage.generated_audio_ttl_hours
    horas, para que el directorio no crezca indefinidamente.
    """
    while True:
        await asyncio.sleep(AUDIO_CLEANUP_INTERVAL_SECONDS)
        try:
            removed = ctx.audio.cleanup_old(settings.storage.generated_audio_ttl_hours * 3600)
            if removed:
                logger.info(f"Limpieza periódica: {removed} audio(s) antiguo(s) eliminados")
        except Exception as e:
            logger.warning(f"Error en la limpieza periódica de audios: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación."""
    ctx = build_context()
    load_runtime_config()
    apply_log_level()
    register_routes(app, ctx)
    cleanup_task = asyncio.create_task(audio_cleanup_loop(ctx))
    try:
        await startup_procedure(ctx)
        yield
    finally:
        cleanup_task.cancel()


# Inicializar aplicación FastAPI
app = FastAPI(title="Qwen3-TTS API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    **settings.cors.model_dump(),
)


@app.exception_handler(GPUOutOfMemoryError)
async def gpu_out_of_memory_handler(request: Request, exc: GPUOutOfMemoryError):
    """Respuesta controlada ante CUDA OOM: nunca se filtran detalles internos."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "GPU_OUT_OF_MEMORY",
                "message": GPU_OOM_MESSAGE,
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(TTSValidationError)
async def tts_validation_handler(request: Request, exc: TTSValidationError):
    """Error de validación de una petición TTS (400, en formato estándar)."""
    return JSONResponse(
        status_code=400,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


async def startup_procedure(ctx: AppContext):
    """Procedimiento de inicialización del servidor."""

    print("\n" + "." * 60)
    print("Iniciando qwen3-tts server")
    print("." * 60)

    # Limpiar audios generados más antiguos que el TTL configurado
    ctx.audio.cleanup_old(settings.storage.generated_audio_ttl_hours * 3600)

    # Modelos locales
    local_models = ctx.models.list_local_models()

    print(f"\n Modelos locales disponibles ({len(local_models)}):")
    for i, m in enumerate(local_models, 1):
        print(f"   {i}. {m}")

    # Seleccionar modelo por defecto (configurable, fallback al primero disponible)
    if len(local_models) > 0:
        selected_model = None
        if settings.tts.default_model and settings.tts.default_model in local_models:
            selected_model = settings.tts.default_model
        else:
            selected_model = local_models[0]
            if settings.tts.default_model:
                print(f"\nModelo por defecto '{settings.tts.default_model}' no encontrado, usando '{selected_model}'")

        print(f"\nModelo seleccionado por defecto: {selected_model}")

        try:
            async with ctx.queue.model_lock():
                info = await ctx.models.load_model(selected_model)
            print(f"   Tipo: {info.model_type}")
            print(f"   VRAM disponible: {get_vram_available()} GB")
        except Exception as e:
            print(f"\nError cargando modelo por defecto: {e}")
    else:
        print("\nNo hay modelos disponibles. El servidor funcionará sin modelo inicial.")

    # Voces locales
    local_voices = ctx.voices.list()

    print(f"\nVoces locales disponibles ({len(local_voices)}):")
    for i, v in enumerate(local_voices, 1):
        status = "OK" if v["valid"] else ("KO" if not v["has_reference_audio"] else "!?")
        print(f"   {i}. {v['name']} (id: {v['id']}) {status}")

    # Intentar clonar voz por defecto si hay modelos y voces disponibles
    if len(local_models) > 0 and len(local_voices) >= 1 and ctx.models.is_loaded():

        selected_voice = None
        if settings.tts.default_voice and any(
            v["name"] == settings.tts.default_voice for v in local_voices
        ):
            selected_voice = settings.tts.default_voice
        else:
            selected_voice = local_voices[0]["name"]

        print(f"\nIntentando clonar voz por defecto: {selected_voice}")

        try:
            async with ctx.queue.inference_lock():
                await ctx.voices.load_voice(selected_voice)
            print(f"Voz '{selected_voice}' clonada correctamente y aplicada por defecto")
        except Exception as e:
            print(f"Error creando voz clonada (continuar sin voice cloning): {e}")

    print("\n" + "." * 30)
    print("Escuchando...")
    print("." * 30 + "\n")
    print(f"INFO:     WebUI disponible en: http://localhost:{settings.server.port}/webui")
    print(f"INFO:     Documentación de la API: http://localhost:{settings.server.port}/webui/docs\n")


# Punto de entrada principal
if __name__ == "__main__":
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)
