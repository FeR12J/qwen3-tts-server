#!/usr/bin/env python3
"""Aplicación principal del servidor TTS."""

import os
import asyncio
import gc
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
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
from services import whisper_service
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
    queue = QueueService(
        max_parallel_inference=settings.runtime.max_parallel_inference,
        enabled=settings.runtime.queue_enabled,
        max_size=settings.runtime.queue_max_size,
    )
    metrics = MetricsService(settings, queue)
    models = ModelManager(metrics)
    voices = VoiceManager(models)
    audio = AudioService(settings, queue)
    tts = TTSService(settings, queue, models, voices, audio, metrics)
    configure_whisper(audio, metrics)
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
            removed = ctx.audio.cleanup_old(settings.runtime.generated_audio_ttl_hours * 3600)
            if removed:
                logger.info(f"Limpieza periódica: {removed} audio(s) antiguo(s) eliminados")
        except Exception as e:
            logger.warning(f"Error en la limpieza periódica de audios: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación.

    El apagado lo inicia uvicorn al recibir SIGTERM/SIGINT: primero deja de
    aceptar conexiones nuevas y espera a que terminen las peticiones en
    curso (incluidas las inferencias activas); solo entonces se ejecuta
    aquí la liberación ordenada de recursos.
    """
    load_runtime_config()
    apply_log_level()
    ctx = build_context()
    ctx.queue.start()
    register_routes(app, ctx)
    cleanup_task = asyncio.create_task(audio_cleanup_loop(ctx))
    try:
        await startup_procedure(ctx)
        yield
    finally:
        await shutdown_procedure(ctx, cleanup_task)


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

    logger.info("Iniciando qwen3-tts server")

    # Limpiar audios generados más antiguos que el TTL configurado
    ctx.audio.cleanup_old(settings.runtime.generated_audio_ttl_hours * 3600)

    # Modelos locales
    local_models = ctx.models.list_local_models()

    logger.info(f"Modelos locales disponibles ({len(local_models)}): {local_models}")

    # Seleccionar modelo por defecto (configurable, fallback al primero disponible)
    if len(local_models) > 0:
        selected_model = None
        if settings.tts.default_model and settings.tts.default_model in local_models:
            selected_model = settings.tts.default_model
        else:
            selected_model = local_models[0]
            if settings.tts.default_model:
                logger.warning(
                    f"Modelo por defecto '{settings.tts.default_model}' no encontrado, "
                    f"usando '{selected_model}'"
                )

        logger.info(f"Modelo seleccionado por defecto: {selected_model}")

        try:
            async with ctx.queue.model_lock():
                info = await ctx.models.load_model(selected_model)
            logger.info(f"Modelo cargado: {info.model_id} ({info.model_type})")
            logger.info(f"VRAM disponible: {get_vram_available()} GB")
        except Exception as e:
            logger.warning(f"Error cargando modelo por defecto: {e}")
    else:
        logger.warning("No hay modelos disponibles. El servidor funcionará sin modelo inicial.")

    # Voces locales
    local_voices = ctx.voices.list()

    logger.info(f"Voces locales disponibles ({len(local_voices)})")
    for v in local_voices:
        status = "OK" if v["valid"] else ("KO" if not v["has_reference_audio"] else "!?")
        logger.info(f"  {v['name']} (id: {v['id']}) {status}")

    # Intentar clonar voz por defecto si hay modelos y voces disponibles
    if len(local_models) > 0 and len(local_voices) >= 1 and ctx.models.is_loaded():

        selected_voice = None
        if settings.tts.default_voice and any(
            v["name"] == settings.tts.default_voice for v in local_voices
        ):
            selected_voice = settings.tts.default_voice
        else:
            selected_voice = local_voices[0]["name"]

        logger.info(f"Intentando clonar voz por defecto: {selected_voice}")

        try:
            async with ctx.queue.inference_lock():
                await ctx.voices.load_voice(selected_voice)
            logger.info(f"Voz '{selected_voice}' clonada correctamente y aplicada por defecto")
        except Exception as e:
            logger.warning(f"Error creando voz clonada (continuar sin voice cloning): {e}")

    logger.info("Escuchando...")
    logger.info(f"WebUI disponible en: http://localhost:{settings.server.port}/webui")
    logger.info(f"Documentación de la API: http://localhost:{settings.server.port}/webui/docs")


async def shutdown_procedure(ctx: AppContext, cleanup_task: asyncio.Task):
    """Apagado ordenado del servidor (al recibir SIGTERM/SIGINT).

    uvicorn ya ha hecho, antes de llegar aquí: (1) dejar de aceptar
    conexiones nuevas y (2) esperar a que terminen las peticiones en curso
    (jobs activos incluidos, sin cortar inferencias a mitad de ejecución).

    Aquí solo queda: (3) detener workers/tareas de fondo, (4) liberar
    modelos y VRAM, (5) limpiar recursos, (6) salir.
    """
    logger.info("Apagando servidor...")

    # 3. Detener tareas de fondo: limpieza periódica de audios, workers de
    #    la cola de inferencia (drenando jobs pendientes) y reproducción.
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    try:
        await ctx.queue.stop()
    except Exception as e:
        logger.warning(f"Error deteniendo la cola de inferencia: {e}")

    try:
        await ctx.queue.stop_playback()
    except Exception as e:
        logger.warning(f"Error deteniendo la reproducción: {e}")

    # 4. Liberar modelos (TTS y Whisper). Ya no hay inferencias en curso,
    #    pero se toma el model_lock para respetar la exclusión mutua.
    async with ctx.queue.model_lock():
        try:
            active = await ctx.models.get_active_model()
            if active is not None:
                await ctx.models.unload_model(active.model_id)
                ctx.voices.unload_voice()
                logger.info(f"Modelo TTS descargado: {active.model_id}")
        except Exception as e:
            logger.warning(f"Error descargando el modelo TTS: {e}")
        try:
            if whisper_service.unload_if_loaded():
                logger.info("Modelo Whisper descargado")
        except Exception as e:
            logger.warning(f"Error descargando Whisper: {e}")

    # Liberar memoria sobrante antes de salir
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Servidor detenido correctamente")


# Punto de entrada principal
if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host=settings.server.host,
        port=settings.server.port,
        # Apagado ordenado: esperar (sin límite) a que terminen las peticiones
        # e inferencias en curso al recibir SIGTERM/SIGINT, en lugar de cortarlas.
        timeout_graceful_shutdown=None,
    )
    try:
        uvicorn.Server(config).run()
    except KeyboardInterrupt:
        # Ctrl+C: el apagado ordenado (shutdown_procedure) ya se ha completado;
        # el runner de asyncio (Py3.14) re-lanza KeyboardInterrupt al salir.
        # Mismo comportamiento que la CLI de uvicorn: salir sin traceback.
        pass
