#!/usr/bin/env python3
"""Aplicación principal del servidor TTS."""

import asyncio
import contextvars
import gc
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import MutableHeaders

from config.settings import settings
from utils.logging import setup_logging
from services.config_service import (
    load_runtime_config, apply_log_level, get_runtime_config,
)
from services.errors import APIError
from services.queue_service import QueueService
from services.model_manager import ModelManager
from services.voice_manager import VoiceManager
from services.audio_service import AudioService
from services.whisper_service import configure as configure_whisper
from services import whisper_service
from services.metrics_service import MetricsService
from services.tts_service import TTSService
from routes import (
    create_tts_routes,
    create_models_routes,
    create_voices_routes,
    create_system_routes,
    create_whisper_routes,
    create_auth_routes,
    create_webui_routes,
)


# Estado CORS por petición (contextvars): evita carreras entre requests
# concurrentes al cambiar la configuración en caliente.
_cors_state = contextvars.ContextVar("cors_state", default=None)


class DynamicCORSMiddleware(CORSMiddleware):
    """CORS editable en tiempo de ejecución desde el panel.

    Lee cors_enabled / cors_origins / cors_allow_wildcard de la
    configuración runtime en cada petición, de modo que los cambios se
    aplican sin reiniciar el servidor. Nunca se permite "*" por defecto
    (configuración raíz: enabled=false); el wildcard solo se admite si se
    habilita explícitamente desde el panel.
    """

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            rc = get_runtime_config()
            if rc.get("cors_enabled", False):
                _cors_state.set((
                    bool(rc.get("cors_allow_wildcard", False)),
                    tuple(rc.get("cors_origins") or []),
                ))
            else:
                _cors_state.set((False, ()))
        await super().__call__(scope, receive, send)

    def preflight_response(self, request_headers):
        wildcard, _ = _cors_state.get() or (False, ())
        if wildcard:
            headers = dict(self.preflight_headers)
            if self.allow_credentials:
                headers["Access-Control-Allow-Origin"] = request_headers["origin"]
                headers["Vary"] = "Origin"
            else:
                headers["Access-Control-Allow-Origin"] = "*"
            return Response(status_code=200, headers=headers)
        return super().preflight_response(request_headers)

    async def send(self, message, send, request_headers):
        if message["type"] != "http.response.start":
            await send(message)
            return
        wildcard, origins = _cors_state.get() or (False, ())
        message.setdefault("headers", [])
        headers = MutableHeaders(scope=message)
        origin = request_headers["Origin"]
        if wildcard:
            if self.allow_credentials:
                headers["Access-Control-Allow-Origin"] = origin
                headers.add_vary_header("Origin")
            else:
                headers["Access-Control-Allow-Origin"] = "*"
        elif origin in origins:
            headers["Access-Control-Allow-Origin"] = origin
            headers.add_vary_header("Origin")
        await send(message)

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
    DynamicCORSMiddleware,
    allow_origins=settings.cors.origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    request_id = exc.request_id or request.headers.get("x-request-id") or uuid.uuid4().hex
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Cualquier HTTPException (FastAPI/uvicorn) se normaliza al mismo formato."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    code = f"HTTP_{exc.status_code}"
    message = exc.detail
    if isinstance(exc.detail, dict):
        # Detalles que ya llevan estructura (p.ej. STREAMING_DISABLED)
        err = exc.detail.get("error") or {}
        message = err.get("message", exc.detail)
        code = err.get("code", code)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    """Errores 422 de validación de parámetros/body (FastAPI) unificados."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = ".".join(str(p) for p in first.get("loc", []))
        message = f"Parámetro inválido '{loc}': {first.get('msg', '')}"
    else:
        message = "Petición inválida"
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": message,
                "request_id": request_id,
            }
        },
    )


def register_exception_handlers(app: FastAPI):
    """Registrar los handlers de error unificados en una app FastAPI.

    La app principal los registra vía decoradores en este módulo; esta
    función permite a apps de prueba (y embebidas) usar el mismo formato
    sin duplicar los handlers.
    """
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_handler)


async def startup_procedure(ctx: AppContext):
    """Procedimiento de inicialización del servidor."""

    logger.info("Iniciando qwen3-tts server")

    # Limpiar audios generados más antiguos que el TTL configurado
    ctx.audio.cleanup_old(settings.runtime.generated_audio_ttl_hours * 3600)

    # Modelos locales
    local_models = ctx.models.list_local_models()

    logger.info(f"Modelos locales disponibles ({len(local_models)}): {local_models}")

    # El servidor arranca sin modelo cargado: el usuario elige cuándo cargar
    # (panel o POST /model/load), evitando reservar VRAM al iniciarse.
    logger.info("Servidor iniciado sin modelo cargado (usa /model/load o el panel)")

    # Voces locales
    local_voices = ctx.voices.list()

    logger.info(f"Voces locales disponibles ({len(local_voices)})")
    for v in local_voices:
        status = "OK" if v["valid"] else ("KO" if not v["has_reference_audio"] else "!?")
        logger.info(f"  {v['name']} (id: {v['id']}) {status}")

    logger.info("Escuchando...")
    logger.info(f"WebUI disponible en: http://localhost:{settings.runtime.port}/webui")
    logger.info(f"Documentación de la API: http://localhost:{settings.runtime.port}/webui/docs")


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
    # Cargar la configuración runtime ANTES de construir uvicorn.Config:
    # el puerto editable (settings.runtime.port) se aplica al bind.
    load_runtime_config()
    apply_log_level()
    effective_port = settings.runtime.port or settings.server.port
    config = uvicorn.Config(
        app,
        host=settings.server.host,
        port=effective_port,
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
