#!/usr/bin/env python3
"""Configuración centralizada del servidor TTS (Pydantic Settings).

Única fuente de verdad de toda la configuración del proyecto. No se usan
llamadas dispersas a os.getenv: cualquier variable de entorno se lee aquí
(prefijo TTS_, subcampos con __), p.ej. TTS_SERVER__HOST=0.0.0.0.
"""

import os

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.paths import (
    BASE_DIR,
    MODELS_DIR,
    VOICES_DIR,
    AUDIOS_DIR,
    CLONE_PROMPTS_DIR,
    DATA_DIR,
    WEBUI_DIR,
)


# -- Grupos de configuración -----------------------------------------------


class ServerSettings(BaseModel):
    """Host y puerto del servidor HTTP."""
    host: str = "0.0.0.0"
    port: int = 8001


class PathsSettings(BaseModel):
    """Directorios del proyecto."""
    base_dir: str = BASE_DIR
    models_dir: str = MODELS_DIR
    voices_dir: str = VOICES_DIR
    audios_dir: str = AUDIOS_DIR
    clone_prompts_dir: str = CLONE_PROMPTS_DIR
    data_dir: str = DATA_DIR
    webui_dir: str = WEBUI_DIR


class ModelSettings(BaseModel):
    """Modelo TTS que se carga al arrancar (si no existe, el primero)."""
    default_model: str = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    default_voice: str = "annab"


class WhisperSettings(BaseModel):
    """Modelo de transcripción Whisper (carga lazy)."""
    whisper_model: str = "whisper-large-v3"
    max_transcribe_audio_bytes: int = 100 * 1024 * 1024


class CorsSettings(BaseModel):
    """Política CORS del servidor."""
    allow_origins: list = ["*"]
    allow_credentials: bool = False
    allow_methods: list = ["*"]
    allow_headers: list = ["*"]


class LimitsSettings(BaseModel):
    """Límites estáticos (tamaños, rotaciones, retención)."""
    audios_max_age_days: int = 7
    max_voice_audio_bytes: int = 50 * 1024 * 1024


class QueueSettings(BaseModel):
    """Cola de inferencia GPU."""
    max_parallel_inference: int = 1


class AuthSettings(BaseModel):
    """Autenticación por clave API."""
    api_keys_enabled: bool = False
    keys_file: str = Field(default_factory=lambda: os.path.join(DATA_DIR, "apikeys.json"))


class LoggingSettings(BaseModel):
    """Logging y registro de peticiones."""
    log_file: str = Field(default_factory=lambda: os.path.join(BASE_DIR, "requests.log"))
    log_max_bytes: int = 5 * 1024 * 1024


class RuntimeSettings(BaseModel):
    """Configuración en tiempo de ejecución: editable desde el panel y
    persistida en disco (data/runtime.json)."""
    max_text_chars: int = 1000
    playback_wait_timeout: int = 300
    def_language: str = "Spanish"
    def_voice: str = "Serena"
    def_instruct: str = (
        "Habla en español de España con acento neutro. Evita cualquier tono robótico."
    )
    log_level: str = "INFO"
    log_requests: bool = True
    api_keys_enabled: bool = False
    # Dispositivo de inferencia: "auto" (GPU si hay), "cuda:N" o "cpu"
    device: str = "auto"
    # dtype: "auto" (según GPU), "bfloat16", "float16" o "float32"
    dtype: str = "auto"
    # Gestión de VRAM compartida TTS <-> Whisper (GPUs pequeñas)
    unload_tts_for_whisper: bool = True
    unload_whisper_for_tts: bool = True


# -- Objeto Settings único -------------------------------------------------


class Settings(BaseSettings):
    """Configuración centralizada del servidor TTS."""

    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerSettings = ServerSettings()
    paths: PathsSettings = PathsSettings()
    model: ModelSettings = ModelSettings()
    whisper: WhisperSettings = WhisperSettings()
    cors: CorsSettings = CorsSettings()
    limits: LimitsSettings = LimitsSettings()
    queue: QueueSettings = QueueSettings()
    auth: AuthSettings = AuthSettings()
    logging: LoggingSettings = LoggingSettings()
    runtime: RuntimeSettings = RuntimeSettings()


settings = Settings()


def get_settings() -> Settings:
    """Devolver la instancia única de configuración."""
    return settings
