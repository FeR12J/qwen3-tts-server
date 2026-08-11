#!/usr/bin/env python3
"""Configuración centralizada del servidor TTS (Pydantic Settings).

Única fuente de verdad de toda la configuración del proyecto. No se usan
llamadas dispersas a os.getenv: cualquier variable de entorno se lee aquí
(prefix QWEN_TTS_), p.ej. QWEN_TTS_HOST=0.0.0.0, QWEN_TTS_PORT=8001.

Variables planas soportadas (mapeadas por Settings.settings_customise_sources):

    QWEN_TTS_HOST              -> server.host
    QWEN_TTS_PORT              -> server.port
    QWEN_TTS_DEVICE            -> runtime.device
    QWEN_TTS_DTYPE             -> runtime.dtype
    QWEN_TTS_MODEL             -> tts.default_model
    QWEN_TTS_REQUIRE_API_KEY   -> runtime.api_keys_enabled
    QWEN_TTS_VOICES_DIR        -> paths.voices_dir
    QWEN_TTS_AUDIO_DIR         -> paths.audios_dir

El resto de variables usa el nombre compuesto por subgrupo
(QWEN_TTS_<GRUPO>__<CAMPO>), p.ej. QWEN_TTS_CORS__ALLOW_ORIGINS.
Precedencia: variables de entorno > data/runtime.json > defaults.
"""

import os
from typing import ClassVar, Literal

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


class TtsSettings(BaseModel):
    """Modelo TTS que se carga al arrancar (si no existe, el primero)."""
    default_model: str = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    default_voice: str = "annab"


class WhisperSettings(BaseModel):
    """Modelo de transcripción Whisper (carga lazy)."""
    whisper_model: str = "whisper-large-v3"
    max_transcribe_audio_bytes: int = 100 * 1024 * 1024


class TextSettings(BaseModel):
    """Procesado de texto: división en fragmentos de textos largos.

    - ``chunking``: modo de división del TextChunker:
        * ``sentence``: fragmentos de frases completas (párrafos no mezclados).
        * ``paragraph``: fragmentos de párrafos completos (solo se subdividen
          los párrafos que exceden el tamaño de fragmento).

    El límite de caracteres por petición vive en limits.max_text_characters.
    """
    chunking: Literal["sentence", "paragraph"] = "sentence"


class CorsSettings(BaseModel):
    """Política CORS del servidor."""
    allow_origins: list = ["*"]
    allow_credentials: bool = False
    allow_methods: list = ["*"]
    allow_headers: list = ["*"]


class LimitsSettings(BaseModel):
    """Límites de entrada del servidor (se comprueban antes de usar GPU).

    - ``max_text_characters``: longitud máxima del texto por petición TTS.
    - ``max_reference_audio_mb``: tamaño máximo del audio de referencia
      (clonación de voz) en MB.
    - ``max_audio_duration_seconds``: duración máxima estimada del audio
      generado (estimada desde la longitud del texto, ~16 caracteres/segundo).
    """
    audios_max_age_days: int = 7
    max_voice_audio_bytes: int = 50 * 1024 * 1024
    max_text_characters: int = 10000
    max_reference_audio_mb: int = 25
    max_audio_duration_seconds: int = 30


class QueueSettings(BaseModel):
    """Cola de inferencia GPU."""
    max_parallel_inference: int = 1


class AuthSettings(BaseModel):
    """Archivo donde se persisten las claves API."""
    keys_file: str = Field(default_factory=lambda: os.path.join(DATA_DIR, "apikeys.json"))


class LoggingSettings(BaseModel):
    """Logging y registro de peticiones."""
    log_file: str = Field(default_factory=lambda: os.path.join(BASE_DIR, "requests.log"))
    log_max_bytes: int = 5 * 1024 * 1024


class RuntimeSettings(BaseModel):
    """Configuración en tiempo de ejecución: editable desde el panel y
    persistida en disco (data/runtime.json). Las variables de entorno
    QWEN_TTS_DEVICE, QWEN_TTS_DTYPE y QWEN_TTS_REQUIRE_API_KEY tienen
    prioridad sobre el archivo persistido."""
    max_text_chars: int = 1000
    playback_wait_timeout: int = 300
    def_language: str = "spanish"
    def_voice: str = "Serena"
    def_instruct: str = (
        "Habla en español de España con acento neutro. Evita cualquier tono robótico."
    )
    log_level: str = "INFO"
    log_requests: bool = True
    api_keys_enabled: bool = False
    # Endpoint /tts/stream habilitado (generación por frases en streaming)
    streaming_enabled: bool = True
    # Guardar una copia de cada audio generado en el directorio audios/
    save_audios: bool = True
    # Dispositivo de inferencia: "auto" (GPU si hay), "cuda:N" o "cpu"
    device: str = "auto"
    # dtype: "auto" (según GPU), "bfloat16", "float16" o "float32"
    dtype: str = "auto"
    # Gestión de VRAM compartida TTS <-> Whisper (GPUs pequeñas)
    unload_tts_for_whisper: bool = True
    unload_whisper_for_tts: bool = True


# -- Objeto Settings único -------------------------------------------------

VERSION = "1.1.0"
"""Versión del servidor TTS (expuesta en /version)."""


class Settings(BaseSettings):
    """Configuración centralizada del servidor TTS."""

    # Variables de entorno planas QWEN_TTS_* -> (grupo, campo)
    FLAT_ENV: ClassVar[dict] = {
        "QWEN_TTS_HOST": ("server", "host"),
        "QWEN_TTS_PORT": ("server", "port"),
        "QWEN_TTS_DEVICE": ("runtime", "device"),
        "QWEN_TTS_DTYPE": ("runtime", "dtype"),
        "QWEN_TTS_MODEL": ("tts", "default_model"),
        "QWEN_TTS_REQUIRE_API_KEY": ("runtime", "api_keys_enabled"),
        "QWEN_TTS_VOICES_DIR": ("paths", "voices_dir"),
        "QWEN_TTS_AUDIO_DIR": ("paths", "audios_dir"),
    }

    model_config = SettingsConfigDict(
        env_prefix="QWEN_TTS_",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    server: ServerSettings = ServerSettings()
    paths: PathsSettings = PathsSettings()
    tts: TtsSettings = TtsSettings()
    text: TextSettings = TextSettings()
    whisper: WhisperSettings = WhisperSettings()
    cors: CorsSettings = CorsSettings()
    limits: LimitsSettings = LimitsSettings()
    queue: QueueSettings = QueueSettings()
    auth: AuthSettings = AuthSettings()
    logging: LoggingSettings = LoggingSettings()
    runtime: RuntimeSettings = RuntimeSettings()

    @classmethod
    def _flat_env_values(cls) -> dict:
        """Valores de las variables de entorno planas, agrupados por grupo."""
        init = {}
        for env_name, (group, field) in cls.FLAT_ENV.items():
            value = os.environ.get(env_name)
            if value is None or value == "":
                continue
            init.setdefault(group, {})[field] = value
        return init

    def settings_customise_sources(
        cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Inyectar las variables planas QWEN_TTS_* como init (pydantic las
        tipa: "true" -> bool, "8001" -> int). Tienen máxima precedencia."""
        from pydantic_settings.sources.base import InitSettingsSource

        flat = cls._flat_env_values()
        if flat:
            init_settings = InitSettingsSource(
                cls,
                {**flat, **getattr(init_settings, "init_kwargs", {})},
                nested_model_default_partial_update=getattr(
                    init_settings, "nested_model_default_partial_update", None
                ),
            )
        return init_settings, env_settings, dotenv_settings, file_secret_settings


settings = Settings()


def get_settings() -> Settings:
    """Devolver la instancia única de configuración."""
    return settings
