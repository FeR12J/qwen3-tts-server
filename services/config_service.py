#!/usr/bin/env python3
"""Configuración en tiempo de ejecución persistente del servidor TTS.

La fuente de verdad única es ``config.settings.settings`` (Settings pydantic);
este módulo solo gestiona la parte editable (settings.runtime) y su
persistencia en disco.
"""

import os
import logging

from config.settings import settings, RuntimeSettings
from storage.config_storage import load_runtime_file, save_runtime_file

logger = logging.getLogger("tts")

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VALID_DTYPES = ("auto", "bfloat16", "float16", "float32")

RUNTIME_FIELDS = tuple(RuntimeSettings.model_fields)

# Variables de entorno con prioridad sobre data/runtime.json
RUNTIME_ENV_VARS = {
    "device": "QWEN_TTS_DEVICE",
    "dtype": "QWEN_TTS_DTYPE",
    "api_keys_enabled": "QWEN_TTS_REQUIRE_API_KEY",
}

# Defaults editables (runtime) heredados de los grupos estáticos, que son la
# fuente de las variables de entorno QWEN_TTS_<GRUPO>__<CAMPO>: field -> (grupo, campo)
RUNTIME_DEFAULTS_SEED = {
    "chunking": ("text", "chunking"),
    "normalize_reference_audio": ("audio", "normalize_reference_audio"),
    "normalization_dbfs": ("audio", "normalization_dbfs"),
    "generated_audio_ttl_hours": ("storage", "generated_audio_ttl_hours"),
    "max_parallel_inference": ("queue", "max_parallel_inference"),
    "max_text_characters": ("limits", "max_text_characters"),
    "max_estimated_audio_duration_seconds": ("limits", "max_estimated_audio_duration_seconds"),
    "max_reference_audio_mb": ("limits", "max_reference_audio_mb"),
    "max_reference_duration_seconds": ("limits", "max_reference_duration_seconds"),
    "max_voice_audio_duration_seconds": ("limits", "max_voice_audio_duration_seconds"),
    "max_transcribe_duration_seconds": ("limits", "max_transcribe_duration_seconds"),
    "min_sample_rate": ("limits", "min_sample_rate"),
    "max_sample_rate": ("limits", "max_sample_rate"),
    "max_channels": ("limits", "max_channels"),
}


def _seed_runtime_defaults() -> dict:
    """Defaults editables heredados de los grupos estáticos.

    Así, una variable QWEN_TTS_LIMITS__MAX_CHANNELS=4 (o cualquier otro
    ajuste estático por entorno) se aplica también al valor editable del
    panel, manteniendo la precedencia documentada: entorno > runtime.json
    > defaults.
    """
    out = {}
    for field, (group, key) in RUNTIME_DEFAULTS_SEED.items():
        out[field] = getattr(getattr(settings, group), key)
    # Los límites de bytes se guardan en MB en el runtime (unidades legibles)
    out["max_voice_audio_bytes_mb"] = settings.limits.max_voice_audio_bytes // (1024 * 1024)
    out["max_transcribe_audio_bytes_mb"] = settings.limits.max_transcribe_audio_bytes // (1024 * 1024)
    return out


def _env_runtime_values() -> dict:
    """Valores runtime ya resueltos por pydantic desde variables de entorno."""
    return {
        field: getattr(settings.runtime, field)
        for field, env_var in RUNTIME_ENV_VARS.items()
        if os.environ.get(env_var)
    }


def load_runtime_config():
    """Cargar configuración persistida desde disco (si existe).

    Precedencia: variables de entorno QWEN_TTS_* > data/runtime.json >
    defaults (heredados de los grupos estáticos).
    """
    env_values = _env_runtime_values()
    settings.runtime = RuntimeSettings()
    data = load_runtime_file()
    valid = {
        key: value
        for key, value in data.items()
        if key in RUNTIME_FIELDS and value is not None
    }
    if valid or env_values:
        settings.runtime = settings.runtime.model_copy(
            update={**_seed_runtime_defaults(), **valid, **env_values}
        )


def save_runtime_config():
    """Persistir la configuración en disco."""
    save_runtime_file(settings.runtime.model_dump())


def get_runtime_config() -> dict:
    """Devolver la configuración en tiempo de ejecución."""
    return settings.runtime.model_dump()


def get_runtime() -> RuntimeSettings:
    """Devolver la configuración en tiempo de ejecución tipada."""
    return settings.runtime


def get_limits():
    """Límites de entrada vigentes (editables desde el panel).

    Reemplaza a ``settings.limits`` en las rutas/servicios: los valores
    editables (runtime) tienen prioridad sobre los estáticos. Expone los
    mismos nombres que usaban los consumidores; los límites de bytes en
    bytes (los campos runtime se guardan en MB).
    """
    from types import SimpleNamespace

    rc = settings.runtime
    return SimpleNamespace(
        max_text_characters=rc.max_text_characters,
        max_estimated_audio_duration_seconds=rc.max_estimated_audio_duration_seconds,
        max_reference_audio_mb=rc.max_reference_audio_mb,
        max_reference_duration_seconds=rc.max_reference_duration_seconds,
        max_voice_audio_bytes=rc.max_voice_audio_bytes_mb * 1024 * 1024,
        max_voice_audio_duration_seconds=rc.max_voice_audio_duration_seconds,
        max_transcribe_audio_bytes=rc.max_transcribe_audio_bytes_mb * 1024 * 1024,
        max_transcribe_duration_seconds=rc.max_transcribe_duration_seconds,
        min_sample_rate=rc.min_sample_rate,
        max_sample_rate=rc.max_sample_rate,
        max_channels=rc.max_channels,
    )


def update_runtime_config(changes: dict) -> dict:
    """Actualizar claves válidas de la configuración en tiempo de ejecución."""
    valid = {
        key: value
        for key, value in changes.items()
        if key in RUNTIME_FIELDS and value is not None
    }
    if valid:
        settings.runtime = settings.runtime.model_copy(update=valid)
    save_runtime_config()
    return get_runtime_config()


def apply_log_level():
    """Aplicar el nivel de logging configurado al logger principal."""
    level = settings.runtime.log_level
    if level not in VALID_LOG_LEVELS:
        level = "INFO"
    logging.getLogger("tts").setLevel(level)


def resolve_device() -> str:
    """Resolver el dispositivo a usar según la configuración y lo disponible.

    Soporta: "auto", "cpu", "cuda" (GPU 0) y "cuda:N". Si la GPU solicitada
    no existe, resuelve de forma tolerante (warning + cuda:0 / cpu) para que
    el estado y el panel sigan funcionando. Para cargar modelos hay que usar
    ``validated_device``, que sí falla si el dispositivo no existe.
    """
    import torch

    requested = settings.runtime.device
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested != "auto":
            logger.warning(f"Dispositivo '{requested}' solicitado pero no hay CUDA, usando cpu")
        return "cpu"
    if requested in ("auto", "cuda"):
        return "cuda:0"
    if requested.startswith("cuda:"):
        try:
            idx = int(requested.split(":", 1)[1])
            if 0 <= idx < torch.cuda.device_count():
                return requested
        except ValueError:
            pass
        logger.warning(f"GPU '{requested}' no disponible, usando cuda:0")
        return "cuda:0"
    logger.warning(f"Dispositivo '{requested}' inválido, usando cuda:0")
    return "cuda:0"


def validate_device(device: str) -> bool:
    """Comprobar que el valor de device es válido y existe (si es cuda/cuda:N)."""
    if device in ("auto", "cpu"):
        return True
    import torch
    if device == "cuda":
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
            return torch.cuda.is_available() and 0 <= idx < torch.cuda.device_count()
        except (ValueError, ImportError):
            return False
    return False


def validated_device() -> str:
    """Dispositivo real para cargar modelos, validando la configuración.

    A diferencia de ``resolve_device`` (tolerante), falla con un ValueError
    claro si el dispositivo configurado no existe, ANTES de intentar cargar
    ningún modelo:

    - "auto"  -> cuda:0 (o cpu si no hay CUDA)
    - "cuda"  -> cuda:0 (o ValueError si no hay CUDA)
    - "cpu"   -> cpu
    - "cuda:N" -> cuda:N (o ValueError si la GPU N no existe)
    """
    import torch

    requested = settings.runtime.device
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return resolve_device()
    if requested == "cuda":
        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            raise ValueError(
                "Dispositivo 'cuda' solicitado pero no hay ninguna GPU CUDA disponible"
            )
        return "cuda:0"
    if requested.startswith("cuda:"):
        try:
            idx = int(requested.split(":", 1)[1])
        except ValueError:
            raise ValueError(
                f"Dispositivo '{requested}' inválido (use 'auto', 'cpu', 'cuda' o 'cuda:N')"
            )
        if not (torch.cuda.is_available() and 0 <= idx < torch.cuda.device_count()):
            raise ValueError(
                f"Dispositivo '{requested}' solicitado pero no existe "
                f"({torch.cuda.device_count() if torch.cuda.is_available() else 0} GPU(s) disponible(s))"
            )
        return requested
    raise ValueError(
        f"Dispositivo '{requested}' inválido (use 'auto', 'cpu', 'cuda' o 'cuda:N')"
    )


def _supports_bf16(device: str) -> bool:
    """¿La GPU indicada soporta bfloat16? (no asumir que todas la soportan)."""
    import torch

    if not device.startswith("cuda:"):
        return True
    idx = int(device.split(":", 1)[1])
    try:
        with torch.cuda.device(idx):
            return bool(torch.cuda.is_bf16_supported())
    except (RuntimeError, TypeError):
        return False


def resolve_dtype() -> str:
    """Resolver el dtype configurado ("auto" -> según la GPU real).

    En GPU: bfloat16 si la GPU lo soporta, si no float16. En CPU: float32.
    """
    requested = settings.runtime.dtype
    if requested in ("bfloat16", "float16", "float32"):
        return requested
    device = resolve_device()
    if device.startswith("cuda:"):
        return "bfloat16" if _supports_bf16(device) else "float16"
    return "float32"


def validated_dtype() -> str:
    """Dtype real para cargar modelos, compatible con el hardware.

    - "auto": bfloat16 si la GPU lo soporta, si no float16 (CPU: float32).
    - Explícito: se valida la compatibilidad antes de cargar; por ejemplo,
      "bfloat16" en una GPU que no lo soporta falla con un mensaje claro en
      lugar de asumir que todas las GPUs lo soportan.
    """
    requested = settings.runtime.dtype
    if requested in ("bfloat16", "float16", "float32"):
        device = validated_device()
        if requested == "bfloat16" and device.startswith("cuda:") and not _supports_bf16(device):
            raise ValueError(
                f"dtype 'bfloat16' no soportado por {device} "
                f"(use 'auto', 'float16' o 'float32')"
            )
        return requested
    return resolve_dtype()


def validate_dtype(dtype: str) -> bool:
    """Comprobar que el valor de dtype es válido."""
    return dtype in VALID_DTYPES
