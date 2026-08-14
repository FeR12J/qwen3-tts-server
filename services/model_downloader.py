#!/usr/bin/env python3
"""Descarga de los modelos soportados por el servidor (Hugging Face Hub).

Solo se permite descargar los modelos de la lista ``SUPPORTED_MODELS``: un
repo_id arbitrario nunca se acepta (el endpoint solo recibe el nombre local
del modelo). Los archivos se guardan en ``settings.paths.models_dir`` con el
mismo nombre de directorio que usa el servidor para cargarlos, de modo que
al terminar aparecen automáticamente en /models y en la tabla del panel.

Las descargas se ejecutan en segundo plano (asyncio) y solo una a la vez;
el estado de cada modelo se consulta con ``list_status()``.
"""

import asyncio
import logging
import os
import re
from typing import Optional

from config.settings import settings
from services.errors import APIError

logger = logging.getLogger("tts")

# Whitelist de modelos soportados: nombre del directorio local (el que el
# servidor usa para cargar) -> repo de Hugging Face. NO añadir repos
# arbitrarios: el panel solo puede descargar estos.
SUPPORTED_MODELS = [
    {
        "name": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "kind": "tts",
        "description": "TTS 0.6B con clonado de voz (más ligero y rápido)",
    },
    {
        "name": "Qwen3-TTS-12Hz-1.7B-Base",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "kind": "tts",
        "description": "TTS 1.7B base (sin clonado de voz)",
    },
    {
        "name": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "kind": "tts",
        "description": "TTS 1.7B con clonado de voz",
    },
    {
        "name": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "kind": "tts",
        "description": "TTS 1.7B con diseño de voz por descripción (por defecto)",
    },
    {
        "name": "whisper-small",
        "repo_id": "openai/whisper-small",
        "kind": "whisper",
        "description": "Whisper small para transcripción (244M parámetros, el más ligero)",
    },
    {
        "name": "whisper-medium",
        "repo_id": "openai/whisper-medium",
        "kind": "whisper",
        "description": "Whisper medium para transcripción (769M parámetros, equilibrio precisión/velocidad)",
    },
    {
        "name": "whisper-large-v3",
        "repo_id": "openai/whisper-large-v3",
        "kind": "whisper",
        "description": "Whisper large-v3 para transcripción (1550M parámetros, el más preciso)",
    },
]

SUPPORTED_BY_NAME = {m["name"]: m for m in SUPPORTED_MODELS}

# Archivos de pesos: si el directorio contiene alguno, la descarga se
# considera completa (una descarga a medias solo tendría archivos parciales).
_WEIGHT_RE = re.compile(r"^(model.*\.safetensors|model.*\.bin|pytorch_model\.bin)$")

# Estado de descarga por nombre de modelo: {"status": idle|downloading|done|error,
# "error": str | None}
_STATE: dict = {}


def _model_dir(name: str) -> str:
    return os.path.join(settings.paths.models_dir, name)


def _is_installed(name: str) -> bool:
    """¿El modelo está instalado en el directorio local (con pesos)?"""
    base = _model_dir(name)
    if not os.path.isdir(base):
        return False
    for root, _dirs, files in os.walk(base):
        if any(_WEIGHT_RE.match(f) for f in files):
            return True
    return False


def _dir_size(name: str) -> Optional[int]:
    """Tamaño en bytes del directorio del modelo (o None si no existe)."""
    base = _model_dir(name)
    if not os.path.isdir(base):
        return None
    total = 0
    for root, _dirs, files in os.walk(base):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def list_status() -> list:
    """Estado de cada modelo soportado para el panel.

    Cada fila: name, repo_id, kind, description, installed, status, error,
    size_bytes. El estado de descarga (downloading/error) tiene prioridad
    sobre la detección de instalación.
    """
    rows = []
    for m in SUPPORTED_MODELS:
        name = m["name"]
        st = _STATE.get(name, {})
        status = st.get("status", "idle")
        installed = _is_installed(name)
        if installed and status not in ("downloading", "error"):
            status = "done"
        rows.append({
            "name": name,
            "repo_id": m["repo_id"],
            "kind": m["kind"],
            "description": m["description"],
            "installed": installed,
            "status": status,
            "error": st.get("error"),
            "size_bytes": _dir_size(name),
        })
    return rows


async def start_download(model_id: str) -> dict:
    """Iniciar la descarga de un modelo soportado (en segundo plano).

    Validación estricta: ``model_id`` debe ser uno de los nombres de
    SUPPORTED_MODELS (nunca un repo_id arbitrario). Si el modelo ya está
    instalado o en descarga, no se lanza una segunda descarga.
    """
    model = SUPPORTED_BY_NAME.get(model_id)
    if model is None:
        raise APIError(
            "MODEL_NOT_SUPPORTED",
            f"El modelo '{model_id}' no está soportado. Instalables: "
            + ", ".join(SUPPORTED_BY_NAME),
            400,
        )

    if _STATE.get(model_id, {}).get("status") == "downloading":
        raise APIError(
            "DOWNLOAD_IN_PROGRESS", f"El modelo '{model_id}' ya se está descargando", 409
        )

    if _is_installed(model_id):
        _STATE[model_id] = {"status": "done", "error": None}
        return {
            "status": "ok",
            "started": False,
            "model": model_id,
            "message": f"El modelo '{model_id}' ya está instalado",
        }

    # Solo una descarga a la vez (los repos comparten ~GB de descarga y la
    # detección de instalación no es fiable con dos escrituras simultáneas).
    for name, st in _STATE.items():
        if st.get("status") == "downloading":
            raise APIError(
                "ANOTHER_DOWNLOAD_IN_PROGRESS",
                f"Ya se está descargando '{name}'. Espera a que termine para empezar otra.",
                409,
            )

    _STATE[model_id] = {"status": "downloading", "error": None}
    asyncio.create_task(_run_download(model))
    return {
        "status": "ok",
        "started": True,
        "model": model_id,
        "message": f"Descarga de '{model_id}' iniciada en segundo plano",
    }


def _snapshot_download(model: dict) -> None:
    """Descarga bloqueante de un modelo a su directorio local (en hilo)."""
    from huggingface_hub import snapshot_download

    target = _model_dir(model["name"])
    os.makedirs(target, exist_ok=True)
    # local_dir_use_symlinks=False: copias reales (funciona offline después
    # de la descarga; en huggingface_hub >= 0.26 el param se ignora).
    try:
        snapshot_download(
            repo_id=model["repo_id"],
            local_dir=target,
            local_dir_use_symlinks=False,
        )
    except TypeError:
        # Version nueva sin el parámetro: descargar directo a local_dir
        snapshot_download(repo_id=model["repo_id"], local_dir=target)


async def _run_download(model: dict) -> None:
    """Wrapper asíncrono: ejecuta la descarga en un hilo y actualiza el estado."""
    name = model["name"]
    try:
        await asyncio.to_thread(_snapshot_download, model)
        _STATE[name] = {"status": "done", "error": None}
        logger.info(f"Modelo descargado: {name} -> {model['repo_id']} ({model['kind']})")
    except Exception as e:
        logger.error(f"Error descargando el modelo '{name}': {e}")
        _STATE[name] = {"status": "error", "error": str(e)}