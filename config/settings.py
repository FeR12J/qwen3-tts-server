#!/usr/bin/env python3
"""Configuración estática del servidor TTS."""

import os

from utils.paths import (
    BASE_DIR,
    MODELS_DIR,
    VOICES_DIR,
    AUDIOS_DIR,
    CLONE_PROMPTS_DIR,
)

CONFIG = {
    "host": "0.0.0.0",
    "port": 8001,
    "local_models_dir": MODELS_DIR,
    "local_voices_dir": VOICES_DIR,
    "whisper_model": "whisper-large-v3",
    "max_text_chars": 1000,
    "playback_wait_timeout": 300,
    "clone_prompts_dir": CLONE_PROMPTS_DIR,
    "audios_dir": AUDIOS_DIR,
    "log_file": os.path.join(BASE_DIR, "requests.log"),
    # Modelo que se carga al arrancar. Si está vacío o no existe, se usa el primero disponible.
    "default_model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    # Voz que se clona al arrancar. Si está vacío o no existe, se usa la primera disponible.
    "default_voice": "annab",
    # Tamaño máximo de requests.log antes de rotar (bytes)
    "log_max_bytes": 5 * 1024 * 1024,
    # Edad máxima (días) de los audios antes de ser eliminados al arrancar
    "audios_max_age_days": 7,
}

# Inicializar directorios
os.makedirs(CLONE_PROMPTS_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(AUDIOS_DIR, exist_ok=True)
