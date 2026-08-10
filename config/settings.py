#!/usr/bin/env python3
"""Configuración del servidor TTS."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "host": "0.0.0.0",
    "port": 8001,
    "local_models_dir": os.path.join(BASE_DIR, "models"),
    "local_voices_dir": os.path.join(BASE_DIR, "voices"),
    "whisper_model": "whisper-large-v3",
    "max_text_chars": 1000,
    "playback_wait_timeout": 300,
    "clone_prompts_dir": os.path.join(BASE_DIR, "clone_prompts"),
    "audios_dir": os.path.join(BASE_DIR, "audios"),
    "log_file": os.path.join(BASE_DIR, "requests.log"),
    # Modelo que se carga al arrancar. Si está vacío o no existe, se usa el primero disponible.
    "default_model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    # Voz que se clona al arrancar. Si está vacío o no existe, se usa la primera disponible.
    "default_voice": "annab",
    # Tamaño máximo de requests.log antes de rotar (bytes)
    "log_max_bytes": 5 * 1024 * 1024,
    # Edad máxima (días) de los audios antes de ser eliminados al arrancar
    "audios_max_age_days": 7
}

# Valores por defecto
def_language = "Spanish"
def_voice = "Serena"
def_instruct = "Habla en español de España con acento neutro. Evita cualquier tono robótico."

# Inicializar directorios
os.makedirs(CONFIG["clone_prompts_dir"], exist_ok=True)
os.makedirs(CONFIG["local_voices_dir"], exist_ok=True)
os.makedirs(CONFIG["audios_dir"], exist_ok=True)
