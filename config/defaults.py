#!/usr/bin/env python3
"""Valores por defecto de configuración en tiempo de ejecución."""

# Valores por defecto
def_language = "Spanish"
def_voice = "Serena"
def_instruct = "Habla en español de España con acento neutro. Evita cualquier tono robótico."

DEFAULTS = {
    "max_text_chars": 1000,
    "playback_wait_timeout": 300,
    "def_language": def_language,
    "def_voice": def_voice,
    "def_instruct": def_instruct,
    "log_level": "INFO",
    "log_requests": True,
    "api_keys_enabled": False,
    # Dispositivo de inferencia: "auto" (GPU si hay), "cuda:N" (GPU concreta) o "cpu"
    "device": "auto",
}
