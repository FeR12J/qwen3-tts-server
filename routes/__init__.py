#!/usr/bin/env python3
"""Rutas del servidor TTS."""

from .tts import create_tts_routes
from .models import create_models_routes
from .voices import create_voices_routes
from .system import create_system_routes
from .whisper import create_whisper_routes
from .auth import create_auth_routes
from .webui import create_webui_routes

__all__ = [
    "create_tts_routes",
    "create_models_routes",
    "create_voices_routes",
    "create_system_routes",
    "create_whisper_routes",
    "create_auth_routes",
    "create_webui_routes",
]
