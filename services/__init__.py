#!/usr/bin/env python3
"""Servicios del servidor TTS."""

from .model_service import load_model
from .tts_service import generate_tts

__all__ = [
    "load_model",
    "generate_tts"
]
