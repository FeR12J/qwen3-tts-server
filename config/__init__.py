#!/usr/bin/env python3
"""Configuración centralizada del servidor TTS."""

from .settings import (
    Settings,
    settings,
    get_settings,
)

__all__ = [
    "Settings",
    "settings",
    "get_settings",
]
