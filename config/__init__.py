#!/usr/bin/env python3
"""Configuración del servidor TTS."""

from .settings import CONFIG
from .defaults import (
    DEFAULTS,
    def_language,
    def_voice,
    def_instruct,
)

__all__ = [
    "CONFIG",
    "DEFAULTS",
    "def_language",
    "def_voice",
    "def_instruct",
]
