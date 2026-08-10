#!/usr/bin/env python3
"""Utilidades del servidor TTS."""

from .paths import BASE_DIR
from .gpu import get_vram_available, get_dtype, list_devices
from .logging import setup_logging, rotate_log_if_needed, log_request
from .text import truncate_text

__all__ = [
    "BASE_DIR",
    "get_vram_available",
    "get_dtype",
    "list_devices",
    "setup_logging",
    "rotate_log_if_needed",
    "log_request",
    "truncate_text",
]
