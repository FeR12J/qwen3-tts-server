#!/usr/bin/env python3
"""Esquemas del servidor TTS."""

from .schemas import (
    TTSRequest,
    TTSRequestOpenWebUI,
    LoadModelRequest,
    LoadVoiceRequest
)

__all__ = [
    "TTSRequest",
    "TTSRequestOpenWebUI",
    "LoadModelRequest",
    "LoadVoiceRequest"
]
