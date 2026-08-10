#!/usr/bin/env python3
"""Esquemas Pydantic de la API."""

from .tts import TTSRequest, TTSRequestOpenWebUI
from .models import LoadModelRequest
from .voices import LoadVoiceRequest
from .whisper import WhisperStatusResponse
from .system import ConfigUpdate, ApiKeyCreate, SystemStatus
from .errors import ErrorResponse

__all__ = [
    "TTSRequest",
    "TTSRequestOpenWebUI",
    "LoadModelRequest",
    "LoadVoiceRequest",
    "WhisperStatusResponse",
    "ConfigUpdate",
    "ApiKeyCreate",
    "SystemStatus",
    "ErrorResponse",
]
