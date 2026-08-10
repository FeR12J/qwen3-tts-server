#!/usr/bin/env python3
"""Esquemas Pydantic para las solicitudes."""

from typing import Optional
from pydantic import BaseModel


class TTSRequest(BaseModel):
    """Solicitud estándar de TTS."""
    text: str
    language: Optional[str] = None
    speaker: Optional[str] = None
    instruct: Optional[str] = None


class TTSRequestOpenWebUI(BaseModel):
    """Solicitud compatible con OpenWebUI."""
    text: Optional[str] = None
    input: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    instruct: Optional[str] = None


class LoadModelRequest(BaseModel):
    """Solicitud para cargar modelo."""
    model_id: str


class LoadVoiceRequest(BaseModel):
    """Solicitud para cargar voz."""
    voice_name: str
