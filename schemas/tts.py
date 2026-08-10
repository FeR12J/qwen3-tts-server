#!/usr/bin/env python3
"""Esquemas de las solicitudes de TTS."""

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
