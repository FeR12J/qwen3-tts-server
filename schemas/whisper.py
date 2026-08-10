#!/usr/bin/env python3
"""Esquemas del servicio Whisper.

La subida de audio para transcribir usa multipart (UploadFile), por lo que
no requiere un esquema JSON para el cuerpo de la petición.
"""

from pydantic import BaseModel


class WhisperStatusResponse(BaseModel):
    """Estado del servicio de transcripción."""
    status: str
    model_loaded: bool
    model: str
    device: str
