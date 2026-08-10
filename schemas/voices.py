#!/usr/bin/env python3
"""Esquemas de las solicitudes de voces."""

from pydantic import BaseModel


class LoadVoiceRequest(BaseModel):
    """Solicitud para cargar voz."""
    voice_name: str
