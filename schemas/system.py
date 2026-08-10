#!/usr/bin/env python3
"""Esquemas de administración del sistema y del panel web."""

from typing import Optional
from pydantic import BaseModel


class ConfigUpdate(BaseModel):
    """Campos de configuración en tiempo de ejecución actualizables."""
    max_text_chars: Optional[int] = None
    playback_wait_timeout: Optional[int] = None
    def_language: Optional[str] = None
    def_voice: Optional[str] = None
    def_instruct: Optional[str] = None
    log_level: Optional[str] = None
    log_requests: Optional[bool] = None
    api_keys_enabled: Optional[bool] = None
    device: Optional[str] = None


class ApiKeyCreate(BaseModel):
    """Solicitud para crear una clave API."""
    name: str


class SystemStatus(BaseModel):
    """Estado global del servidor."""
    status: str
    current_model: Optional[str]
    clone_active: bool
    vram_available_gb: float
    device: str
