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
    streaming_enabled: Optional[bool] = None
    save_audios: Optional[bool] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    unload_tts_for_whisper: Optional[bool] = None
    unload_whisper_for_tts: Optional[bool] = None
    chunking: Optional[str] = None
    normalize_reference_audio: Optional[bool] = None
    normalization_dbfs: Optional[float] = None
    generated_audio_ttl_hours: Optional[float] = None
    max_parallel_inference: Optional[int] = None
    max_text_characters: Optional[int] = None
    max_estimated_audio_duration_seconds: Optional[int] = None
    max_reference_audio_mb: Optional[int] = None
    max_reference_duration_seconds: Optional[int] = None
    max_voice_audio_bytes_mb: Optional[int] = None
    max_voice_audio_duration_seconds: Optional[int] = None
    max_transcribe_audio_bytes_mb: Optional[int] = None
    max_transcribe_duration_seconds: Optional[int] = None
    min_sample_rate: Optional[int] = None
    max_sample_rate: Optional[int] = None
    max_channels: Optional[int] = None


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
