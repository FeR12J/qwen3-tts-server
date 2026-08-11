#!/usr/bin/env python3
"""Esquemas de las solicitudes de TTS.

Schema unificado para todos los endpoints TTS (/tts, /tts/stream, /tts/play,
/tts/audio/speech). Los campos corresponden a parámetros reales del
modelo Qwen3-TTS instalado; la validación según el tipo de modelo
(decide qué campos son obligatorios) se hace en services/tts_service.py
antes de consumir GPU.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class TTSRequest(BaseModel):
    """Solicitud unificada de generación TTS.

    - ``text``: texto a sintetizar (obligatorio; ``input`` se acepta por
      compatibilidad con OpenWebUI).
    - ``model``: modelo a usar (si difiere del activo, se cambia).
    - ``language``: idioma de síntesis.
    - ``speaker``: voz por defecto (modelos CustomVoice).
    - ``voice``: voz local (voice.wav + text.txt) como referencia de clonación.
    - ``voice_description`` / ``instruct`` (alias legacy): descripción de la
      voz para VoiceDesign.
    - ``reference_audio`` / ``reference_text``: clonación de voz (modelos Base).
    - ``temperature``: temperatura de generación (pasada a la librería).
    - ``output_format``: "wav" o "pcm" (PCM 16-bit little-endian).
    """
    text: Optional[str] = None
    input: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    voice: Optional[str] = None
    voice_description: Optional[str] = None
    instruct: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    temperature: Optional[float] = None
    output_format: Literal["wav", "pcm"] = "wav"
