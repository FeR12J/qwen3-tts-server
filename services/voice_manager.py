#!/usr/bin/env python3
"""Gestión de voces y voice cloning."""

import logging

from storage import voice_storage

logger = logging.getLogger("tts")


class VoiceManager:
    """Voces locales y prompt de clonación activo."""

    def __init__(self, model_manager):
        self._model_manager = model_manager
        self._clone_prompt = None

    @property
    def clone_prompt(self):
        return self._clone_prompt

    @property
    def clone_active(self) -> bool:
        return self._clone_prompt is not None

    def list_voices(self) -> list:
        """Listar voces con el estado de su estructura."""
        return voice_storage.list_voices_detail()

    async def load_voice(self, voice_name: str) -> str:
        """Cargar una voz existente y generar su prompt de clonación."""
        wav_path, txt_path = voice_storage.get_voice_files(voice_name)
        logger.info(f"Creando voz clonada para: {voice_name}")
        self._clone_prompt = await self._model_manager.create_voice_clone_prompt(wav_path, txt_path)
        logger.info(f"Voz clonada creada y aplicada: {voice_name}")
        return voice_name

    async def create_voice(self, voice_name: str, text: str, audio_bytes: bytes) -> str:
        """Guardar una voz nueva (wav + transcripción) y aplicarla como clon."""
        logger.info(f"Voz '{voice_name}' guardada en el directorio de voces")
        wav_path, txt_path = voice_storage.save_voice_files(voice_name, audio_bytes, text)
        logger.info(f"Creando voz clonada para: {voice_name}")
        self._clone_prompt = await self._model_manager.create_voice_clone_prompt(wav_path, txt_path)
        logger.info(f"Voz clonada creada y aplicada: {voice_name}")
        return voice_name

    def unload_voice(self) -> bool:
        """Desactivar el voice cloning. Devuelve si estaba activo."""
        was_active = self._clone_prompt is not None
        self._clone_prompt = None
        if was_active:
            logger.info("Voice cloning desactivado")
        return was_active
