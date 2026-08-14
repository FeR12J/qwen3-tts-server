#!/usr/bin/env python3
"""Gestión de voces locales y voice cloning (VoiceManager).

CRUD completo sobre la estructura ``voices/<voice_id>/`` (metadata.json +
reference.wav + reference.txt), con resolución por id o por nombre y prompt
de clonación activo.
"""

import logging
import secrets
from datetime import datetime

from security.validation import is_safe_voice_ref
from storage import voice_storage

logger = logging.getLogger("tts")

VOICE_ID_PREFIX = "voice_"


def _generate_voice_id() -> str:
    """Id de voz generado por el servidor (``voice_<hex>``). Nunca proviene
    del cliente, por lo que no puede usarse para construir rutas."""
    for _ in range(5):
        voice_id = f"{VOICE_ID_PREFIX}{secrets.token_hex(4)}"
        if voice_storage.read_metadata(voice_id) is None:
            return voice_id
    raise ValueError("No se pudo generar un voice_id único")


class VoiceManager:
    """Voces locales (CRUD) y prompt de clonación activo."""

    def __init__(self, model_manager):
        self._model_manager = model_manager
        self._clone_prompt = None
        self._active_voice_id = None

    @property
    def clone_prompt(self):
        return self._clone_prompt

    @property
    def clone_active(self) -> bool:
        return self._clone_prompt is not None

    @property
    def active_voice_id(self):
        return self._active_voice_id

    @staticmethod
    def _check_no_paths(value: str, field: str) -> None:
        """Rechazar valores con forma de ruta (defensa independiente de las rutas)."""
        v = str(value or "")
        if ("/" in v or "\\" in v or "\x00" in v or ".." in v
                or v in (".", "..") or v.startswith("/") or v.startswith("\\")):
            raise ValueError(f"{field} no puede ser una ruta de archivo: '{value}'")

    # -- Resolución --------------------------------------------------------

    def _resolve_id(self, voice_id_or_name: str) -> str | None:
        """Resolver un id o un nombre de voz a su id de directorio.

        Rechaza valores con forma de ruta (separadores, ``..``, rutas
        absolutas): el cliente solo puede referirse a voces por id o nombre,
        nunca por rutas internas.
        """
        if not voice_id_or_name:
            return None
        value = str(voice_id_or_name).strip()
        if not is_safe_voice_ref(value):
            return None
        try:
            meta = voice_storage.read_metadata(value)
        except voice_storage.VoiceNotFoundError:
            meta = None
        if meta is not None:
            return meta["id"]
        for item in voice_storage.list_voices():
            if item.get("name") == value:
                return item["id"]
        return None

    # -- CRUD --------------------------------------------------------------

    def list(self) -> list:
        """Listar voces con su metadata y estado de archivos."""
        return voice_storage.list_voices()

    def get(self, voice_id: str) -> dict:
        """Metadata de una voz. Lanza FileNotFoundError si no existe."""
        meta = voice_storage.read_metadata(voice_id)
        if meta is None:
            raise FileNotFoundError(f"Voz '{voice_id}' no encontrada")
        valid = True
        try:
            voice_storage.get_voice_files(voice_id)
        except (ValueError, OSError):
            valid = False
        return {**meta, "valid": valid}

    async def create(self, name: str, text: str, audio_bytes: bytes,
                     language: str | None = None,
                     description: str | None = None) -> str:
        """Crear una voz (metadata + reference.wav + reference.txt).

        El id lo genera el servidor (``voice_<hex>``): nunca se construyen
        rutas a partir de datos del cliente; el nombre visible solo es
        metadata. Devuelve el id creado. La voz creada se aplica como clon
        activo.
        """
        if not name or not name.strip():
            raise ValueError("name es obligatorio")
        self._check_no_paths(name, "name")
        if not text or not text.strip():
            raise ValueError("La transcripción no puede estar vacía")

        voice_id = _generate_voice_id()
        voice_storage.save_voice(
            voice_id,
            metadata={
                "name": name.strip(),
                "language": language,
                "description": description,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            audio_bytes=audio_bytes,
            text=text.strip(),
        )
        await self._activate_clone(voice_id)
        logger.info(f"Voz creada: {voice_id} ('{name}')")
        return voice_id

    async def update(self, voice_id: str, name: str | None = None,
                     language: str | None = None, description: str | None = None,
                     text: str | None = None, audio_bytes: bytes | None = None) -> dict:
        """Actualizar metadata y/o archivos de referencia de una voz.

        Los campos no proporcionados se conservan. Si cambia el audio o la
        transcripción y la voz es el clon activo, se regenera el prompt.
        Devuelve la metadata actualizada.
        """
        meta = voice_storage.read_metadata(voice_id)
        if meta is None:
            raise FileNotFoundError(f"Voz '{voice_id}' no encontrada")

        new_meta = dict(meta)
        if name is not None:
            if not name.strip():
                raise ValueError("name no puede estar vacío")
            self._check_no_paths(name, "name")
            new_meta["name"] = name.strip()
        if language is not None:
            new_meta["language"] = language
        if description is not None:
            new_meta["description"] = description

        reference_changed = audio_bytes is not None or text is not None
        new_text = text if text is not None else new_meta.get("reference_text", "")
        voice_storage.save_voice(
            voice_id,
            metadata=new_meta,
            audio_bytes=audio_bytes,
            text=new_text,
        )
        if reference_changed and self._active_voice_id == voice_id:
            await self._activate_clone(voice_id)
        logger.info(f"Voz actualizada: {voice_id}")
        return self.get(voice_id)

    def delete(self, voice_id: str) -> bool:
        """Eliminar una voz. Devuelve True si existía.

        Si la voz es el clon activo, se desactiva el voice cloning.
        """
        if self._active_voice_id == voice_id:
            self.unload_voice()
        existed = voice_storage.delete_voice(voice_id)
        if existed:
            logger.info(f"Voz eliminada: {voice_id}")
        return existed

    # -- Referencias (para TTS) --------------------------------------------

    def get_reference(self, voice_id_or_name: str) -> tuple | None:
        """(wav_path, txt_path) de referencia de una voz por id o nombre.

        Devuelve None si la voz no existe o no tiene archivos válidos.
        """
        voice_id = self._resolve_id(voice_id_or_name)
        if voice_id is None:
            return None
        try:
            return voice_storage.get_voice_files(voice_id)
        except (ValueError, OSError):
            return None

    # -- Clonación ---------------------------------------------------------

    async def _activate_clone(self, voice_id: str):
        """Generar el prompt de clonación de la voz y aplicarlo."""
        wav_path, txt_path = voice_storage.get_voice_files(voice_id)
        logger.info(f"Creando voz clonada para: {voice_id}")
        self._clone_prompt = await self._model_manager.create_voice_clone_prompt(wav_path, txt_path)
        self._active_voice_id = voice_id
        logger.info(f"Voz clonada creada y aplicada: {voice_id}")

    async def load_voice(self, voice_id: str) -> str:
        """Cargar una voz existente (por id o nombre) y clonarla."""
        resolved = self._resolve_id(voice_id)
        if resolved is None:
            raise FileNotFoundError(f"Voz '{voice_id}' no encontrada")
        await self._activate_clone(resolved)
        return resolved

    def unload_voice(self) -> bool:
        """Desactivar el voice cloning. Devuelve si estaba activo."""
        was_active = self._clone_prompt is not None
        self._clone_prompt = None
        self._active_voice_id = None
        if was_active:
            logger.info("Voice cloning desactivado")
        return was_active
