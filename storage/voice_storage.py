#!/usr/bin/env python3
"""Persistencia de voces locales en disco.

Estructura por voz:

    voices/
        <voice_id>/
            metadata.json   # id, name, language, description, created_at,
                            # reference_audio, reference_text
            reference.wav   # audio de referencia (16 kHz mono normalizado)
            reference.txt   # transcripción del audio de referencia

Se soporta lectura de voces legadas (directorio con voice.wav + text.txt y
sin metadata.json), sintetizando su metadata sobre la marcha.
"""

import json
import logging
import os
import shutil
from datetime import datetime

from config.settings import settings

logger = logging.getLogger("tts")

# Archivos del formato nuevo
METADATA_FILE = "metadata.json"
REFERENCE_AUDIO_FILE = "reference.wav"
REFERENCE_TEXT_FILE = "reference.txt"
# Archivos del formato legado (voice.wav + text.txt)
LEGACY_AUDIO_FILE = "voice.wav"
LEGACY_TEXT_FILE = "text.txt"


class VoiceNotFoundError(FileNotFoundError):
    """Voz no encontrada (404)."""


def _voice_dir(voice_id: str) -> str:
    """Directorio de la voz, garantizado dentro de ``voices_dir``.

    Resuelve rutas reales (incluidos enlaces simbólicos) y lanza
    VoiceNotFoundError si el id intenta escapar del directorio de voces
    (rutas absolutas, ``..``, separadores...).
    """
    base = os.path.realpath(settings.paths.voices_dir)
    path = os.path.realpath(os.path.join(base, voice_id or ""))
    if not voice_id or voice_id in (".", "..") or path == base or not path.startswith(base + os.sep):
        raise VoiceNotFoundError(f"Voz '{voice_id}' no encontrada")
    return path


def _synthesize_legacy_metadata(voice_id: str) -> dict | None:
    """Metadata sintetizada para voces legadas (voice.wav + text.txt).

    Devuelve None si el directorio no existe o no tiene los archivos legados.
    """
    voice_dir = _voice_dir(voice_id)
    wav = os.path.join(voice_dir, LEGACY_AUDIO_FILE)
    txt = os.path.join(voice_dir, LEGACY_TEXT_FILE)
    if not (os.path.isdir(voice_dir) and os.path.exists(wav) and os.path.exists(txt)):
        return None
    try:
        with open(txt, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
    except OSError:
        ref_text = ""
    try:
        created_at = datetime.fromtimestamp(os.path.getmtime(wav)).isoformat(timespec="seconds")
    except OSError:
        created_at = None
    return {
        "id": voice_id,
        "name": voice_id,
        "language": None,
        "description": None,
        "created_at": created_at,
        "reference_audio": LEGACY_AUDIO_FILE,
        "reference_text": ref_text,
    }


def read_metadata(voice_id: str) -> dict | None:
    """Metadata de la voz (metadata.json), o sintetizada si es legada.

    Devuelve None si la voz no existe. Un metadata.json corrupto se ignora
    (la voz pasa a considerarse inválida).
    """
    voice_dir = _voice_dir(voice_id)
    meta_path = os.path.join(voice_dir, METADATA_FILE)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                meta.setdefault("id", voice_id)
                return meta
        except (OSError, ValueError) as e:
            logger.warning(f"metadata.json corrupto en {voice_dir}: {e}")
        return None
    return _synthesize_legacy_metadata(voice_id)


def write_metadata(voice_id: str, metadata: dict) -> None:
    """Escribir metadata.json de la voz."""
    voice_dir = _voice_dir(voice_id)
    os.makedirs(voice_dir, exist_ok=True)
    meta = {**metadata, "id": voice_id}
    meta_path = os.path.join(voice_dir, METADATA_FILE)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def list_voices() -> list:
    """Listar voces con su metadata y el estado de sus archivos."""
    voices = []
    try:
        dirs = sorted(
            d for d in os.listdir(settings.paths.voices_dir)
            if os.path.isdir(os.path.join(settings.paths.voices_dir, d))
        )
    except OSError:
        return []
    for voice_id in dirs:
        try:
            meta = read_metadata(voice_id)
        except VoiceNotFoundError:
            meta = None
        if meta is None:
            voices.append({
                "id": voice_id,
                "name": voice_id,
                "valid": False,
                "has_reference_audio": False,
                "has_reference_text": False,
            })
            continue
        valid = True
        try:
            get_voice_files(voice_id)
        except (ValueError, OSError):
            valid = False
        voice_dir = _voice_dir(voice_id)
        meta = {**meta, "valid": valid,
                "has_reference_audio": os.path.exists(os.path.join(voice_dir, REFERENCE_AUDIO_FILE))
                or os.path.exists(os.path.join(voice_dir, LEGACY_AUDIO_FILE)),
                "has_reference_text": os.path.exists(os.path.join(voice_dir, REFERENCE_TEXT_FILE))
                or os.path.exists(os.path.join(voice_dir, LEGACY_TEXT_FILE))}
        voices.append(meta)
    return voices


def get_voice_files(voice_id: str) -> tuple:
    """Devolver (wav_path, txt_path) de referencia de una voz existente.

    Lanza VoiceNotFoundError si la voz no existe y ValueError si falta el
    audio o la transcripción de referencia.
    """
    meta = read_metadata(voice_id)
    if meta is None:
        raise VoiceNotFoundError(f"Voz '{voice_id}' no encontrada en {settings.paths.voices_dir}")

    voice_dir = _voice_dir(voice_id)
    ref_audio = meta.get("reference_audio") or REFERENCE_AUDIO_FILE
    wav_path = os.path.join(voice_dir, ref_audio)
    if not os.path.exists(wav_path):
        legacy = os.path.join(voice_dir, LEGACY_AUDIO_FILE)
        wav_path = legacy if os.path.exists(legacy) else None
    if not wav_path:
        raise ValueError(f"Falta el audio de referencia en {voice_dir}")

    txt_path = os.path.join(voice_dir, REFERENCE_TEXT_FILE)
    if not os.path.exists(txt_path):
        legacy = os.path.join(voice_dir, LEGACY_TEXT_FILE)
        txt_path = legacy if os.path.exists(legacy) else None
    if not txt_path:
        ref_text = (meta.get("reference_text") or "").strip()
        if not ref_text:
            raise ValueError(f"Falta la transcripción de referencia en {voice_dir}")
        txt_path = os.path.join(voice_dir, REFERENCE_TEXT_FILE)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ref_text)
    return wav_path, txt_path


def save_voice(voice_id: str, metadata: dict, audio_bytes: bytes | None = None,
               text: str | None = None) -> tuple:
    """Guardar (o actualizar) los archivos de una voz. Devuelve (wav, txt).

    Solo escribe audio_bytes y text si se proporcionan (permitiendo
    actualizaciones parciales). Escribe metadata.json y elimina los archivos
    legados (voice.wav/text.txt) si existieran.
    """
    voice_dir = _voice_dir(voice_id)
    os.makedirs(voice_dir, exist_ok=True)

    wav_path = os.path.join(voice_dir, REFERENCE_AUDIO_FILE)
    if audio_bytes is not None:
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

    txt_path = os.path.join(voice_dir, REFERENCE_TEXT_FILE)
    if text is not None:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    meta = {
        **metadata,
        "id": voice_id,
        "reference_audio": REFERENCE_AUDIO_FILE,
        "reference_text": text if text is not None else metadata.get("reference_text", ""),
    }
    write_metadata(voice_id, meta)

    for legacy in (LEGACY_AUDIO_FILE, LEGACY_TEXT_FILE):
        legacy_path = os.path.join(voice_dir, legacy)
        if os.path.exists(legacy_path):
            try:
                os.remove(legacy_path)
            except OSError as e:
                logger.warning(f"No se pudo eliminar archivo legado {legacy_path}: {e}")

    return wav_path, txt_path


def delete_voice(voice_id: str) -> bool:
    """Eliminar el directorio de la voz. Devuelve True si existía."""
    voice_dir = _voice_dir(voice_id)
    if not os.path.isdir(voice_dir):
        return False
    shutil.rmtree(voice_dir, ignore_errors=True)
    logger.info(f"Voz eliminada: {voice_id}")
    return True
