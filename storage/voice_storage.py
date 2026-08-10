#!/usr/bin/env python3
"""Persistencia de voces locales en disco."""

import os

from utils.paths import VOICES_DIR


def list_voice_dirs() -> list:
    """Listar directorios de voces en el directorio local."""
    try:
        return sorted(
            d for d in os.listdir(VOICES_DIR)
            if os.path.isdir(os.path.join(VOICES_DIR, d))
        )
    except OSError:
        return []


def list_voices_detail() -> list:
    """Listar voces con el estado de su estructura (voice.wav y text.txt)."""
    voices = []
    for v in list_voice_dirs():
        voice_path = os.path.join(VOICES_DIR, v)
        has_wav = os.path.exists(os.path.join(voice_path, "voice.wav"))
        has_txt = os.path.exists(os.path.join(voice_path, "text.txt"))
        voices.append({
            "name": v,
            "valid": has_wav and has_txt,
            "has_voice_wav": has_wav,
            "has_text_txt": has_txt,
        })
    return voices


def get_voice_files(voice_name: str) -> tuple:
    """Devolver (wav_path, txt_path) de una voz existente.

    Lanza FileNotFoundError si la voz no existe y ValueError si falta
    voice.wav o text.txt.
    """
    voice_path = os.path.join(VOICES_DIR, voice_name)
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voz '{voice_name}' no encontrada en {VOICES_DIR}")

    wav_path = os.path.join(voice_path, "voice.wav")
    txt_path = os.path.join(voice_path, "text.txt")

    if not os.path.exists(wav_path):
        raise ValueError(f"Falta voice.wav en {voice_path}")
    if not os.path.exists(txt_path):
        raise ValueError(f"Falta text.txt en {voice_path}")

    return wav_path, txt_path


def save_voice_files(voice_name: str, audio_bytes: bytes, text: str) -> tuple:
    """Guardar los archivos de una voz en voices/<nombre>/. Devuelve (wav, txt)."""
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    os.makedirs(voice_dir, exist_ok=True)
    wav_path = os.path.join(voice_dir, "voice.wav")
    txt_path = os.path.join(voice_dir, "text.txt")

    with open(wav_path, "wb") as f:
        f.write(audio_bytes)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return wav_path, txt_path
