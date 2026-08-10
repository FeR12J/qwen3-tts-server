#!/usr/bin/env python3
"""Servicio de audio: codificación, guardado, limpieza y reproducción local."""

import io
import os
import shutil
import subprocess
import tempfile
import logging
import time
from datetime import datetime

import soundfile as sf

from fastapi import HTTPException

logger = logging.getLogger("tts")

# Reproductores de audio disponibles en el sistema para reproducir en este equipo
PLAYERS = [p for p in ["mpv", "ffplay", "paplay", "aplay", "play"] if shutil.which(p)]


class AudioService:
    """Operaciones de audio del servidor."""

    def __init__(self, config, queue):
        """config: config.settings.Settings (objeto Settings único)."""
        self._config = config
        self._queue = queue

    def encode_wav(self, wav, sr) -> bytes:
        """Codificar audio a bytes WAV."""
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format="wav")
        return buffer.getvalue()

    def save(self, wav, sr, prefix: str) -> str:
        """Guardar audio WAV en disco y devolver la ruta."""
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self._config.paths.audios_dir}/{prefix}_{dt}.wav"
        sf.write(path, wav, sr)
        logger.info(f"Audio guardado: {path}")
        return path

    def cleanup_old(self, max_age_days: int) -> int:
        """Eliminar audios generados hace más de max_age_days días."""
        if max_age_days <= 0:
            return 0
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        try:
            for fname in os.listdir(self._config.paths.audios_dir):
                path = os.path.join(self._config.paths.audios_dir, fname)
                try:
                    if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        removed += 1
                except OSError as e:
                    logger.warning(f"No se pudo eliminar {path}: {e}")
        except OSError as e:
            logger.warning(f"Error leyendo directorio de audios {self._config.paths.audios_dir}: {e}")
        if removed:
            logger.info(f"Limpieza: {removed} audio(s) antiguo(s) eliminado(s) de {self._config.paths.audios_dir}")
        return removed

    def pick_player(self):
        """Primer reproductor de audio disponible en el sistema."""
        return PLAYERS[0] if PLAYERS else None

    async def play(self, audio_bytes: bytes, sr: int, timeout: int) -> dict:
        """Reproducir audio en este equipo, serializando con las reproducciones previas.

        Debe ejecutarse dentro de queue.inference_lock(). Devuelve info de la reproducción.
        """
        player = self.pick_player()
        if not player:
            raise HTTPException(
                500,
                "No hay reproductor de audio disponible (mpv, ffplay, paplay, aplay o play)",
            )

        async with self._queue.playback(timeout):
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"tts_play_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav",
            )
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)

            if player == "ffplay":
                cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
            else:
                cmd = [player, tmp_path]

            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._queue.register_playback(proc)
            logger.info(f"Audio en reproducción en este equipo (player={player}): {tmp_path}")

        return {
            "player": player,
            "temp_file": tmp_path,
            "sample_rate": sr,
        }
