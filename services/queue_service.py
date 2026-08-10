#!/usr/bin/env python3
"""Sincronización de operaciones sobre modelos e inferencia GPU."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import HTTPException

logger = logging.getLogger("tts")


class QueueService:
    """Locks separados para el ciclo de vida del modelo y la inferencia GPU.

    - ``model_lock``: serializa load/unload/switch. Al ejecutarse, espera a
      que termine la inferencia en curso y bloquea nuevas inferencias, de
      modo que el modelo no cambie mientras se genera audio.
    - ``inference_lock``: controla la inferencia GPU (TTS, voice cloning,
      Whisper). La exclusión mutua evita carreras como:

          Request A -> unload
          Request B -> inference   (modelo a mitad de descarga)
          Request C -> load
    """

    def __init__(self, max_parallel_inference: int = 1):
        self._model_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(max_parallel_inference)
        self._playback_lock = asyncio.Lock()
        self._playback_proc = [None]

    @asynccontextmanager
    async def model_lock(self):
        """Contexto para operaciones de ciclo de vida del modelo (load/unload/switch)."""
        async with self._model_lock:
            # Exclusión mutua con la inferencia: espera a la que esté en curso
            # y bloquea nuevas mientras se manipula el modelo.
            async with self._inference_semaphore:
                yield

    @asynccontextmanager
    async def inference_lock(self):
        """Contexto para inferencia GPU (generación, transcripción, cloning)."""
        async with self._inference_semaphore:
            yield

    @asynccontextmanager
    async def playback(self, timeout: int):
        """Contexto que serializa reproducciones: espera (o cancela) la anterior."""
        async with self._playback_lock:
            proc = self._playback_proc[0]
            if proc is not None and proc.poll() is None:
                logger.info(f"Esperando a que termine la reproducción anterior (máx. {timeout}s)...")
                try:
                    await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.error(f"Timeout esperando a la reproducción anterior ({timeout}s)")
                    proc.kill()
                    await asyncio.to_thread(proc.wait)
                    raise HTTPException(
                        504,
                        f"Timeout esperando a que termine la reproducción anterior ({timeout}s)",
                    )
            self._playback_proc[0] = None
            yield
        # La reproducción queda registrada por el llamador antes de salir del lock

    def register_playback(self, proc):
        """Registrar el proceso de reproducción en curso."""
        self._playback_proc[0] = proc
