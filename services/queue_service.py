#!/usr/bin/env python3
"""Cola de inferencia y serialización de reproducciones."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import HTTPException

logger = logging.getLogger("tts")


class QueueService:
    """Serializa la inferencia (semáforo) y la reproducción local de audio."""

    def __init__(self, max_parallel: int = 1):
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._playback_lock = asyncio.Lock()
        self._playback_proc = [None]

    @asynccontextmanager
    async def infer(self):
        """Contexto que serializa la inferencia de modelos."""
        async with self._semaphore:
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
