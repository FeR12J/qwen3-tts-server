#!/usr/bin/env python3
"""Sincronización de operaciones sobre modelos e inferencia GPU.

Cola interna (en proceso, sin servicios externos):

    HTTP requests -> asyncio.Queue (FIFO, max_size) -> worker GPU -> inferencia

Con ``enabled`` activado, cada petición de inferencia obtiene un turno en la
cola en vez de acumularse en el semáforo: si la cola está llena se responde
429 (Too Many Requests) y el servidor no acumula esperas ilimitadas; durante
el apagado se responde 503 (Service Unavailable). Un worker de GPU concede
los turnos de uno en uno (o tantos como ``max_parallel_inference``) y espera
a que la petición termine antes de atender la siguiente. Con ``enabled``
desactivado se usa directamente el semáforo, como antes.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import HTTPException

logger = logging.getLogger("tts")

# Marcador de fin de cola (apagado ordenado)
_END = object()


class _QueueToken:
    """Turno de ejecución concedido por el worker a una petición."""

    __slots__ = ("_granted", "_done")

    def __init__(self):
        self._granted = asyncio.Event()
        self._done = asyncio.Event()

    async def wait_granted(self) -> None:
        """Esperar a que el worker nos conceda el turno."""
        await self._granted.wait()

    def grant(self) -> None:
        """Conceder el turno (llamado por el worker)."""
        self._granted.set()

    def finish(self) -> None:
        """Notificar al worker que la inferencia ha terminado."""
        self._done.set()

    async def wait_done(self) -> None:
        """Esperar a que la petición termine su inferencia (llamado por el worker)."""
        await self._done.wait()


class QueueService:
    """Locks separados para el ciclo de vida del modelo y la inferencia GPU.

    - ``model_lock``: serializa load/unload/switch. Al ejecutarse, espera a
      que termine la inferencia en curso y bloquea nuevas inferencias, de
      modo que el modelo no cambie mientras se genera audio.
    - ``inference_lock``: controla la inferencia GPU (TTS, voice cloning,
      Whisper). Con la cola activada, las peticiones se encolan FIFO y un
      worker de GPU concede el turno; sin cola, la exclusión la da el
      semáforo. La exclusión mutua evita carreras como:

          Request A -> unload
          Request B -> inference   (modelo a mitad de descarga)
          Request C -> load

    REGLA ARQUITECTÓNICA (validación fuera de la GPU):
    Nunca se valida input caro dentro de inference_lock. Orden obligatorio:

        validate request  -> validate text  -> validate audio
        -> acquire lock -> prepare model -> inference

    La GPU solo debe ocuparse preparando el modelo y generando; un upload
    inválido debe fallar con 400 sin haber bloqueado la inferencia.
    """

    def __init__(self, max_parallel_inference: int = 1, enabled: bool = False,
                 max_size: int = 4):
        self._model_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(max_parallel_inference)
        self._playback_lock = asyncio.Lock()
        self._playback_proc = [None]

        self._queue_enabled = bool(enabled)
        self._queue = asyncio.Queue(maxsize=max_size) if enabled else None
        self._workers_started = False
        self._stopping = False
        self._worker_tasks = []

    # -- Cola interna ------------------------------------------------------

    @property
    def queue_size(self) -> int:
        """Peticiones actualmente en espera en la cola interna (0 si desactivada)."""
        if self._queue is None:
            return 0
        return self._queue.qsize()

    def start(self):
        """Arrancar los workers de la cola (llamado en el inicio del servidor).

        También se arrancan de forma perezosa en el primer ``inference_lock``
        si no se llamó a ``start()`` (p.ej. en tests sin lifespan).
        """
        if self._queue is None or self._workers_started or self._stopping:
            return
        self._workers_started = True
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop())
            for _ in range(self._inference_semaphore._value)
        ]

    async def stop(self):
        """Detener los workers tras drenar los jobs pendientes (shutdown).

        No corta ninguna inferencia: el worker termina los turnos ya
        concedidos y encolados antes de salir.
        """
        if self._queue is None or not self._workers_started:
            return
        self._stopping = True
        for _ in self._worker_tasks:
            self._queue.put_nowait(_END)
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []

    async def _worker_loop(self):
        """Worker de GPU: concede turnos de la cola FIFO y espera a que la
        petición termine antes de atender la siguiente."""
        while True:
            token = await self._queue.get()
            self._queue.task_done()
            if token is _END:
                break
            try:
                token.grant()
                await token.wait_done()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Error en worker de cola de inferencia: {e}")
            if self._stopping and self._queue.empty():
                break

    # -- Locks -------------------------------------------------------------

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
        """Contexto para inferencia GPU (generación, transcripción, cloning).

        Con la cola activada, la petición se encola (FIFO, máximo
        ``max_size`` en espera; si la cola está llena, 429) y el worker le
        concede el turno de ejecución. Sin cola, se usa el semáforo.
        """
        if self._queue is None:
            async with self._inference_semaphore:
                yield
            return

        self.start()  # arranque perezoso del worker
        if self._stopping:
            # El servidor está en apagado: no se aceptan peticiones nuevas.
            raise HTTPException(
                503,
                "Servidor en proceso de apagado: no se aceptan nuevas peticiones.",
            )
        token = _QueueToken()
        try:
            self._queue.put_nowait(token)
        except asyncio.QueueFull:
            # Servidor activo pero la cola de espera está llena: la petición
            # debe reintentarse más tarde.
            raise HTTPException(
                429,
                f"Cola de inferencia llena ({self._queue.maxsize} peticiones en espera). "
                "Reintenta en unos segundos.",
            )
        try:
            await token.wait_granted()
            async with self._inference_semaphore:
                yield
        finally:
            # Liberar al worker aunque la petición se cancele o falle
            token.finish()

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

    async def stop_playback(self) -> None:
        """Detener la reproducción en curso, si la hay (usado en el apagado)."""
        async with self._playback_lock:
            proc = self._playback_proc[0]
            if proc is not None and proc.poll() is None:
                proc.kill()
                await asyncio.to_thread(proc.wait)
                logger.info("Reproducción en curso detenida")
            self._playback_proc[0] = None
