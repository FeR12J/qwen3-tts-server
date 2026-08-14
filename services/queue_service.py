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

from services.errors import APIError, QueueFullError

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
      que terminen TODAS las inferencias en curso y bloquea las nuevas
      (barrera flag + contador, válida con ``max_parallel_inference > 1``),
      de modo que el modelo no cambie mientras se genera audio.
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
        self._max_parallel = max_parallel_inference
        self._model_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(max_parallel_inference)
        self._playback_lock = asyncio.Lock()
        self._playback_proc = [None]

        self._queue = asyncio.Queue(maxsize=max_size) if enabled else None
        self._workers_started = False
        self._stopping = False
        self._worker_tasks = []
        self._running = 0

        # Barrera entre operaciones de modelo y la inferencia. Con
        # max_parallel_inference > 1 un solo slot de semáforo NO excluye a
        # todas las inferencias: se usa un evento (model_idle: set = NO hay
        # operación de modelo en curso) + contador de inferencias activas.
        # La atomicidad (sin await entre la comprobación y la mutación) la
        # garantiza el event loop.
        #
        # Nota: los eventos están invertidos a propósito. Esperar "a que el
        # evento se desmarque" con ``while ev.is_set(): await ev.wait()``
        # sería un bucle ocupado (wait() en un evento set devuelve sin
        # ceder el loop). Con el evento en su forma "disponible", wait()
        # siempre suspende cuando hay que esperar.
        self._model_idle = asyncio.Event()
        self._model_idle.set()
        self._inference_idle = asyncio.Event()
        self._inference_idle.set()
        self._inference_active = 0

    # -- Cola interna ------------------------------------------------------

    @property
    def queue_size(self) -> int:
        """Peticiones actualmente en espera en la cola interna (0 si desactivada)."""
        if self._queue is None:
            return 0
        return self._queue.qsize()

    @property
    def running(self) -> int:
        """Inferencias en curso (generación/transcripción/clonación)."""
        return self._running

    @property
    def active_requests(self) -> int:
        """Peticiones activas: en espera en la cola + en ejecución."""
        return self._running + self.queue_size

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
            for _ in range(self._max_parallel)
        ]

    async def stop(self):
        """Detener los workers tras drenar los jobs pendientes (shutdown).

        No corta ninguna inferencia: el worker termina los turnos ya
        concedidos y encolados antes de salir. Los marcadores _END se
        encolan con ``await put`` (bloqueante): si la cola está llena, se
        espera a que los workers la dreinen en vez de lanzar QueueFull y
        dejar workers huérfanos.
        """
        if self._queue is None or not self._workers_started:
            return
        self._stopping = True
        for _ in self._worker_tasks:
            await self._queue.put(_END)
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

    @property
    def inference_active(self) -> int:
        """Inferencias que ya pasaron la barrera y están usando (o a punto
        de usar) la GPU. Lo usan prepare_for_tts/prepare_for_whisper para no
        intercambiar modelos con otra inferencia en curso (N > 1)."""
        return self._inference_active

    @asynccontextmanager
    async def model_lock(self):
        """Contexto para operaciones de ciclo de vida del modelo (load/unload/switch).

        Exclusión mutua COMPLETA con la inferencia: espera a que no queden
        inferencias activas y bloquea las nuevas mientras se manipula el
        modelo. Con max_parallel_inference > 1 no basta agarrar un slot del
        semáforo (dejaría N-1 inferencias en paralelo); la barrera es el
        flag ``_model_op`` + el contador de inferencias activas.
        """
        async with self._model_lock:
            # Primero se bloquean las nuevas inferencias (clear de
            # _model_idle) y DESPUÉS se espera a que las activas terminen:
            # si se esperara con _model_idle aún set, peticiones nuevas
            # entrarían durante la espera (incrementando el contador) y la
            # operación de modelo podría hambrearse bajo carga sostenida.
            # _enter_inference espera a que _model_idle esté set antes de
            # incrementar, sin await entre la comprobación y el incremento
            # (atómico en el event loop), así que no se pierde ninguna.
            self._model_idle.clear()
            try:
                while self._inference_active > 0:
                    self._inference_idle.clear()
                    await self._inference_idle.wait()
                yield
            finally:
                self._model_idle.set()

    @asynccontextmanager
    async def inference_lock(self):
        """Contexto para inferencia GPU (generación, transcripción, cloning).

        Con la cola activada, la petición se encola (FIFO, máximo
        ``max_size`` en espera; si la cola está llena, 429) y el worker le
        concede el turno de ejecución. Sin cola, se usa el semáforo.
        """
        if self._queue is None:
            async with self._inference_semaphore:
                await self._enter_inference()
                try:
                    yield
                finally:
                    self._exit_inference()
            return

        self.start()  # arranque perezoso del worker
        if self._stopping:
            # El servidor está en apagado: no se aceptan peticiones nuevas.
            raise APIError(
                "SERVICE_UNAVAILABLE",
                "Servidor en proceso de apagado: no se aceptan nuevas peticiones.",
                503,
            )
        token = _QueueToken()
        try:
            self._queue.put_nowait(token)
        except asyncio.QueueFull:
            # Servidor activo pero la cola de espera está llena: la petición
            # debe reintentarse más tarde.
            raise QueueFullError(self._queue.maxsize)
        try:
            await token.wait_granted()
            async with self._inference_semaphore:
                await self._enter_inference()
                try:
                    yield
                finally:
                    self._exit_inference()
        finally:
            # Liberar al worker aunque la petición se cancele o falle
            token.finish()

    async def _enter_inference(self):
        """Entrar en la sección de inferencia: esperar a que no haya una
        operación de modelo en curso y marcarse como activo.

        ``_model_idle`` está set cuando NO hay operación de modelo: si no lo
        está, wait() suspende de verdad (no hay bucle ocupado). No hay await
        entre la comprobación y el incremento: si una model_lock arranca en
        ese intervalo, verá el contador ya incrementado y esperará.
        """
        while not self._model_idle.is_set():
            await self._model_idle.wait()
        self._inference_active += 1
        self._running += 1

    def _exit_inference(self):
        """Salir de la sección de inferencia y avisar si ya no quedan activas."""
        self._inference_active -= 1
        self._running -= 1
        if self._inference_active == 0:
            self._inference_idle.set()

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
                    raise APIError(
                        "PLAYBACK_TIMEOUT",
                        f"Timeout esperando a que termine la reproducción anterior ({timeout}s)",
                        504,
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
