#!/usr/bin/env python3
"""Tests unitarios de la separación model_lock / inference_lock y de la
cola interna (asyncio.Queue + worker GPU)."""

import asyncio

import pytest

from services.errors import APIError, QueueFullError
from services.queue_service import QueueService


def test_model_lock_waits_for_inference_in_progress():
    async def scenario():
        q = QueueService()
        order = []

        async def inference():
            async with q.inference_lock():
                order.append("infer-start")
                await asyncio.sleep(0.1)
                order.append("infer-end")

        async def model_op():
            async with q.model_lock():
                order.append("model")

        t1 = asyncio.create_task(inference())
        await asyncio.sleep(0.02)
        t2 = asyncio.create_task(model_op())
        await asyncio.gather(t1, t2)

        assert order == ["infer-start", "infer-end", "model"]

    asyncio.run(scenario())


def test_inference_waits_for_model_op():
    async def scenario():
        q = QueueService()
        order = []

        async def model_op():
            async with q.model_lock():
                order.append("model-start")
                await asyncio.sleep(0.1)
                order.append("model-end")

        async def inference():
            async with q.inference_lock():
                order.append("infer")

        t1 = asyncio.create_task(model_op())
        await asyncio.sleep(0.02)
        t2 = asyncio.create_task(inference())
        await asyncio.gather(t1, t2)

        assert order == ["model-start", "model-end", "infer"]

    asyncio.run(scenario())


def test_new_inference_blocked_during_model_op():
    async def scenario():
        q = QueueService()
        order = []

        async def model_op():
            async with q.model_lock():
                order.append("model-start")
                await asyncio.sleep(0.1)
                order.append("model-end")

        async def inference():
            async with q.inference_lock():
                order.append("infer")

        t1 = asyncio.create_task(model_op())
        await asyncio.sleep(0.02)
        t2 = asyncio.create_task(inference())  # se encola tras la operación de modelo
        await asyncio.gather(t1, t2)

        assert order == ["model-start", "model-end", "infer"]

    asyncio.run(scenario())


def test_inferences_serialized_by_default():
    """Con max_parallel_inference=1 las inferencias no se solapan."""
    async def scenario():
        q = QueueService()
        order = []

        async def inference(i):
            async with q.inference_lock():
                order.append(f"start{i}")
                await asyncio.sleep(0.05)
                order.append(f"end{i}")

        await asyncio.gather(inference(1), inference(2))

        assert len(order) == 4
        # la primera en empezar debe terminar antes de que empiece la segunda
        first = order[0]
        assert order[1] == first.replace("start", "end")
        assert order[2].startswith("start")
        assert order[3].startswith("end")

    asyncio.run(scenario())


def test_parallel_inferences_with_semaphore():
    """Con max_parallel_inference=2 las inferencias pueden solaparse."""
    async def scenario():
        q = QueueService(max_parallel_inference=2)
        order = []

        async def inference(i):
            async with q.inference_lock():
                order.append(f"start{i}")
                await asyncio.sleep(0.05)
                order.append(f"end{i}")

        await asyncio.gather(inference(1), inference(2))

        starts = [o for o in order if o.startswith("start")]
        assert len(starts) == 2  # ambas empezaron antes de terminar ninguna
        assert order[:2] == starts

    asyncio.run(scenario())


def test_model_lock_excludes_all_inferences_with_parallelism():
    """Con max_parallel_inference=2, model_lock excluye a TODAS las
    inferencias (un solo slot de semáforo no bastaba: N-1 podían correr en
    paralelo con la carga/descarga de modelo)."""
    async def scenario():
        q = QueueService(max_parallel_inference=2)
        order = []
        active = 0
        max_active_during_model_op = 0

        async def inference(i):
            nonlocal active
            async with q.inference_lock():
                active += 1
                order.append(f"infer{i}-start")
                await asyncio.sleep(0.1)
                active -= 1
                order.append(f"infer{i}-end")

        async def model_op():
            nonlocal max_active_during_model_op
            # Dos inferencias en curso: la operación de modelo debe esperar
            # a que AMBAS terminen.
            await asyncio.gather(inference(1), inference(2))
            async with q.model_lock():
                max_active_during_model_op = max(
                    max_active_during_model_op, active
                )
                order.append("model")
                await asyncio.sleep(0.02)

        await model_op()
        assert order.index("model") > order.index("infer1-end")
        assert order.index("model") > order.index("infer2-end")
        assert max_active_during_model_op == 0

    asyncio.run(scenario())


def test_no_inference_starts_during_model_op_with_parallelism():
    """Con N=2, una inferencia que llega durante una model_lock espera a que
    termine (no se cuela por el otro slot del semáforo)."""
    async def scenario():
        q = QueueService(max_parallel_inference=2)
        order = []

        async def model_op():
            async with q.model_lock():
                order.append("model-start")
                await asyncio.sleep(0.1)
                order.append("model-end")

        async def inference(i):
            async with q.inference_lock():
                order.append(f"infer{i}")

        t1 = asyncio.create_task(model_op())
        await asyncio.sleep(0.02)
        t2 = asyncio.create_task(inference(1))
        t3 = asyncio.create_task(inference(2))
        await asyncio.gather(t1, t2, t3)

        assert order.index("model-end") < order.index("infer1")
        assert order.index("model-end") < order.index("infer2")

    asyncio.run(scenario())


def test_model_lock_not_starved_by_continuous_inferences():
    """Una operación de modelo en espera no se hambreiza bajo carga sostenida.

    model_lock debe bloquear las nuevas inferencias (clear de _model_idle)
    ANTES de esperar a que las activas terminen: si espera con _model_idle
    aún set, una presión continua se cuela una y otra vez (reentra sin
    ceder el event loop al terminar cada turno) y el load/unload/switch se
    retrasa indefinidamente. Con el bug, wait_for lanza TimeoutError."""
    async def scenario():
        q = QueueService(max_parallel_inference=2)
        active = 0
        stop_pressure = asyncio.Event()

        async def inference(i):
            nonlocal active
            async with q.inference_lock():
                active += 1
                await asyncio.sleep(0.02)
                active -= 1

        async def pressure():
            i = 0
            while not stop_pressure.is_set():
                i += 1
                await inference(i)

        async def model_op():
            # Dejar que la presión tenga inferencias en curso al empezar
            await asyncio.sleep(0.05)
            async with q.model_lock():
                # Exclusión total: ninguna inferencia activa dentro de la
                # sección de manipulación del modelo.
                assert active == 0

        pressure_task = asyncio.create_task(pressure())
        model_task = asyncio.create_task(model_op())
        # Margen holgado: la operación debe esperar solo al turno en curso.
        await asyncio.wait_for(model_task, timeout=2.0)
        stop_pressure.set()
        await pressure_task

    asyncio.run(scenario())


# -- Cola interna (HTTP -> asyncio.Queue -> worker GPU) ---------------------


def test_queue_fifo_order():
    """Con la cola activada, el worker concede turnos en orden FIFO."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=True, max_size=10)
        q.start()
        order = []

        async def inference(i):
            async with q.inference_lock():
                order.append(("start", i))
                await asyncio.sleep(0.02)
                order.append(("end", i))

        await asyncio.gather(*[asyncio.create_task(inference(i)) for i in (1, 2, 3)])
        assert [i for t, i in order if t == "start"] == [1, 2, 3]
        await q.stop()

    asyncio.run(scenario())


def test_queue_full_rejects_with_429():
    """Con max_size=2, la tercera petición en espera recibe 429."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=True, max_size=2)
        q.start()
        release = asyncio.Event()

        async def inference(i):
            async with q.inference_lock():
                if i == 1:
                    await release.wait()

        t1 = asyncio.create_task(inference(1))
        await asyncio.sleep(0.05)  # t1 en ejecución (worker concedió)
        t2 = asyncio.create_task(inference(2))
        t3 = asyncio.create_task(inference(3))
        await asyncio.sleep(0.05)  # t2 y t3 ocupando la cola (2 en espera)

        with pytest.raises(QueueFullError) as exc:
            async with q.inference_lock():
                pass
        assert exc.value.status_code == 429
        assert exc.value.code == "QUEUE_FULL"

        release.set()
        await asyncio.gather(t1, t2, t3)
        await q.stop()

    asyncio.run(scenario())


def test_queue_rejects_with_503_during_shutdown():
    """Durante el apagado (stop() en curso) se responde 503, no 429."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=True, max_size=4)
        q.start()
        release = asyncio.Event()

        async def inference():
            async with q.inference_lock():
                await release.wait()

        t1 = asyncio.create_task(inference())
        await asyncio.sleep(0.05)  # t1 en ejecución
        stop_task = asyncio.create_task(q.stop())
        await asyncio.sleep(0.05)  # stop() en curso, workers drenando

        with pytest.raises(APIError) as exc:
            async with q.inference_lock():
                pass
        assert exc.value.status_code == 503
        assert exc.value.code == "SERVICE_UNAVAILABLE"

        release.set()
        await asyncio.gather(t1, stop_task)

    asyncio.run(scenario())


def test_queue_waits_when_not_full():
    """Si hay hueco en la cola, la petición espera su turno sin 429."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=True, max_size=4)
        q.start()
        release = asyncio.Event()
        done = []

        async def inference(i):
            async with q.inference_lock():
                done.append(i)
                if i == 1:
                    await release.wait()

        t1 = asyncio.create_task(inference(1))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(inference(2))
        await asyncio.sleep(0.05)
        assert done == [1]          # t2 en espera de turno
        release.set()
        await asyncio.gather(t1, t2)
        assert done == [1, 2]       # t2 ejecutó después
        await q.stop()

    asyncio.run(scenario())


def test_queue_stop_drains_pending_jobs():
    """stop() no corta inferencias: termina las encoladas antes de salir."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=True, max_size=4)
        q.start()
        release = asyncio.Event()
        done = []

        async def inference(i):
            async with q.inference_lock():
                done.append(f"s{i}")
                if i == 1:
                    await release.wait()
                done.append(f"e{i}")

        t1 = asyncio.create_task(inference(1))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(inference(2))
        await asyncio.sleep(0.05)   # t2 encolado

        stop_task = asyncio.create_task(q.stop())
        await asyncio.sleep(0.05)   # stop esperando al worker
        release.set()
        await asyncio.gather(t1, t2, stop_task)

        assert done == ["s1", "e1", "s2", "e2"]

    asyncio.run(scenario())


def test_queue_stop_with_full_queue():
    """stop() con la cola llena no lanza QueueFull (put_nowait lo hacía y
    dejaba workers huérfanos): los marcadores _END esperan su hueco y los
    workers terminan drenando."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=True, max_size=2)
        q.start()
        release = asyncio.Event()
        done = []

        async def inference(i):
            async with q.inference_lock():
                done.append(f"s{i}")
                if i == 1:
                    await release.wait()
                done.append(f"e{i}")

        t1 = asyncio.create_task(inference(1))
        await asyncio.sleep(0.05)   # t1 en ejecución
        t2 = asyncio.create_task(inference(2))
        t3 = asyncio.create_task(inference(3))
        await asyncio.sleep(0.05)   # t2 y t3 llenan la cola (max_size=2)
        assert q.queue_size == 2

        stop_task = asyncio.create_task(q.stop())
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(t1, t2, t3, stop_task)

        # stop() drena también lo encolado (t3), sin lanzar QueueFull
        assert done == ["s1", "e1", "s2", "e2", "s3", "e3"]
        assert q._worker_tasks == []  # los workers han terminado

    asyncio.run(scenario())


def test_queue_disabled_never_429():
    """Sin cola, las peticiones esperan en el semáforo (sin 429)."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=False, max_size=1)
        release = asyncio.Event()
        done = []

        async def inference(i):
            async with q.inference_lock():
                done.append(i)
                if i == 1:
                    await release.wait()

        t1 = asyncio.create_task(inference(1))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(inference(2))
        await asyncio.sleep(0.05)
        assert done == [1]
        release.set()
        await asyncio.gather(t1, t2)
        assert done == [1, 2]

    asyncio.run(scenario())


def test_active_requests_with_queue():
    """active_requests = en ejecución + en espera (cola FIFO)."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=True, max_size=4)
        q.start()
        release = asyncio.Event()
        done = []

        async def inference(i):
            async with q.inference_lock():
                done.append(i)
                if i == 1:
                    await release.wait()

        t1 = asyncio.create_task(inference(1))
        await asyncio.sleep(0.05)   # t1 en ejecución
        t2 = asyncio.create_task(inference(2))
        await asyncio.sleep(0.05)   # t2 en espera

        assert q.running == 1
        assert q.queue_size == 1
        assert q.active_requests == 2

        release.set()
        await asyncio.gather(t1, t2)
        assert q.running == 0
        assert q.queue_size == 0
        assert q.active_requests == 0
        await q.stop()

    asyncio.run(scenario())


def test_active_requests_without_queue():
    """Sin cola, lo en espera del semáforo cuenta como en ejecución."""
    async def scenario():
        q = QueueService(max_parallel_inference=1, enabled=False, max_size=4)
        release = asyncio.Event()
        done = []

        async def inference(i):
            async with q.inference_lock():
                done.append(i)
                if i == 1:
                    await release.wait()

        t1 = asyncio.create_task(inference(1))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(inference(2))
        await asyncio.sleep(0.05)

        assert q.running == 1
        assert q.queue_size == 0
        assert q.active_requests == 1

        release.set()
        await asyncio.gather(t1, t2)
        assert q.active_requests == 0

    asyncio.run(scenario())
