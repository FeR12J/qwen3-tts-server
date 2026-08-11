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
