#!/usr/bin/env python3
"""Tests unitarios de la separación model_lock / inference_lock."""

import asyncio

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
