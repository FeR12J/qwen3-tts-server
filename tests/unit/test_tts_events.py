#!/usr/bin/env python3
"""Tests del logging estructurado: cada petición lleva request_id y emite
eventos key=value (tts_started / tts_completed / tts_chunk_emitted)."""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from schemas.tts import TTSRequest
from services import config_service
from services.audio_service import AudioService
from services.model_manager import ModelInfo
from services.tts_service import TTSService

SR = 24000


class DummyQueue:
    @asynccontextmanager
    async def inference_lock(self):
        yield


class DummyModels:
    def __init__(self):
        self.info = ModelInfo("test-base", "base", None, None)

    async def get_active_model(self):
        return self.info


class DummyVoices:
    clone_prompt = "voz cargada de prueba"


async def _noop(*args, **kwargs):
    pass


def _events(caplog, event_name: str) -> list:
    return [
        r.getMessage() for r in caplog.records
        if f"event={event_name}" in r.getMessage()
    ]


def _make_service():
    audio = AudioService(SimpleNamespace(paths=SimpleNamespace(audios_dir=".")), None)
    svc = TTSService(
        config=SimpleNamespace(),
        queue=DummyQueue(),
        model_manager=DummyModels(),
        voice_manager=DummyVoices(),
        audio_service=audio,
        metrics=None,
    )

    async def fake_generate_one(request, text, info):
        return [np.full(1000, 0.3, dtype="float32")], SR

    svc._generate_one = fake_generate_one
    return svc


def test_synthesize_events_with_request_id(caplog, monkeypatch):
    """synthesize emite tts_started y tts_completed con el mismo request_id."""
    monkeypatch.setattr("services.gpu_management.prepare_for_tts", _noop)
    caplog.set_level(logging.INFO, logger="tts")
    svc = _make_service()

    req = SimpleNamespace(headers={"x-request-id": "rid123"})
    asyncio.run(svc.synthesize(TTSRequest(text="Hola mundo"), http_request=req))

    started = _events(caplog, "tts_started")
    completed = _events(caplog, "tts_completed")
    assert len(started) == 1
    assert len(completed) == 1

    # request_id idéntico en ambos eventos (el del header x-request-id)
    assert "request_id=rid123" in started[0]
    assert "request_id=rid123" in completed[0]

    # Campos del evento de inicio
    assert "model=test-base" in started[0]
    assert "model_type=base" in started[0]
    assert "text_length=10" in started[0]

    # Campos del evento final
    assert "duration_ms=" in completed[0]
    assert "audio_duration_ms=41" in completed[0]  # 1000/24000 s * 1000


def test_synthesize_failed_event(caplog, monkeypatch):
    """Un error de generación emite tts_failed con request_id."""
    monkeypatch.setattr("services.gpu_management.prepare_for_tts", _noop)
    caplog.set_level(logging.INFO, logger="tts")
    svc = _make_service()

    async def broken(request, text, info):
        raise RuntimeError("boom")

    svc._generate_one = broken

    req = SimpleNamespace(headers={"x-request-id": "rid999"})
    with pytest.raises(RuntimeError):
        asyncio.run(svc.synthesize(TTSRequest(text="Hola"), http_request=req))

    failed = _events(caplog, "tts_failed")
    assert len(failed) == 1
    assert "request_id=rid999" in failed[0]
    assert "duration_ms=" in failed[0]


def test_stream_chunk_events(caplog, monkeypatch):
    """stream_synthesize emite tts_started, tts_chunk_emitted y tts_completed."""
    monkeypatch.setattr("services.gpu_management.prepare_for_tts", _noop)
    monkeypatch.setattr(config_service.settings.runtime, "max_text_chars", 20)
    caplog.set_level(logging.INFO, logger="tts")
    svc = _make_service()

    async def fake_generate_one(request, text, info):
        return [np.full(800, 0.3, dtype="float32")], SR

    svc._generate_one = fake_generate_one

    text = "Frase uno. Frase dos. Frase tres. Frase cuatro. Frase cinco."
    req = SimpleNamespace(
        text=text,
        input=None,
        headers={"x-request-id": "rid456"},
    )
    plan = asyncio.run(svc.stream_plan(TTSRequest(text=text)))
    chunks = []
    asyncio.run(_collect(svc, req, plan, chunks))
    assert len(chunks) == 5

    started = _events(caplog, "tts_started")
    chunk_events = _events(caplog, "tts_chunk_emitted")
    completed = _events(caplog, "tts_completed")

    assert len(started) == 1 and "request_id=rid456" in started[0]
    assert f"text_length={len(text)}" in started[0]

    assert len(chunk_events) == 5
    assert "request_id=rid456" in chunk_events[0]
    assert "chunk_index=1" in chunk_events[0]
    assert "audio_duration_ms=33" in chunk_events[0]  # 800/24000 s * 1000

    assert len(completed) == 1
    assert "request_id=rid456" in completed[0]
    assert "audio_duration_ms=165" in completed[0]  # 5 * 33 ms
    assert "streaming=True" in completed[0]


async def _collect(svc, req, plan, out):
    async for r in svc.stream_synthesize(req, plan, http_request=req):
        out.append(r)


def test_generated_request_id_when_missing_header(caplog, monkeypatch):
    """Sin header x-request-id se genera un request_id (uuid hex)."""
    monkeypatch.setattr("services.gpu_management.prepare_for_tts", _noop)
    caplog.set_level(logging.INFO, logger="tts")
    svc = _make_service()

    asyncio.run(svc.synthesize(TTSRequest(text="Hola"), http_request=None))

    started = _events(caplog, "tts_started")
    completed = _events(caplog, "tts_completed")
    assert len(started) == 1 and len(completed) == 1
    rid_started = started[0].split()[0].split("=")[1]
    rid_completed = completed[0].split()[0].split("=")[1]
    assert rid_started == rid_completed
    assert len(rid_started) == 32  # uuid4().hex
