#!/usr/bin/env python3
"""Tests de las rutas de gestión de modelos (/models/status, /model/load,
/model/activate, /model/unload)."""

import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.models import create_models_routes


class DummyQueue:
    @asynccontextmanager
    async def model_lock(self):
        yield


class FakeModels:
    def __init__(self):
        self.registry = {}
        self.active = None

    def _entry(self, state, dtype=None, device=None, error=None):
        return {"type": "qwen3-tts", "state": state, "device": device,
                "dtype": dtype, "loaded_at": "2026-01-01T00:00:00Z",
                "error": error}

    def seed(self):
        self.registry["Qwen3-TTS-12Hz-1.7B-VoiceDesign"] = self._entry("ready", "bfloat16", "cuda:0")
        self.registry["Qwen3-TTS-0.6B"] = self._entry("ready", "bfloat16", "cuda:0")
        self.active = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    def list_local_models(self):
        return list(self.registry) + ["otro-modelo-local"]

    def list_models_status(self):
        rows = []
        for mid in self.list_local_models():
            e = self.registry.get(mid)
            if e is None:
                rows.append({"model": mid, "type": None, "state": "unloaded",
                             "device": None, "dtype": None, "loaded_at": None,
                             "error": None, "active": False})
            else:
                rows.append({"model": mid, "type": e["type"], "state": e["state"],
                             "device": e["device"], "dtype": e["dtype"],
                             "loaded_at": e["loaded_at"], "error": e["error"],
                             "active": mid == self.active})
        return rows

    def is_loaded_model(self, model_id):
        e = self.registry.get(model_id)
        return e is not None and e["state"] == "ready"

    async def get_active_model(self):
        if self.active is None:
            return None
        return type("M", (), {"model_id": self.active})()

    async def switch_model(self, model_id):
        self.active = model_id
        e = self.registry[model_id]
        return type("M", (), {"model_id": model_id, "model_type": e["type"]})()

    async def unload_model(self, model_id):
        self.registry.pop(model_id, None)
        if self.active == model_id:
            self.active = None


class FakeVoices:
    def __init__(self):
        self.unloaded = False

    def unload_voice(self):
        self.unloaded = True


class FakeMetrics:
    def vram_available_gb(self):
        return 0.0


@pytest.fixture
def client():
    from app import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    models = FakeModels()
    models.seed()
    ctx = type("Ctx", (), {
        "queue": DummyQueue(), "models": models,
        "voices": FakeVoices(), "metrics": FakeMetrics(),
    })()
    create_models_routes(app, ctx)
    return TestClient(app), models, ctx


def test_models_status_shape(client):
    tc, models, ctx = client
    res = tc.get("/models/status")
    assert res.status_code == 200
    rows = res.json()["models"]
    assert len(rows) == 3
    for row in rows:
        assert set(row.keys()) == {
            "model", "type", "state", "device", "dtype", "loaded_at", "error", "active",
        }
    active = [r for r in rows if r["active"]]
    assert len(active) == 1
    assert active[0]["model"] == "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    unloaded = [r for r in rows if r["model"] == "otro-modelo-local"][0]
    assert unloaded["state"] == "unloaded"
    assert unloaded["device"] is None


def test_model_activate_loaded(client):
    tc, models, ctx = client
    res = tc.post("/model/activate", json={"model_id": "Qwen3-TTS-0.6B"})
    assert res.status_code == 200
    assert res.json()["loaded_model"] == "Qwen3-TTS-0.6B"
    assert models.active == "Qwen3-TTS-0.6B"


def test_model_activate_not_loaded_409(client):
    tc, models, ctx = client
    res = tc.post("/model/activate", json={"model_id": "otro-modelo-local"})
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "MODEL_NOT_LOADED"


def test_model_activate_invalid_id(client):
    tc, models, ctx = client
    res = tc.post("/model/activate", json={"model_id": "  "})
    assert res.status_code == 400


def test_model_unload_specific(client):
    tc, models, ctx = client
    res = tc.post("/model/unload", json={"model_id": "Qwen3-TTS-0.6B"})
    assert res.status_code == 200
    assert res.json()["unloaded_model"] == "Qwen3-TTS-0.6B"
    assert "Qwen3-TTS-0.6B" not in models.registry
    # no era el activo: la voz clonada no se toca
    assert ctx.voices.unloaded is False
    # el modelo activo sigue intacto
    assert models.active == "Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def test_model_unload_active_confirms_voice_reset(client):
    tc, models, ctx = client
    res = tc.post("/model/unload", json={"model_id": "Qwen3-TTS-12Hz-1.7B-VoiceDesign"})
    assert res.status_code == 200
    assert models.active is None
    assert ctx.voices.unloaded is True


def test_model_unload_without_body_unloads_active(client):
    tc, models, ctx = client
    res = tc.post("/model/unload")
    assert res.status_code == 200
    assert res.json()["unloaded_model"] == "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert models.active is None


def test_model_unload_not_loaded_is_noop(client):
    tc, models, ctx = client
    res = tc.post("/model/unload", json={"model_id": "otro-modelo-local"})
    assert res.status_code == 200
    assert "no estaba cargado" in res.json()["message"]