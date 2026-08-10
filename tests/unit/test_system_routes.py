#!/usr/bin/env python3
"""Tests de los endpoints de salud y estado del sistema (/health, /ready,
/system/status, /version)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import VERSION
from routes.system import create_system_routes


class DummyModels:
    def __init__(self, loaded=True):
        self._loaded = loaded
        self.status = {
            "state": "ready" if loaded else "unloaded",
            "model_id": "Qwen3-TTS-12Hz-1.7B-VoiceDesign" if loaded else None,
        }

    def is_loaded(self):
        return self._loaded

    async def get_active_model(self):
        return None

    async def get_model_status(self):
        return self.status


class DummyVoices:
    clone_active = False


class DummyMetrics:
    def vram_available_gb(self):
        return 0.0


class Ctx:
    def __init__(self, models):
        self.models = models
        self.voices = DummyVoices()
        self.metrics = DummyMetrics()


@pytest.fixture
def client():
    app = FastAPI()
    create_system_routes(app, Ctx(DummyModels(loaded=True)))
    return TestClient(app)


def test_health_is_fast_and_simple(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ready(client):
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"ready": True, "tts_model_loaded": True}


def test_ready_model_unloaded():
    app = FastAPI()
    create_system_routes(app, Ctx(DummyModels(loaded=False)))
    res = TestClient(app).get("/ready")
    assert res.json()["tts_model_loaded"] is False


def test_system_status_shape(client):
    res = client.get("/system/status")
    assert res.status_code == 200
    data = res.json()
    assert set(data.keys()) == {"gpu", "tts", "whisper"}
    assert set(data["gpu"].keys()) == {
        "available", "name", "total_vram_mb", "used_vram_mb", "free_vram_mb",
    }
    assert set(data["tts"].keys()) == {"state", "model"}
    assert data["tts"]["state"] == "ready"
    assert set(data["whisper"].keys()) == {"state"}
    assert data["whisper"]["state"] in ("loaded", "unloaded")


def test_version_shape(client):
    res = client.get("/version")
    assert res.status_code == 200
    data = res.json()
    assert data["server"] == "qwen3-tts-server"
    assert data["version"] == VERSION
    for key in ("qwen_tts", "torch", "transformers", "cuda"):
        assert key in data
