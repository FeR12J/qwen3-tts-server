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

    def get_metrics(self):
        return {
            "tts_requests": 120,
            "tts_errors": 3,
            "tts_active": 1,
            "tts_queue_size": 0,
            "whisper_requests": 42,
            "model_loads": 7,
            "model_unloads": 6,
            "average_tts_ms": 843,
            "average_queue_wait_ms": 120,
            "average_model_load_ms": 3500,
            "average_rtf": 0.16,
            "average_ttfb_ms": 780,
            "average_vram_used_mb": 4100,
        }


class DummyQueue:
    running = 2
    queue_size = 1

    @property
    def active_requests(self):
        return self.running + self.queue_size


class Ctx:
    def __init__(self, models):
        self.models = models
        self.voices = DummyVoices()
        self.metrics = DummyMetrics()
        self.queue = DummyQueue()


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
    assert set(data.keys()) == {"gpu", "tts", "whisper", "storage"}
    assert set(data["gpu"].keys()) == {
        "available", "name", "total_vram_mb", "used_vram_mb", "free_vram_mb",
    }
    assert set(data["tts"].keys()) == {
        "state", "model", "running", "waiting", "active_requests",
    }
    assert data["tts"]["state"] == "ready"
    assert data["tts"]["running"] == 2
    assert data["tts"]["waiting"] == 1
    assert data["tts"]["active_requests"] == 3
    assert set(data["whisper"].keys()) == {"model", "model_loaded", "state", "device"}
    assert data["whisper"]["state"] in ("loaded", "unloaded")
    for key in ("voices", "temporaries"):
        assert set(data["storage"][key].keys()) == {"path", "exists", "files", "size_mb"}


def test_version_shape(client):
    res = client.get("/version")
    assert res.status_code == 200
    data = res.json()
    assert data["server"] == "qwen3-tts-server"
    assert data["version"] == VERSION
    for key in ("qwen_tts", "torch", "transformers", "cuda"):
        assert key in data


def test_metrics_shape(client):
    """GET /metrics devuelve las métricas sencillas esperadas."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert res.json() == {
        "tts_requests": 120,
        "tts_errors": 3,
        "tts_active": 1,
        "tts_queue_size": 0,
        "whisper_requests": 42,
        "model_loads": 7,
        "model_unloads": 6,
        "average_tts_ms": 843,
        "average_queue_wait_ms": 120,
        "average_model_load_ms": 3500,
        "average_rtf": 0.16,
        "average_ttfb_ms": 780,
        "average_vram_used_mb": 4100,
    }
