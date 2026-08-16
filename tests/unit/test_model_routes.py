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

from config.settings import settings
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


@pytest.fixture(autouse=True)
def isolated_api_keys(tmp_path, monkeypatch):
    """Aislar el store de claves API en un archivo temporal vacío.

    Estos tests ejercitan la lógica de las rutas, no la autenticación
    (cubierta en test_apikey_service): sin esto, un data/apikeys.json real
    con claves saldría del modo bootstrap y todas las rutas ADMIN
    responderían 401.
    """
    import storage.api_key_storage as aks
    from services import apikey_service

    monkeypatch.setattr(aks, "APIKEYS_FILE", str(tmp_path / "apikeys.json"))
    monkeypatch.setattr(apikey_service, "_keys", None)
    monkeypatch.setattr(apikey_service, "_keys_mtime", None)


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


def test_models_status_excludes_whisper(client):
    """Los modelos Whisper no aparecen en la tabla de modelos TTS."""
    tc, models, ctx = client
    orig = models.list_local_models
    models.list_local_models = lambda: orig() + ["whisper-large-v3", "whisper-small"]
    res = tc.get("/models/status")
    assert res.status_code == 200
    names = {r["model"] for r in res.json()["models"]}
    assert "whisper-large-v3" not in names
    assert "whisper-small" not in names
    assert "otro-modelo-local" in names


def test_model_load_whisper_rejected_400(client):
    """/model/load rechaza modelos Whisper con un error claro (400)."""
    tc, _, _ = client
    res = tc.post("/model/load", json={"model_id": "whisper-large-v3"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_MODEL_TYPE"
    assert "transcripción" in res.json()["error"]["message"]


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


# -- Descarga de modelos (whitelist) ------------------------------------------


@pytest.fixture
def download_env(monkeypatch, tmp_path):
    """Sin red: models_dir temporal, stub de snapshot y estado limpio."""
    import services.model_downloader as md

    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(settings.paths, "models_dir", str(d))
    md._STATE.clear()
    def stub_snapshot(model):
        t = d / model["name"]
        t.mkdir(exist_ok=True)
        (t / "model.safetensors").write_bytes(b"x" * 16)

    monkeypatch.setattr(md, "_snapshot_download", stub_snapshot)
    return d


def test_download_status_public(client):
    tc, _, _ = client
    res = tc.get("/models/download/status")
    assert res.status_code == 200
    models = res.json()["models"]
    names = {m["name"] for m in models}
    assert "Qwen3-TTS-12Hz-1.7B-VoiceDesign" in names
    assert "whisper-large-v3" in names
    assert all("repo_id" in m and "installed" in m for m in models)


def test_download_whitelist_rejects_unknown(client, download_env):
    tc, _, _ = client
    res = tc.post("/models/download", json={"model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MODEL_NOT_SUPPORTED"


def test_download_starts_and_tracks_state(client, download_env):
    import asyncio

    import services.model_downloader as md

    tc, _, _ = client
    res = tc.post("/models/download", json={"model_id": "whisper-large-v3"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["started"] is True

    # El loop del TestClient puede cerrarse antes de que la tarea en segundo
    # plano marque el estado; forzar la finalización si sigue pendiente.
    if md._STATE["whisper-large-v3"]["status"] == "downloading":
        asyncio.run(md._run_download(md.SUPPORTED_BY_NAME["whisper-large-v3"]))

    res = tc.get("/models/download/status")
    by_name = {m["name"]: m for m in res.json()["models"]}
    assert by_name["whisper-large-v3"]["installed"] is True
    assert by_name["whisper-large-v3"]["status"] == "done"


def test_download_invalid_id(client, download_env):
    tc, _, _ = client
    res = tc.post("/models/download", json={"model_id": "  "})
    assert res.status_code == 400