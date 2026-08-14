#!/usr/bin/env python3
"""Tests de la ruta de configuración del panel (/webui/api/config), centrados
en el cambio de modelo Whisper: validación, persistencia en runtime y descarga
del modelo anterior si estaba cargado."""

import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings
from services import config_service, whisper_service
from routes.webui import create_webui_routes


class DummyQueue:
    @asynccontextmanager
    async def model_lock(self):
        yield


@pytest.fixture(autouse=True)
def isolated_api_keys(tmp_path, monkeypatch):
    """Aislar el store de claves API (modo bootstrap: sin claves, admin pasa)."""
    import storage.api_key_storage as aks
    from services import apikey_service

    monkeypatch.setattr(aks, "APIKEYS_FILE", str(tmp_path / "apikeys.json"))
    monkeypatch.setattr(apikey_service, "_keys", None)
    monkeypatch.setattr(apikey_service, "_keys_mtime", None)


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    """Runtime aislado y sin persistir en disco (no toca data/runtime.json)."""
    original_runtime = settings.runtime
    monkeypatch.setattr(settings, "runtime", config_service.RuntimeSettings())
    monkeypatch.setattr(config_service, "save_runtime_config", lambda: None)
    yield
    settings.runtime = original_runtime


@pytest.fixture
def client(monkeypatch):
    from app import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    ctx = type("Ctx", (), {"queue": DummyQueue()})()
    create_webui_routes(app, ctx)
    return TestClient(app), ctx


def test_set_config_whisper_model_updates_runtime(client):
    """POST /webui/api/config con whisper_model actualiza el runtime."""
    tc, _ = client
    res = tc.post("/webui/api/config", json={"whisper_model": "whisper-small"})
    assert res.status_code == 200
    assert res.json()["whisper_model"] == "whisper-small"
    assert settings.runtime.whisper_model == "whisper-small"


def test_set_config_whisper_model_invalid_400(client):
    """whisper_model desconocido se rechaza (400) y no toca el runtime."""
    tc, _ = client
    res = tc.post("/webui/api/config", json={"whisper_model": "whisper-huge"})
    assert res.status_code == 400
    assert settings.runtime.whisper_model == "whisper-large-v3"


def test_set_config_whisper_model_unloads_when_loaded(client, monkeypatch):
    """Si Whisper está cargado, cambiar el modelo lo descarga (bajo model_lock)
    para que la próxima transcripción cargue el elegido."""
    tc, _ = client
    unloaded = []
    monkeypatch.setattr(whisper_service, "is_loaded", lambda: True)

    async def fake_unload():
        unloaded.append(True)

    monkeypatch.setattr(whisper_service, "unload", fake_unload)

    res = tc.post("/webui/api/config", json={"whisper_model": "whisper-medium"})
    assert res.status_code == 200
    assert settings.runtime.whisper_model == "whisper-medium"
    assert unloaded == [True]


def test_set_config_whisper_model_no_unload_when_not_loaded(client, monkeypatch):
    """Si Whisper no está cargado, cambiar el modelo no dispara descarga."""
    tc, _ = client
    unloaded = []
    monkeypatch.setattr(whisper_service, "is_loaded", lambda: False)

    async def fake_unload():
        unloaded.append(True)

    monkeypatch.setattr(whisper_service, "unload", fake_unload)

    res = tc.post("/webui/api/config", json={"whisper_model": "whisper-large-v3"})
    assert res.status_code == 200
    assert unloaded == []


def test_set_config_requires_key_when_enabled(client, monkeypatch):
    """Con 'Exigir clave API' activado, la ruta admin exige clave válida."""
    from services import apikey_service

    monkeypatch.setattr(settings.runtime, "api_keys_enabled", True)
    created = apikey_service.create_key("admin")
    tc, _ = client

    res = tc.post("/webui/api/config", json={"whisper_model": "whisper-medium"})
    assert res.status_code == 401

    res = tc.post(
        "/webui/api/config",
        json={"whisper_model": "whisper-medium"},
        headers={"X-API-Key": created["key"]},
    )
    assert res.status_code == 200
    assert settings.runtime.whisper_model == "whisper-medium"


def test_set_config_open_when_disabled_even_with_keys(client, monkeypatch):
    """Con 'Exigir clave API' desactivado, la ruta admin no pide clave
    aunque existan claves creadas."""
    from services import apikey_service

    monkeypatch.setattr(settings.runtime, "api_keys_enabled", False)
    apikey_service.create_key("admin")
    tc, _ = client

    res = tc.post("/webui/api/config", json={"whisper_model": "whisper-small"})
    assert res.status_code == 200
    assert settings.runtime.whisper_model == "whisper-small"
