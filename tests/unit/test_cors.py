#!/usr/bin/env python3
"""Tests de CORS: sin "*" por defecto, configurable en tiempo de ejecución
desde el panel (DynamicCORSMiddleware) y puerto editable."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi.testclient import TestClient

from config.settings import CorsSettings, settings
from services import config_service
from security.validation import validate_config_update

ALLOWED = "http://localhost:3000"
OTHER = "http://evil.example.com"


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    """Runtime aislado por test (sin archivo persistido)."""
    monkeypatch.setattr(
        config_service,
        "settings",
        config_service.settings,  # no-op: model_copy en cada test
    )
    defaults = config_service.RuntimeSettings()
    monkeypatch.setattr(config_service.settings, "runtime", defaults)
    yield


def test_cors_defaults_never_wildcard():
    """La política CORS por defecto no usa '*'."""
    cors = CorsSettings()
    assert cors.enabled is False
    assert "*" not in cors.origins
    assert cors.origins == ["http://localhost:3000"]
    assert cors.allow_wildcard is False


def _client():
    """App mínima con el DynamicCORSMiddleware real (sin lifespan)."""
    from fastapi import FastAPI
    from app import DynamicCORSMiddleware

    app = FastAPI()
    app.add_middleware(
        DynamicCORSMiddleware,
        allow_origins=settings.cors.origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return TestClient(app)


def test_no_cors_headers_when_disabled(clean_runtime):
    """CORS desactivado: sin cabeceras access-control-allow-origin."""
    client = _client()
    res = client.get("/health", headers={"Origin": ALLOWED})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in res.headers


def test_cors_headers_for_allowed_origin_when_enabled(clean_runtime, monkeypatch):
    """Activado con el origen en la lista: la cabecera CORS aparece y el
    origen no permitido no la recibe (sin reiniciar)."""
    monkeypatch.setattr(config_service.settings.runtime, "cors_enabled", True)
    monkeypatch.setattr(config_service.settings.runtime, "cors_origins", [ALLOWED])

    client = _client()
    res = client.get("/health", headers={"Origin": ALLOWED})
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == ALLOWED

    res2 = client.get("/health", headers={"Origin": OTHER})
    assert "access-control-allow-origin" not in res2.headers


def test_cors_preflight(clean_runtime, monkeypatch):
    """Preflight OPTIONS con origen permitido devuelve 200 y las cabeceras."""
    monkeypatch.setattr(config_service.settings.runtime, "cors_enabled", True)
    monkeypatch.setattr(config_service.settings.runtime, "cors_origins", [ALLOWED])

    client = _client()
    res = client.options(
        "/health",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == ALLOWED


def test_port_default_seeded_from_server():
    """El puerto editable (runtime) se siembra desde server.port."""
    assert settings.runtime.port == settings.server.port


def test_port_update_rejected_out_of_range(clean_runtime):
    """validate_config_update rechaza puertos fuera de 1-65535."""
    with pytest.raises(Exception) as e:
        validate_config_update({"port": 0}, config_service)
    assert e.value.status_code == 400
    with pytest.raises(Exception) as e:
        validate_config_update({"port": 70000}, config_service)
    assert e.value.status_code == 400


def test_port_update_accepted(clean_runtime):
    validate_config_update({"port": 9090}, config_service)
    rc = config_service.update_runtime_config({"port": 9090})
    assert rc["port"] == 9090


def test_cors_origins_wildcard_rejected(clean_runtime):
    """cors_origins no admite '*'."""
    with pytest.raises(Exception) as e:
        validate_config_update({"cors_origins": ["*"]}, config_service)
    assert e.value.status_code == 400
    with pytest.raises(Exception) as e:
        validate_config_update({"cors_origins": ["http://a.com", "*"]}, config_service)
    assert e.value.status_code == 400


def test_cors_runtime_update_persists(clean_runtime):
    """El ajuste de CORS desde el panel se guarda en runtime."""
    validate_config_update(
        {"cors_enabled": True, "cors_origins": ["http://localhost:3000"]},
        config_service,
    )
    rc = config_service.update_runtime_config(
        {"cors_enabled": True, "cors_origins": ["http://localhost:3000"]},
    )
    assert rc["cors_enabled"] is True
    assert rc["cors_origins"] == ["http://localhost:3000"]


def test_cors_wildcard_off_by_default(clean_runtime):
    """Wildcard desactivado por defecto: orígenes no listados no reciben CORS."""
    assert settings.runtime.cors_allow_wildcard is False
    rc = config_service.load_runtime_config()
    assert rc is None or settings.runtime.cors_allow_wildcard is False


def test_cors_wildcard_allows_any_origin(clean_runtime, monkeypatch):
    """cors_allow_wildcard=true: cualquier origen recibe la cabecera '*'."""
    monkeypatch.setattr(config_service.settings.runtime, "cors_enabled", True)
    monkeypatch.setattr(config_service.settings.runtime, "cors_allow_wildcard", True)

    client = _client()
    res = client.get("/health", headers={"Origin": OTHER})
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "*"


def test_cors_wildcard_preflight(clean_runtime, monkeypatch):
    """Preflight con wildcard activado devuelve '*'."""
    monkeypatch.setattr(config_service.settings.runtime, "cors_enabled", True)
    monkeypatch.setattr(config_service.settings.runtime, "cors_allow_wildcard", True)

    client = _client()
    res = client.options(
        "/health",
        headers={
            "Origin": OTHER,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "*"


def test_cors_wildcard_validation(clean_runtime):
    """cors_allow_wildcard acepta true/false y rechaza otros tipos."""
    validate_config_update({"cors_allow_wildcard": True}, config_service)
    validate_config_update({"cors_allow_wildcard": False}, config_service)
    with pytest.raises(Exception) as e:
        validate_config_update({"cors_allow_wildcard": "yes"}, config_service)
    assert e.value.status_code == 400


def test_cors_wildcard_runtime_update_persists(clean_runtime):
    """El toggle '*' desde el panel se guarda en runtime."""
    validate_config_update(
        {"cors_enabled": True, "cors_allow_wildcard": True}, config_service
    )
    rc = config_service.update_runtime_config(
        {"cors_enabled": True, "cors_allow_wildcard": True}
    )
    assert rc["cors_enabled"] is True
    assert rc["cors_allow_wildcard"] is True