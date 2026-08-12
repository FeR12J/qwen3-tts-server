#!/usr/bin/env python3
"""Tests de errores unificados: todas las respuestas de error usan

    {"error": {"code": ..., "message": ..., "request_id": ...}}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import register_exception_handlers
from services.errors import (
    APIError,
    AuthenticationError,
    GPUOutOfMemoryError,
    InvalidAudioError,
    InvalidVoiceError,
    ModelLoadingError,
    ModelNotLoadedError,
    QueueFullError,
)


def test_exception_codes_and_statuses():
    """Cada excepción interna tiene su code y status HTTP."""
    cases = [
        (ModelNotLoadedError(), "MODEL_NOT_LOADED", 409),
        (ModelLoadingError(), "MODEL_LOADING_ERROR", 503),
        (InvalidVoiceError(), "INVALID_VOICE", 400),
        (InvalidAudioError(), "INVALID_AUDIO", 400),
        (GPUOutOfMemoryError(), "GPU_OUT_OF_MEMORY", 503),
        (QueueFullError(4), "QUEUE_FULL", 429),
        (AuthenticationError(), "AUTHENTICATION_ERROR", 401),
        (APIError("SERVICE_UNAVAILABLE", "apagado", 503), "SERVICE_UNAVAILABLE", 503),
    ]
    for exc, code, status in cases:
        assert exc.code == code
        assert exc.status_code == status
        assert exc.message


def _app_with_error_route():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom():
        raise ModelNotLoadedError()

    @app.get("/http")
    def http_error():
        raise HTTPException(404, "No encontrado")

    @app.get("/introspect")
    def introspect(request):
        raise APIError("X", "mensaje", 400, request_id=None)

    return app


def test_api_error_shape_with_request_id_header():
    """El error unificado incluye el request_id de la cabecera x-request-id."""
    client = TestClient(_app_with_error_route())
    res = client.get("/boom", headers={"x-request-id": "abc123"})
    assert res.status_code == 409
    assert res.json() == {
        "error": {
            "code": "MODEL_NOT_LOADED",
            "message": "The requested TTS model is not loaded.",
            "request_id": "abc123",
        }
    }


def test_api_error_generates_request_id_when_missing():
    """Sin cabecera x-request-id se genera un request_id (uuid hex)."""
    client = TestClient(_app_with_error_route())
    res = client.get("/boom")
    body = res.json()["error"]
    assert body["code"] == "MODEL_NOT_LOADED"
    assert len(body["request_id"]) == 32


def test_http_exception_normalized():
    """Cualquier HTTPException se normaliza al formato unificado."""
    client = TestClient(_app_with_error_route())
    res = client.get("/http", headers={"x-request-id": "rid404"})
    assert res.status_code == 404
    assert res.json() == {
        "error": {
            "code": "HTTP_404",
            "message": "No encontrado",
            "request_id": "rid404",
        }
    }


def test_request_validation_error_normalized():
    """Errores 422 de FastAPI se normalizan con code VALIDATION_ERROR."""
    app = FastAPI()
    register_exception_handlers(app)
    from pydantic import BaseModel

    class Body(BaseModel):
        text: str

    @app.post("/validated")
    def validated(body: Body):
        return {"ok": True}

    client = TestClient(app)
    res = client.post("/validated", json={"text": 123}, headers={"x-request-id": "rid422"})
    assert res.status_code == 422
    body = res.json()["error"]
    assert body["code"] == "VALIDATION_ERROR"
    assert "text" in body["message"]
    assert body["request_id"] == "rid422"