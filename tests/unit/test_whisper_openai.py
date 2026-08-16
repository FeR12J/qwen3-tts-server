#!/usr/bin/env python3
"""Tests de las rutas OpenAI-compatibles de Whisper:
POST /tts/audio/transcriptions y POST /transcribe/load."""

import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings
from routes.whisper import create_whisper_routes
from services import whisper_service

WAV_BYTES = b"RIFF-test-audio-content"


class DummyQueue:
    inference_active = 0

    @asynccontextmanager
    async def model_lock(self):
        yield

    @asynccontextmanager
    async def inference_lock(self):
        yield


class FakeAudio:
    def validate(self, data, **kwargs):
        return None


class FakeModels:
    async def get_active_model(self):
        return None

    async def unload_model(self, model_id):
        return None


class FakeVoices:
    def unload_voice(self):
        return None


@pytest.fixture(autouse=True)
def isolated_api_keys(tmp_path, monkeypatch):
    """Aislar el store de claves API (bootstrap: sin claves)."""
    import storage.api_key_storage as aks
    from services import apikey_service

    monkeypatch.setattr(aks, "APIKEYS_FILE", str(tmp_path / "apikeys.json"))
    monkeypatch.setattr(apikey_service, "_keys", None)
    monkeypatch.setattr(apikey_service, "_keys_mtime", None)


@pytest.fixture
def client(monkeypatch):
    """App con las rutas Whisper y transcribe() falso (sin GPU)."""
    from app import register_exception_handlers

    calls = []
    calls_model = []

    async def fake_transcribe(audio, language=None, task="transcribe",
                              timestamps=None):
        calls.append({"language": language, "timestamps": timestamps})
        return {
            "text": "Hola mundo",
            "language": "es",
            "duration_seconds": 2.5,
            "model": "whisper-small",
            "device": "cpu",
            "timestamps": timestamps,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hola"},
                {"start": 1.0, "end": 2.5, "text": "mundo"},
            ],
            "words": [{"word": "Hola", "start": 0.0, "end": 0.5}],
        }

    async def fake_load():
        calls_model.append(True)

    monkeypatch.setattr(whisper_service, "transcribe", fake_transcribe)
    monkeypatch.setattr(whisper_service, "load", fake_load)
    monkeypatch.setattr(whisper_service, "is_loaded", lambda: False)

    app = FastAPI()
    register_exception_handlers(app)
    ctx = type("Ctx", (), {
        "queue": DummyQueue(),
        "audio": FakeAudio(),
        "models": FakeModels(),
        "voices": FakeVoices(),
    })()
    create_whisper_routes(app, ctx)
    tc = TestClient(app)
    return tc, calls


def _files():
    return {"file": ("test.wav", WAV_BYTES, "audio/wav")}


def test_transcriptions_json_default(client):
    """response_format por defecto: {"text": ...} y sin marcas de tiempo."""
    tc, calls = client
    res = tc.post("/tts/audio/transcriptions", files=_files())
    assert res.status_code == 200
    assert res.json() == {"text": "Hola mundo"}
    assert calls == [{"language": None, "timestamps": "off"}]


def test_transcriptions_text_plain(client):
    tc, calls = client
    res = tc.post("/tts/audio/transcriptions", files=_files(),
                  data={"response_format": "text"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert res.text == "Hola mundo"
    assert calls == [{"language": None, "timestamps": "off"}]


def test_transcriptions_verbose_json(client):
    tc, calls = client
    res = tc.post("/tts/audio/transcriptions", files=_files(),
                  data={"response_format": "verbose_json"})
    assert res.status_code == 200
    body = res.json()
    assert body["text"] == "Hola mundo"
    assert body["language"] == "es"
    assert body["duration"] == 2.5
    assert len(body["segments"]) == 2
    assert calls == [{"language": None, "timestamps": "segment"}]


def test_transcriptions_srt(client):
    tc, calls = client
    res = tc.post("/tts/audio/transcriptions", files=_files(),
                  data={"response_format": "srt"})
    assert res.status_code == 200
    assert "00:00:00,000 --> 00:00:01,000" in res.text
    assert "Hola" in res.text
    assert calls == [{"language": None, "timestamps": "segment"}]


def test_transcriptions_vtt(client):
    tc, calls = client
    res = tc.post("/tts/audio/transcriptions", files=_files(),
                  data={"response_format": "vtt"})
    assert res.status_code == 200
    assert res.text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000" in res.text
    assert calls == [{"language": None, "timestamps": "segment"}]


def test_transcriptions_word_granularity(client):
    tc, calls = client
    res = tc.post(
        "/tts/audio/transcriptions",
        files=_files(),
        data={
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        },
    )
    assert res.status_code == 200
    assert "words" in res.json()
    assert calls == [{"language": None, "timestamps": "word"}]


def test_transcriptions_language_passthrough(client):
    tc, calls = client
    res = tc.post("/tts/audio/transcriptions", files=_files(),
                  data={"language": "es"})
    assert res.status_code == 200
    assert calls == [{"language": "es", "timestamps": "off"}]


def test_transcriptions_model_ignored(client):
    """El campo model del cliente no cambia el modelo usado."""
    tc, calls = client
    res = tc.post("/tts/audio/transcriptions", files=_files(),
                  data={"model": "whisper-medium"})
    assert res.status_code == 200
    assert res.json() == {"text": "Hola mundo"}


def test_transcriptions_requires_file(client):
    tc, _ = client
    res = tc.post("/tts/audio/transcriptions")
    assert res.status_code == 400


def test_transcriptions_invalid_response_format(client):
    tc, _ = client
    res = tc.post("/tts/audio/transcriptions", files=_files(),
                  data={"response_format": "xml"})
    assert res.status_code == 400


def test_transcriptions_invalid_granularity(client):
    tc, _ = client
    res = tc.post(
        "/tts/audio/transcriptions",
        files=_files(),
        data={"response_format": "verbose_json",
              "timestamp_granularities": ["line"]},
    )
    assert res.status_code == 400


def test_transcriptions_requires_key_when_enabled(client, monkeypatch):
    """Con claves exigidas, la ruta devuelve 401 sin clave válida."""
    monkeypatch.setattr(settings.runtime, "api_keys_enabled", True)
    tc, _ = client
    res = tc.post("/tts/audio/transcriptions", files=_files())
    assert res.status_code == 401


def test_transcribe_load_ok(client):
    """POST /transcribe/load carga el modelo y devuelve estado."""
    tc, _ = client

    async def fake_load_async():
        return None

    def fake_status():
        return {"model_loaded": True, "model": "whisper-large-v3",
                "device": "cpu", "timestamps": "off"}

    original_load = whisper_service.load
    original_status = whisper_service.status
    whisper_service.load = fake_load_async
    whisper_service.status = fake_status
    try:
        res = tc.post("/transcribe/load")
    finally:
        whisper_service.load = original_load
        whisper_service.status = original_status
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True


def test_transcribe_load_404_when_not_downloaded(client, monkeypatch):
    """Si el modelo no está descargado, /transcribe/load devuelve 404."""
    tc, _ = client

    async def failing_load():
        raise FileNotFoundError("Modelo Whisper no encontrado")

    monkeypatch.setattr(whisper_service, "load", failing_load)
    res = tc.post("/transcribe/load")
    assert res.status_code == 404