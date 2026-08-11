#!/usr/bin/env python3
"""Tests de chunking + streaming HTTP (/tts/stream).

Entrada: texto > chunk_size. Se comprueba que:
- se generan N chunks (N generaciones independientes),
- se reciben N chunks por HTTP (eventos http.response.body del stack ASGI,
  tal como los enviaría uvicorn; httpx/TestClient los fusionarían),
- los N audios recibidos se pueden concatenar sin corrupción
  (orden, longitud, sample rate constantes).
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest
from fastapi import FastAPI

from routes.tts import create_tts_routes
from services import config_service
from services.audio_service import AudioService
from services.model_manager import ModelInfo
from services.tts_service import TTSService

SR = 24000
TEXT = "Frase uno. Frase dos. Frase tres."

# Una onda distinta y reconocible por fragmento, para verificar el orden
# (amplitudes dentro de [-1, 1]: PCM/WAV int16 satura fuera de ese rango).
CHUNK_WAVES = {
    "Frase uno.": np.full(1000, 0.2, dtype="float32"),
    "Frase dos.": np.full(1500, 0.5, dtype="float32"),
    "Frase tres.": np.full(1200, 0.8, dtype="float32"),
}
EXPECTED_CHUNKS = list(CHUNK_WAVES.keys())
TOTAL_SAMPLES = sum(len(w) for w in CHUNK_WAVES.values())
TOTAL_BYTES = TOTAL_SAMPLES * 2  # PCM 16-bit LE
WAV_HEADER_BYTES = 44


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


async def _asgi_request(app, path, json_body):
    """POST directo sobre el stack ASGI (lo que uvicorn enviaría al cliente).

    Devuelve (status, headers, body_parts): cada yield del StreamingResponse
    es un evento http.response.body separado (un "chunk HTTP").
    """
    body = json.dumps(json_body).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    status, headers, body_parts = None, {}, []

    async def send(msg):
        nonlocal status
        if msg["type"] == "http.response.start":
            status = msg["status"]
            headers.update(dict(msg["headers"]))
        elif msg["type"] == "http.response.body":
            body_parts.append(msg.get("body", b""))

    await app(scope, receive, send)
    return status, headers, body_parts


@pytest.fixture
def env(monkeypatch):
    """Rutas HTTP reales + TTSService real con la generación GPU simulada."""
    monkeypatch.setattr("routes.tts.require_api_key", lambda: None)
    monkeypatch.setattr("services.gpu_management.prepare_for_tts", _noop)
    # Fragmentos pequeños para forzar N=3 chunks con TEXT.
    monkeypatch.setattr(config_service.settings.runtime, "max_text_chars", 20)

    audio = AudioService(SimpleNamespace(paths=SimpleNamespace(audios_dir=".")), None)
    svc = TTSService(
        config=SimpleNamespace(),
        queue=DummyQueue(),
        model_manager=DummyModels(),
        voice_manager=DummyVoices(),
        audio_service=audio,
        metrics=None,
    )

    generated = []

    async def fake_generate_one(request, text, info):
        generated.append(text)
        return [CHUNK_WAVES[text]], SR

    monkeypatch.setattr(svc, "_generate_one", fake_generate_one)
    svc._generated = generated

    from app import register_exception_handlers
    app = FastAPI()
    register_exception_handlers(app)
    create_tts_routes(app, SimpleNamespace(tts=svc, audio=audio))
    return app, svc


def _decode_pcm(pcm: bytes) -> np.ndarray:
    """PCM 16-bit LE -> float32 en [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype("float32") / 32767.0


def _assert_segments(audio: np.ndarray):
    """Orden y contenido de los 3 fragmentos concatenados (sin corrupción)."""
    assert len(audio) == TOTAL_SAMPLES
    n1, n2 = len(CHUNK_WAVES["Frase uno."]), len(CHUNK_WAVES["Frase dos."])
    assert np.allclose(audio[:n1], 0.2, atol=5e-4)
    assert np.allclose(audio[n1:n1 + n2], 0.5, atol=5e-4)
    assert np.allclose(audio[n1 + n2:], 0.8, atol=5e-4)


def test_stream_pcm_n_chunks_generated_and_received(env):
    """Texto > chunk_size: N=3 chunks generados y N=3 chunks HTTP (PCM)."""
    app, svc = env
    status, headers, parts = asyncio.run(_asgi_request(
        app, "/tts/stream",
        {"text": TEXT, "output_format": "pcm"},
    ))
    assert status == 200
    assert headers[b"content-type"] == b"audio/L16"
    assert headers[b"x-audio-rate"] == str(SR).encode()

    # N chunks generados, en orden.
    assert svc._generated == EXPECTED_CHUNKS
    # N chunks recibidos por HTTP (uno por generación; el evento final de
    # cierre del stack ASGI es una pieza vacía y no cuenta).
    chunks = [p for p in parts if p]
    assert len(chunks) == len(EXPECTED_CHUNKS) == 3
    # Sin bytes perdidos: 2 bytes/muestra por fragmento.
    assert sum(len(p) for p in chunks) == TOTAL_BYTES

    # Concatenación sin corrupción: orden y contenido correctos.
    _assert_segments(_decode_pcm(b"".join(chunks)))


def test_stream_wav_header_plus_chunks_concatenable(env):
    """WAV streaming: cabecera + N chunks PCM concatenables sin corrupción."""
    app, svc = env
    status, headers, parts = asyncio.run(_asgi_request(
        app, "/tts/stream",
        {"text": TEXT, "output_format": "wav"},
    ))
    assert status == 200
    assert headers[b"content-type"] == b"audio/wav"

    # Primera pieza: cabecera WAV + PCM del primer fragmento; el resto PCM puro.
    chunks = [p for p in parts if p]
    body = b"".join(chunks)
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    assert len(chunks[0]) == WAV_HEADER_BYTES + len(CHUNK_WAVES["Frase uno."]) * 2
    assert len(chunks) == len(EXPECTED_CHUNKS) == 3
    assert len(body) == WAV_HEADER_BYTES + TOTAL_BYTES

    # El flujo completo es decodificable como WAV (integridad del archivo).
    audio, sr = svc._audio.load(body)
    assert sr == SR and audio.dtype == np.float32
    _assert_segments(audio)
