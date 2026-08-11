#!/usr/bin/env python3
"""Tests de /voice/create: la validación de audio ocurre ANTES del
inference_lock y sin preparar modelo/GPU (regla arquitectónica)."""

import io
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.voices import create_voices_routes
from services.audio_service import AudioService


def _wav_bytes(sr=16000, seconds=0.5):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = 0.5 * np.sin(2 * np.pi * 440 * t).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, wav, sr, format="wav")
    return buffer.getvalue()


class DummyQueue:
    """Registra si se llegó a adquirir inference_lock."""

    def __init__(self):
        self.lock_entered = False

    @asynccontextmanager
    async def inference_lock(self):
        self.lock_entered = True
        yield


class DummyModels:
    def __init__(self):
        self.voice = "voice_abc123"

    async def get_active_model(self):
        return SimpleNamespace(model_id="Qwen3-TTS-Base", model_type="base")


class DummyVoices:
    async def create(self, **kwargs):
        return "voice_abc123"

    def get(self, voice_id):
        return {"id": voice_id, "name": "TestVoz"}

    def get_reference(self, voice_id):
        if voice_id != "voice_abc123":
            return None
        return ("/tmp/ref.wav", "/tmp/ref.txt")


@pytest.fixture
def env(monkeypatch):
    """Entorno completo: rutas + spies de prepare_for_tts / require_model_loaded."""
    monkeypatch.setattr("routes.voices.require_admin", lambda: None)
    calls = {"prepare_for_tts": 0, "require_model_loaded": 0}

    async def fake_prepare_for_tts(*args, **kwargs):
        calls["prepare_for_tts"] += 1

    async def fake_require_model_loaded(*args, **kwargs):
        calls["require_model_loaded"] += 1

    monkeypatch.setattr("routes.voices.prepare_for_tts", fake_prepare_for_tts)
    monkeypatch.setattr("routes.voices.require_model_loaded", fake_require_model_loaded)

    queue = DummyQueue()
    ctx = SimpleNamespace(
        queue=queue,
        models=DummyModels(),
        voices=DummyVoices(),
        audio=AudioService(SimpleNamespace(), None),
    )
    from app import register_exception_handlers
    app = FastAPI()
    register_exception_handlers(app)
    create_voices_routes(app, ctx)
    return TestClient(app), queue, calls


def _post_voice(client, wav_bytes, filename="voz.wav", name="TestVoz", text="Hola"):
    return client.post(
        "/voice/create",
        data={"voice_name": name, "text": text},
        files={"audio": (filename, wav_bytes, "audio/wav")},
    )


def test_create_voice_invalid_audio_400_without_lock(env):
    """Audio inválido: 400 ANTES de adquirir inference_lock y sin preparar GPU."""
    client, queue, calls = env
    resp = _post_voice(client, b"esto no es audio, ni wav ni nada parecido")
    assert resp.status_code == 400
    assert queue.lock_entered is False
    assert calls["prepare_for_tts"] == 0
    assert calls["require_model_loaded"] == 0


def test_create_voice_oversized_audio_rejected_before_gpu(env, monkeypatch):
    """Audio que excede max_voice_audio_bytes: 400 sin cargar modelo ni GPU."""
    from config.settings import settings
    # El límite editable (runtime, en MB) es el vigente: 1 MB
    monkeypatch.setattr(settings.runtime, "max_voice_audio_bytes_mb", 1)

    client, queue, calls = env
    # ~4 MB de WAV (70 s a 16 kHz mono, 16 bits) > 1 MB
    resp = _post_voice(client, _wav_bytes(seconds=70))
    assert resp.status_code == 400
    assert "excede" in resp.text
    assert queue.lock_entered is False
    assert calls["prepare_for_tts"] == 0
    assert calls["require_model_loaded"] == 0


def test_create_voice_valid_audio_reaches_critical_section(env):
    """Audio válido: entra en inference_lock y prepara modelo/GPU."""
    client, queue, calls = env
    resp = _post_voice(client, _wav_bytes())
    assert resp.status_code == 200
    assert resp.json()["voice"] == "voice_abc123"
    assert queue.lock_entered is True
    assert calls["prepare_for_tts"] == 1
    assert calls["require_model_loaded"] == 1


def test_voice_preview_audio(env):
    """GET /voices/{id}/audio devuelve el audio de referencia (preview)."""
    import shutil
    import struct
    import tempfile
    import wave

    client, queue, calls = env
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 0) * 1600)
    shutil.copy(tmp, "/tmp/ref.wav")
    try:
        resp = client.get("/voices/voice_abc123/audio")
        assert resp.status_code == 200
        assert len(resp.content) > 0
        assert resp.headers["content-type"] == "audio/wav"
        resp2 = client.get("/voices/otra/audio")
        assert resp2.status_code == 404
    finally:
        os.unlink("/tmp/ref.wav")
        os.unlink(tmp)
