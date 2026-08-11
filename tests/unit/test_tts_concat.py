#!/usr/bin/env python3
"""Tests de la concatenación de fragmentos en el pipeline TTS.

No se garantiza continuidad de voz (cada chunk es una generación
independiente, el modelo se reinicia entre ellos), pero sí que el pipeline
de multi-chunk no introduzca errores:

- sample_rate constante
- channels constante
- dtype constante
- sin muestras perdidas
- duración total correcta
- orden correcto
"""

import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from schemas.tts import TTSRequest
from services import config_service
from services.audio_service import AudioService
from services.model_manager import ModelInfo
from services.tts_service import TTSService

SR = 24000
TEXT = "Frase uno. Frase dos. Frase tres."

# Una onda distinta y reconocible por fragmento, para verificar el orden.
# Amplitudes dentro de [-1, 1]: el WAV (int16) satura fuera de ese rango.
CHUNK_WAVES = {
    "Frase uno.": np.full(1000, 0.2, dtype="float32"),
    "Frase dos.": np.full(1500, 0.5, dtype="float32"),
    "Frase tres.": np.full(1200, 0.8, dtype="float32"),
}
TOTAL_SAMPLES = sum(len(w) for w in CHUNK_WAVES.values())


class DummyQueue:
    @asynccontextmanager
    async def inference_lock(self):
        yield


class DummyVoices:
    clone_prompt = "voz cargada de prueba"


@pytest.fixture
def service(monkeypatch):
    """TTSService real con la generación GPU simulada por fragmento."""
    monkeypatch.setattr("services.gpu_management.prepare_for_tts", _noop)
    # Fragmentos pequeños para forzar 3 chunks con TEXT.
    monkeypatch.setattr(config_service.settings.runtime, "max_text_chars", 20)

    audio = AudioService(SimpleNamespace(paths=SimpleNamespace(audios_dir=".")), None)
    svc = TTSService(
        config=SimpleNamespace(),
        queue=DummyQueue(),
        model_manager=None,
        voice_manager=DummyVoices(),
        audio_service=audio,
        metrics=None,
    )

    async def fake_resolve_model(request):
        return ModelInfo("test-base", "base", None, None)

    generated = []

    async def fake_generate_one(request, text, info):
        generated.append(text)
        return [CHUNK_WAVES[text]], SR

    monkeypatch.setattr(svc, "_resolve_model", fake_resolve_model)
    monkeypatch.setattr(svc, "_generate_one", fake_generate_one)
    svc._generated = generated
    return svc


async def _noop(*args, **kwargs):
    pass


def test_concat_wavs_preserves_order_and_content():
    """_concat_wavs: concatenación exacta (orden, muestras, dtype, canales)."""
    parts = [np.array([1, 2, 3], dtype="float32"), np.array([4, 5], dtype="float32")]
    out = TTSService._concat_wavs(parts)
    assert out.dtype == np.float32
    assert out.tolist() == [1, 2, 3, 4, 5]
    assert len(out) == 5  # sin muestras perdidas


def test_concat_wavs_stereo_preserves_channels():
    left = np.zeros((10, 2), dtype="float32")
    right = np.ones((5, 2), dtype="float32")
    out = TTSService._concat_wavs([left, right])
    assert out.ndim == 2 and out.shape[1] == 2  # channels constante
    assert out.shape[0] == 15
    assert np.all(out[:10] == 0) and np.all(out[10:] == 1)


def test_multi_chunk_pipeline_concatenation(service):
    """synthesize() con 3 fragmentos: el audio final es la concatenación
    exacta de cada generación, sin errores del pipeline."""
    import asyncio
    result = asyncio.run(service.synthesize(TTSRequest(text=TEXT)))

    # Orden correcto: los 3 fragmentos se generaron en orden.
    assert service._generated == list(CHUNK_WAVES.keys())

    # sample_rate constante y el declarado por la generación.
    assert result.sample_rate == SR
    audio, sr = service._audio.load(result.audio)
    assert sr == SR

    # dtype y channels constantes.
    assert audio.dtype == np.float32
    assert audio.ndim == 1  # mono constante

    # Sin muestras perdidas: duración = suma de los fragmentos.
    assert len(audio) == TOTAL_SAMPLES
    assert pytest.approx(len(audio) / SR, abs=1e-3) == TOTAL_SAMPLES / SR

    # Orden correcto: cada fragmento aparece en su posición. El WAV se
    # codifica a int16: tolerancia para la cuantización (~3e-5).
    n1, n2 = len(CHUNK_WAVES["Frase uno."]), len(CHUNK_WAVES["Frase dos."])
    assert np.allclose(audio[:n1], 0.2, atol=5e-4)
    assert np.allclose(audio[n1:n1 + n2], 0.5, atol=5e-4)
    assert np.allclose(audio[n1 + n2:], 0.8, atol=5e-4)
