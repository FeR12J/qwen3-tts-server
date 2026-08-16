#!/usr/bin/env python3
"""Tests de la selección del modelo Whisper (editable desde el panel):
el runtime tiene prioridad sobre el grupo estático y la carga usa el nombre
configurado; si cambia, el modelo anterior se descarga y se recarga el nuevo."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from config.settings import settings
from services import whisper_service as ws


class _FakeModel:
    device = "cpu"
    dtype = "float32"


@pytest.fixture
def svc(monkeypatch, tmp_path):
    """WhisperService aislado: models_dir temporal (monkeypatch restaura)."""
    monkeypatch.setattr(settings.paths, "models_dir", str(tmp_path))
    # Crear los directorios de los modelos "descargados" (solo se comprueba
    # la existencia de la ruta antes de cargar).
    for name in ("whisper-small", "whisper-medium", "whisper-large-v3"):
        (tmp_path / name).mkdir()
    return ws.WhisperService(audio_service=None)


@pytest.fixture
def fake_transformers(monkeypatch):
    """Sustituir transformers: registrar las rutas con las que se 'carga'."""
    import transformers

    loaded_paths = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, *a, **k):
            loaded_paths.append(("processor", path))
            return cls()

    class FakeWhisper:
        @classmethod
        def from_pretrained(cls, path, *a, **k):
            loaded_paths.append(("model", path))
            return _FakeModel()

    monkeypatch.setattr(settings.runtime, "device", "cpu")
    monkeypatch.setattr(settings.runtime, "dtype", "float32")
    monkeypatch.setattr(transformers, "WhisperProcessor", FakeProcessor)
    monkeypatch.setattr(transformers, "WhisperForConditionalGeneration", FakeWhisper)
    return loaded_paths


def test_configured_model_runtime_beats_static(svc, monkeypatch):
    """settings.runtime.whisper_model gana al default estático."""
    monkeypatch.setattr(settings.whisper, "whisper_model", "whisper-small")
    monkeypatch.setattr(settings.runtime, "whisper_model", "whisper-medium")
    assert svc._configured_model_name() == "whisper-medium"


def test_configured_model_falls_back_to_static(svc, monkeypatch):
    """Si el runtime no tuviera el campo, se cae al grupo estático."""
    monkeypatch.setattr(settings.whisper, "whisper_model", "whisper-small")
    # Runtime sin el campo (p.ej. runtime.json antiguo cargado a medias):
    # getattr con default None -> fallback al default estático.
    monkeypatch.setattr(settings, "runtime", SimpleNamespace())
    assert svc._configured_model_name() == "whisper-small"


def test_status_reports_configured_model(svc, monkeypatch):
    """/transcribe/status (status) muestra el modelo configurado (runtime)."""
    monkeypatch.setattr(settings.runtime, "whisper_model", "whisper-small")
    st = svc.status()
    assert st["model"] == "whisper-small"
    assert st["model_loaded"] is False


def test_load_model_uses_runtime_name(svc, fake_transformers, monkeypatch):
    """_ensure_loaded carga desde models_dir/<runtime.whisper_model>."""
    monkeypatch.setattr(settings.runtime, "whisper_model", "whisper-medium")
    svc._ensure_loaded()
    assert svc.is_loaded()
    assert svc._model_name == "whisper-medium"
    paths = [p for kind, p in fake_transformers if kind == "model"]
    assert len(paths) == 1
    assert os.path.basename(paths[0]) == "whisper-medium"


def test_ensure_loaded_reloads_when_model_changes(svc, fake_transformers, monkeypatch):
    """Si el configurado cambia con el modelo cargado, se descarga el
    anterior y se carga el nuevo (aplicación lazy del cambio del panel)."""
    monkeypatch.setattr(settings.runtime, "whisper_model", "whisper-small")
    svc._ensure_loaded()
    assert svc._model_name == "whisper-small"

    monkeypatch.setattr(settings.runtime, "whisper_model", "whisper-large-v3")
    svc._ensure_loaded()
    assert svc._model_name == "whisper-large-v3"
    model_paths = [p for kind, p in fake_transformers if kind == "model"]
    assert [os.path.basename(p) for p in model_paths] == [
        "whisper-small", "whisper-large-v3",
    ]


def test_ensure_loaded_no_reload_same_model(svc, fake_transformers, monkeypatch):
    """Con el mismo modelo configurado no se recarga (ni se descarga)."""
    monkeypatch.setattr(settings.runtime, "whisper_model", "whisper-small")
    svc._ensure_loaded()
    svc._ensure_loaded()
    model_paths = [p for kind, p in fake_transformers if kind == "model"]
    assert len(model_paths) == 1


def test_load_model_missing_dir_raises(svc, fake_transformers, monkeypatch):
    """Modelo configurado sin descargar: FileNotFoundError (404 en la ruta)."""
    monkeypatch.setattr(settings.runtime, "whisper_model", "whisper-medium")
    import shutil
    shutil.rmtree(os.path.join(settings.paths.models_dir, "whisper-medium"))
    with pytest.raises(FileNotFoundError):
        svc._ensure_loaded()
    assert not svc.is_loaded()


@pytest.mark.asyncio
async def test_load_public_method(svc, fake_transformers, monkeypatch):
    """load() (público, async) carga el modelo configurado sin bloquear."""
    monkeypatch.setattr(settings.runtime, "whisper_model", "whisper-medium")
    await svc.load()
    assert svc.is_loaded()
    assert svc._model_name == "whisper-medium"
