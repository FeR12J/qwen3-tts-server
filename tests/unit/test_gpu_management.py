#!/usr/bin/env python3
"""Tests de la gestión de VRAM compartida TTS <-> Whisper."""

import asyncio

import pytest

from config import defaults
from config.settings import CONFIG
from services import config_service
from services import gpu_management as gm
from services import model_manager as mm
from services import whisper_service as ws
from services.model_manager import ModelManager


class FakeModel:
    tts_model_type = "base"

    def generate_voice_clone(self, **kwargs):
        return [], 24000


class FakeQwen3TTS:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return FakeModel()


class FakeVoices:
    def __init__(self):
        self.unloads = 0

    def unload_voice(self):
        self.unloads += 1


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    (models_dir / "model-a").mkdir(parents=True)
    monkeypatch.setitem(CONFIG, "local_models_dir", str(models_dir))
    return models_dir


@pytest.fixture
def whisper_unloaded(monkeypatch):
    monkeypatch.setattr(ws, "_model", None)
    monkeypatch.setattr(ws, "_processor", None)


def test_whisper_management_defaults_enabled():
    assert defaults.DEFAULTS["unload_tts_for_whisper"] is True
    assert defaults.DEFAULTS["unload_whisper_for_tts"] is True


def test_unload_if_loaded_noop_when_not_loaded(whisper_unloaded):
    assert ws.unload_if_loaded() is False
    assert ws.is_loaded() is False


def test_unload_if_loaded_unloads(whisper_unloaded, monkeypatch):
    monkeypatch.setattr(ws, "_model", object())
    monkeypatch.setattr(ws, "_processor", object())
    assert ws.is_loaded() is True
    assert ws.unload_if_loaded() is True
    assert ws.is_loaded() is False


def test_tts_unloaded_before_whisper_and_restored_after(
    models_dir, monkeypatch, whisper_unloaded
):
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    monkeypatch.setitem(config_service._runtime, "unload_tts_for_whisper", True)
    monkeypatch.setitem(config_service._runtime, "unload_whisper_for_tts", True)
    voices = FakeVoices()
    mgr = ModelManager()

    asyncio.run(mgr.load_model("model-a"))
    assert mgr.is_loaded()

    # Petición Whisper: se descarga el modelo TTS para dejar sitio
    asyncio.run(gm.prepare_for_whisper(mgr, voices, ws))
    assert not mgr.is_loaded()
    assert mgr.last_active_id() == "model-a"
    assert voices.unloads == 1

    # Petición TTS: Whisper no está cargado y se restaura el modelo TTS
    asyncio.run(gm.prepare_for_tts(mgr, voices, ws))
    assert mgr.is_loaded()


def test_whisper_unloaded_before_tts(models_dir, monkeypatch, whisper_unloaded):
    monkeypatch.setitem(config_service._runtime, "unload_whisper_for_tts", True)
    monkeypatch.setattr(ws, "_model", object())
    monkeypatch.setattr(ws, "_processor", object())
    mgr = ModelManager()

    assert ws.is_loaded() is True
    asyncio.run(gm.prepare_for_tts(mgr, FakeVoices(), ws))
    assert ws.is_loaded() is False


def test_both_models_kept_when_flags_disabled(
    models_dir, monkeypatch, whisper_unloaded
):
    # GPUs grandes: ambos modelos pueden permanecer cargados
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    monkeypatch.setitem(config_service._runtime, "unload_tts_for_whisper", False)
    monkeypatch.setitem(config_service._runtime, "unload_whisper_for_tts", False)
    monkeypatch.setattr(ws, "_model", object())
    monkeypatch.setattr(ws, "_processor", object())
    voices = FakeVoices()
    mgr = ModelManager()

    asyncio.run(mgr.load_model("model-a"))
    assert mgr.is_loaded()

    # Petición Whisper: TTS se mantiene
    asyncio.run(gm.prepare_for_whisper(mgr, voices, ws))
    assert mgr.is_loaded()
    assert voices.unloads == 0

    # Petición TTS: Whisper se mantiene
    asyncio.run(gm.prepare_for_tts(mgr, voices, ws))
    assert ws.is_loaded() is True
