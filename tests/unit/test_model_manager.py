#!/usr/bin/env python3
"""Tests unitarios del ciclo de vida de ModelManager."""

import asyncio
import time

import pytest

from config.settings import CONFIG
from services import model_manager as mm
from services.model_manager import ModelManager, ModelState


class FakeModel:
    tts_model_type = "base"

    def generate_voice_clone(self, **kwargs):
        time.sleep(0.15)
        return [], 24000

    def create_voice_clone_prompt(self, **kwargs):
        time.sleep(0.05)
        return "fake-prompt"


class FakeQwen3TTS:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return FakeModel()


class SlowQwen3TTS:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        time.sleep(0.2)
        return FakeModel()


class BrokenQwen3TTS:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("OOM simulado")


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    (models_dir / "model-a").mkdir(parents=True)
    (models_dir / "model-b").mkdir()
    monkeypatch.setitem(CONFIG, "local_models_dir", str(models_dir))
    return models_dir


def _status(mgr):
    return asyncio.run(mgr.get_model_status())


def test_no_active_model_initially():
    mgr = ModelManager()
    assert asyncio.run(mgr.get_active_model()) is None
    assert not mgr.is_loaded()
    assert asyncio.run(mgr.get_loaded_models()) == []

    status = _status(mgr)
    assert status["state"] == ModelState.UNLOADED.value
    assert status["model_id"] is None


def test_load_and_active(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    mgr = ModelManager()
    info = asyncio.run(mgr.load_model("model-a"))
    assert info.model_id == "model-a"
    assert info.model_type == "base"

    assert asyncio.run(mgr.get_active_model()) == info
    assert mgr.is_loaded()


def test_status_ready_after_load(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))

    status = _status(mgr)
    assert status["model_id"] == "model-a"
    assert status["type"] == "base"
    assert status["state"] == ModelState.READY.value
    assert status["loaded_at"] is not None
    assert status["device"] is not None
    assert status["dtype"] is not None
    assert status["error"] is None


def test_loading_state_during_load(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", SlowQwen3TTS)
    mgr = ModelManager()

    async def scenario():
        task = asyncio.create_task(mgr.load_model("model-a"))
        await asyncio.sleep(0.05)
        status = await mgr.get_model_status()
        assert status["state"] == ModelState.LOADING.value
        assert status["model_id"] == "model-a"
        assert not mgr.is_loaded()
        await task
        status = await mgr.get_model_status()
        assert status["state"] == ModelState.READY.value

    asyncio.run(scenario())


def test_error_state_when_load_fails(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", BrokenQwen3TTS)
    mgr = ModelManager()

    with pytest.raises(RuntimeError):
        asyncio.run(mgr.load_model("model-a"))

    status = _status(mgr)
    assert status["state"] == ModelState.ERROR.value
    assert status["model_id"] == "model-a"
    assert "OOM" in status["error"]
    assert not mgr.is_loaded()
    assert asyncio.run(mgr.get_active_model()) is None


def test_error_never_reports_ready(models_dir, monkeypatch):
    """Un modelo roto nunca pasa a READY."""
    monkeypatch.setattr(mm, "Qwen3TTSModel", BrokenQwen3TTS)
    mgr = ModelManager()

    with pytest.raises(RuntimeError):
        asyncio.run(mgr.load_model("model-a"))

    assert _status(mgr)["state"] != ModelState.READY.value


def test_generating_state_during_inference(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))

    async def scenario():
        task = asyncio.create_task(
            mgr.generate_voice_clone(text="hola", language="Spanish", voice_clone_prompt="p")
        )
        await asyncio.sleep(0.05)
        status = await mgr.get_model_status()
        assert status["state"] == ModelState.GENERATING.value
        await task
        status = await mgr.get_model_status()
        assert status["state"] == ModelState.READY.value

    asyncio.run(scenario())


def test_load_duplicated_avoids_reload(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))
    asyncio.run(mgr.load_model("model-a"))
    assert len(asyncio.run(mgr.get_loaded_models())) == 1
    assert asyncio.run(mgr.get_active_model()).model_id == "model-a"


def test_switch_model_releases_previous(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))
    info_b = asyncio.run(mgr.switch_model("model-b"))

    assert info_b.model_id == "model-b"
    assert asyncio.run(mgr.get_active_model()).model_id == "model-b"
    # Solo un modelo en memoria (VRAM limitada)
    assert len(asyncio.run(mgr.get_loaded_models())) == 1


def test_switch_to_loaded_keeps_memory(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))
    asyncio.run(mgr.switch_model("model-b"))
    asyncio.run(mgr.switch_model("model-a"))
    assert asyncio.run(mgr.get_active_model()).model_id == "model-a"


def test_unload(models_dir, monkeypatch):
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))
    asyncio.run(mgr.unload_model("model-a"))

    assert not mgr.is_loaded()
    assert asyncio.run(mgr.get_active_model()) is None
    assert asyncio.run(mgr.get_loaded_models()) == []
    status = _status(mgr)
    assert status["state"] == ModelState.UNLOADED.value


def test_unload_nonexistent_is_noop():
    mgr = ModelManager()
    asyncio.run(mgr.unload_model("ghost"))
    assert asyncio.run(mgr.get_active_model()) is None


def test_load_missing_model_raises(models_dir):
    mgr = ModelManager()
    with pytest.raises(FileNotFoundError):
        asyncio.run(mgr.load_model("no-existe"))


def test_load_empty_id_raises():
    mgr = ModelManager()
    with pytest.raises(ValueError):
        asyncio.run(mgr.load_model("  "))


def test_model_without_generation_method_is_error(models_dir, monkeypatch):
    class NoMethodModel:
        tts_model_type = "voice_design"

    class NoMethodTTS:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return NoMethodModel()

    monkeypatch.setattr(mm, "Qwen3TTSModel", NoMethodTTS)
    mgr = ModelManager()

    with pytest.raises(RuntimeError):
        asyncio.run(mgr.load_model("model-a"))

    status = _status(mgr)
    assert status["state"] == ModelState.ERROR.value
    assert "generate_voice_design" in status["error"]
