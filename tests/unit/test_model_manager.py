#!/usr/bin/env python3
"""Tests unitarios del ciclo de vida de ModelManager."""

import asyncio

import pytest

from config.settings import CONFIG
from services import model_manager as mm
from services.model_manager import ModelManager


class FakeModel:
    tts_model_type = "base"


class FakeQwen3TTS:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return FakeModel()


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    (models_dir / "model-a").mkdir(parents=True)
    (models_dir / "model-b").mkdir()
    monkeypatch.setitem(CONFIG, "local_models_dir", str(models_dir))
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    return models_dir


def test_no_active_model_initially():
    mgr = ModelManager()
    assert asyncio.run(mgr.get_active_model()) is None
    assert not mgr.is_loaded()
    assert asyncio.run(mgr.get_loaded_models()) == []


def test_load_and_active(models_dir):
    mgr = ModelManager()
    info = asyncio.run(mgr.load_model("model-a"))
    assert info.model_id == "model-a"
    assert info.model_type == "base"

    active = asyncio.run(mgr.get_active_model())
    assert active == info
    assert mgr.is_loaded()


def test_load_duplicated_avoids_reload(models_dir):
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))
    asyncio.run(mgr.load_model("model-a"))
    assert len(asyncio.run(mgr.get_loaded_models())) == 1
    assert asyncio.run(mgr.get_active_model()).model_id == "model-a"


def test_switch_model_releases_previous(models_dir):
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))
    info_b = asyncio.run(mgr.switch_model("model-b"))

    assert info_b.model_id == "model-b"
    assert asyncio.run(mgr.get_active_model()).model_id == "model-b"
    # Solo un modelo en memoria (VRAM limitada)
    assert len(asyncio.run(mgr.get_loaded_models())) == 1


def test_switch_to_loaded_keeps_memory(models_dir):
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))
    asyncio.run(mgr.switch_model("model-b"))
    asyncio.run(mgr.switch_model("model-a"))
    assert asyncio.run(mgr.get_active_model()).model_id == "model-a"


def test_unload(models_dir):
    mgr = ModelManager()
    asyncio.run(mgr.load_model("model-a"))
    asyncio.run(mgr.unload_model("model-a"))

    assert not mgr.is_loaded()
    assert asyncio.run(mgr.get_active_model()) is None
    assert asyncio.run(mgr.get_loaded_models()) == []


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
