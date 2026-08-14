#!/usr/bin/env python3
"""Tests unitarios del ciclo de vida de ModelManager."""

import asyncio
import json
import time

import pytest
import torch

from config.settings import settings
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


class CUDAOOMQwen3TTS:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 512.00 MiB"
        )


class CUDAOOMAfterLoad(FakeModel):
    def generate_voice_clone(self, **kwargs):
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 512.00 MiB"
        )


class FakeQwen3TTSWithCUDAOOM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return CUDAOOMAfterLoad()


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    (models_dir / "model-a").mkdir(parents=True)
    (models_dir / "model-b").mkdir()
    monkeypatch.setattr(settings.paths, "models_dir", str(models_dir))
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


def test_cancelled_inference_does_not_stick_in_generating(models_dir, monkeypatch):
    """Una inferencia cancelada (cliente desconectado/shutdown) no puede dejar
    el estado atascado en GENERATING: el modelo se cuarentena (ERROR) y es
    recuperable recargándolo, en vez de bloquear el servicio para siempre."""
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTS)
    mgr = ModelManager()

    async def scenario():
        await mgr.load_model("model-a")
        task = asyncio.create_task(
            mgr.generate_voice_clone(text="hola", language="es")
        )
        await asyncio.sleep(0.05)  # estado GENERATING
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        status = await mgr.get_model_status()
        assert status["state"] == ModelState.ERROR.value
        assert "cancelada" in status["error"].lower()
        assert await mgr.get_active_model() is None
        # Recuperable: se puede recargar
        await mgr.load_model("model-a")
        assert (await mgr.get_model_status())["state"] == ModelState.READY.value

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


def test_concurrent_loads_same_model_load_once(models_dir, monkeypatch):
    """Varias peticiones simultáneas del mismo modelo: solo una carga real."""
    calls = []

    class CountingTTS:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(1)
            time.sleep(0.2)
            return FakeModel()

    monkeypatch.setattr(mm, "Qwen3TTSModel", CountingTTS)
    mgr = ModelManager()

    async def scenario():
        results = await asyncio.gather(
            mgr.load_model("model-a"),
            mgr.load_model("model-a"),
            mgr.load_model("model-a"),
        )
        assert all(r.model_id == "model-a" for r in results)
        assert len(calls) == 1  # solo una instancia creada en GPU

    asyncio.run(scenario())

    assert _status(mgr)["state"] == ModelState.READY.value
    assert len(asyncio.run(mgr.get_loaded_models())) == 1


def test_concurrent_loads_same_model_share_error(models_dir, monkeypatch):
    """Si la carga en curso falla, todas las peticiones reciben el mismo error."""
    monkeypatch.setattr(mm, "Qwen3TTSModel", BrokenQwen3TTS)
    mgr = ModelManager()

    async def scenario():
        with pytest.raises(RuntimeError, match="OOM simulado"):
            await asyncio.gather(
                mgr.load_model("model-a"),
                mgr.load_model("model-a"),
            )

    asyncio.run(scenario())

    status = _status(mgr)
    assert status["state"] == ModelState.ERROR.value
    assert "OOM" in status["error"]


def test_retry_after_failed_load_allowed(models_dir, monkeypatch):
    """Tras un ERROR, una nueva carga del mismo modelo se reintenta."""
    failures = [True]

    class FlakyTTS:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            if failures[0]:
                failures[0] = False
                raise RuntimeError("fallo transitorio")
            return FakeModel()

    monkeypatch.setattr(mm, "Qwen3TTSModel", FlakyTTS)
    mgr = ModelManager()

    with pytest.raises(RuntimeError):
        asyncio.run(mgr.load_model("model-a"))

    info = asyncio.run(mgr.load_model("model-a"))
    assert info.model_id == "model-a"
    assert _status(mgr)["state"] == ModelState.READY.value


# -- Recuperación ante CUDA OOM -------------------------------------------


def test_cuda_oom_during_load_raises_controlled_error(models_dir, monkeypatch):
    """Un CUDA OOM al cargar eleva GPUOutOfMemoryError sin filtrar detalles."""
    monkeypatch.setattr(mm, "Qwen3TTSModel", CUDAOOMQwen3TTS)
    mgr = ModelManager()

    with pytest.raises(mm.GPUOutOfMemoryError) as ei:
        asyncio.run(mgr.load_model("model-a"))

    assert str(ei.value) == mm.GPU_OOM_MESSAGE
    assert mm.GPU_OOM_MESSAGE == "Not enough GPU memory to process this request."
    assert "CUDA out of memory" not in str(ei.value)


def test_cuda_oom_during_load_marks_error_and_frees_cache(models_dir, monkeypatch):
    """Tras el OOM: estado ERROR, mensaje público y cache de CUDA liberada."""
    empties = []
    monkeypatch.setattr(mm, "Qwen3TTSModel", CUDAOOMQwen3TTS)
    monkeypatch.setattr(mm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mm.torch.cuda, "empty_cache", lambda: empties.append(1))
    mgr = ModelManager()

    with pytest.raises(mm.GPUOutOfMemoryError):
        asyncio.run(mgr.load_model("model-a"))

    assert empties == [1]
    status = _status(mgr)
    assert status["state"] == ModelState.ERROR.value
    assert status["error"] == mm.GPU_OOM_MESSAGE


def test_concurrent_loads_same_model_share_cuda_oom(models_dir, monkeypatch):
    """Concurrentes: si la carga en curso muere por OOM, todas reciben el error controlado."""
    monkeypatch.setattr(mm, "Qwen3TTSModel", CUDAOOMQwen3TTS)
    mgr = ModelManager()

    async def scenario():
        with pytest.raises(mm.GPUOutOfMemoryError):
            await asyncio.gather(
                mgr.load_model("model-a"),
                mgr.load_model("model-a"),
            )

    asyncio.run(scenario())

    status = _status(mgr)
    assert status["state"] == ModelState.ERROR.value
    assert status["error"] == mm.GPU_OOM_MESSAGE


def test_cuda_oom_during_inference_recovers(models_dir, monkeypatch):
    """OOM en inferencia: estado ERROR, cache liberada y sin estado inconsistente."""
    empties = []
    monkeypatch.setattr(mm, "Qwen3TTSModel", FakeQwen3TTSWithCUDAOOM)
    monkeypatch.setattr(mm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mm.torch.cuda, "empty_cache", lambda: empties.append(1))
    mgr = ModelManager()

    info = asyncio.run(mgr.load_model("model-a"))
    assert info.model_id == "model-a"
    assert mgr.is_loaded()

    with pytest.raises(mm.GPUOutOfMemoryError):
        asyncio.run(mgr.generate_voice_clone(text="hola", language="es"))

    assert empties == [1]
    status = _status(mgr)
    assert status["state"] == ModelState.ERROR.value
    assert status["error"] == mm.GPU_OOM_MESSAGE
    assert asyncio.run(mgr.get_active_model()) is None


def test_gpu_oom_http_response_shape():
    """El handler unificado de errores devuelve el JSON controlado (503)."""
    from fastapi import Request
    from app import api_error_handler

    async def run():
        scope = {
            "type": "http",
            "headers": [(b"x-request-id", b"abc123")],
        }
        resp = await api_error_handler(Request(scope), mm.GPUOutOfMemoryError())
        return resp

    resp = asyncio.run(run())

    assert resp.status_code == 503
    body = json.loads(resp.body)
    assert body == {
        "error": {
            "code": "GPU_OUT_OF_MEMORY",
            "message": "Not enough GPU memory to process this request.",
            "request_id": "abc123",
        }
    }
