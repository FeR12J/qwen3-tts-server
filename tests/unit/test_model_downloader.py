#!/usr/bin/env python3
"""Tests del servicio de descarga de modelos (whitelist, estado, instalación)."""

import asyncio
import os

import pytest

from config.settings import settings
from services import model_downloader as md
from services.errors import APIError


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    """Redirigir models_dir a un directorio temporal y limpiar el estado."""
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(settings.paths, "models_dir", str(d))
    md._STATE.clear()
    yield d


def _write_weights(base: str, name: str):
    target = os.path.join(base, name)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "model.safetensors"), "wb") as f:
        f.write(b"x" * 1024)


# -- Whitelist ---------------------------------------------------------------


def test_solo_acepta_nombres_de_la_lista(models_dir):
    err = pytest.raises(APIError, asyncio.run, md.start_download("otro-modelo"))
    assert err.value.status_code == 400

    # El repo_id de HF NO es un id válido (solo el nombre local)
    err = pytest.raises(
        APIError, asyncio.run, md.start_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    )
    assert err.value.status_code == 400


def test_whitelist_cubre_modelos_soportados(models_dir):
    names = {m["name"] for m in md.SUPPORTED_MODELS}
    assert "Qwen3-TTS-12Hz-0.6B-CustomVoice" in names
    assert "Qwen3-TTS-12Hz-1.7B-Base" in names
    assert "Qwen3-TTS-12Hz-1.7B-CustomVoice" in names
    assert "Qwen3-TTS-12Hz-1.7B-VoiceDesign" in names
    assert "whisper-large-v3" in names
    # Cada repo es de Hugging Face (owner/repo)
    assert all("/" in m["repo_id"] for m in md.SUPPORTED_MODELS)


# -- Detección de instalación ------------------------------------------------


def test_is_installed_con_pesos(models_dir):
    _write_weights(str(models_dir), "whisper-large-v3")
    assert md._is_installed("whisper-large-v3")
    assert not md._is_installed("Qwen3-TTS-12Hz-1.7B-Base")


def test_is_installed_directorio_vacio_o_parcial(models_dir):
    os.makedirs(models_dir / "whisper-large-v3")
    assert not md._is_installed("whisper-large-v3")


def test_dir_size(models_dir):
    _write_weights(str(models_dir), "whisper-large-v3")
    assert md._dir_size("whisper-large-v3") == 1024
    assert md._dir_size("no-existe") is None


# -- Estado ------------------------------------------------------------------


def test_list_status_estados(models_dir):
    _write_weights(str(models_dir), "whisper-large-v3")
    md._STATE["Qwen3-TTS-12Hz-1.7B-Base"] = {"status": "downloading", "error": None}
    md._STATE["Qwen3-TTS-12Hz-0.6B-CustomVoice"] = {
        "status": "error", "error": "boom"
    }
    by_name = {m["name"]: m for m in md.list_status()}

    assert by_name["whisper-large-v3"]["installed"] is True
    assert by_name["whisper-large-v3"]["status"] == "done"

    # En descarga o en error: el estado prevalece aunque el dir esté vacío
    assert by_name["Qwen3-TTS-12Hz-1.7B-Base"]["status"] == "downloading"
    assert by_name["Qwen3-TTS-12Hz-0.6B-CustomVoice"]["status"] == "error"
    assert by_name["Qwen3-TTS-12Hz-0.6B-CustomVoice"]["error"] == "boom"

    assert by_name["Qwen3-TTS-12Hz-1.7B-VoiceDesign"]["installed"] is False
    assert by_name["Qwen3-TTS-12Hz-1.7B-VoiceDesign"]["status"] == "idle"


# -- Descarga ----------------------------------------------------------------


def test_start_download_no_op_si_instalado(models_dir):
    _write_weights(str(models_dir), "whisper-large-v3")
    res = asyncio.run(md.start_download("whisper-large-v3"))
    assert res["started"] is False
    assert md._STATE["whisper-large-v3"]["status"] == "done"


def test_start_download_lanza_en_segundo_plano(models_dir, monkeypatch):
    def fake_snapshot(model):
        _write_weights(str(models_dir), model["name"])

    monkeypatch.setattr(md, "_snapshot_download", fake_snapshot)

    async def main():
        res = await md.start_download("whisper-large-v3")
        assert res["started"] is True
        assert md._STATE["whisper-large-v3"]["status"] == "downloading"
        # Esperar a que termine la tarea en segundo plano
        for _ in range(100):
            if md._STATE["whisper-large-v3"]["status"] != "downloading":
                break
            await asyncio.sleep(0.01)
        assert md._STATE["whisper-large-v3"]["status"] == "done"
        assert md._is_installed("whisper-large-v3")

    asyncio.run(main())


def test_una_sola_descarga_a_la_vez(models_dir):
    md._STATE["whisper-large-v3"] = {"status": "downloading", "error": None}

    with pytest.raises(APIError) as exc:
        asyncio.run(md.start_download("Qwen3-TTS-12Hz-1.7B-Base"))
    assert exc.value.status_code == 409

    with pytest.raises(APIError) as exc:
        asyncio.run(md.start_download("whisper-large-v3"))
    assert exc.value.status_code == 409


def test_descarga_fallida_marca_error(models_dir, monkeypatch):
    def fail_snapshot(model):
        raise RuntimeError("red caida")

    monkeypatch.setattr(md, "_snapshot_download", fail_snapshot)

    async def main():
        await md.start_download("whisper-large-v3")
        for _ in range(100):
            if md._STATE["whisper-large-v3"]["status"] != "downloading":
                break
            await asyncio.sleep(0.01)
        assert md._STATE["whisper-large-v3"]["status"] == "error"
        assert "red caida" in md._STATE["whisper-large-v3"]["error"]
        assert not md._is_installed("whisper-large-v3")

    asyncio.run(main())