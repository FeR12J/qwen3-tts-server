#!/usr/bin/env python3
"""Tests unitarios de VoiceManager (CRUD de voces locales)."""

import json
import os
import re

import pytest

from services.voice_manager import VoiceManager
from storage import voice_storage

REFERENCE_WAV = b"RIFF fake wav data" + b"\x00" * 256


class FakeModelManager:
    """Simula ModelManager: registra los prompts de clonación creados."""

    def __init__(self):
        self.clone_calls = []

    async def create_voice_clone_prompt(self, wav_path, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        self.clone_calls.append((wav_path, txt_path, text))
        return {"prompt": text}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    fake_settings = type("FakeSettings", (), {
        "paths": type("Paths", (), {"voices_dir": str(tmp_path)})(),
    })()
    monkeypatch.setattr(voice_storage, "settings", fake_settings)
    models = FakeModelManager()
    return VoiceManager(models), models, str(tmp_path)


def _voice_dir(root, voice_id):
    return os.path.join(root, voice_id)


def test_create_writes_new_structure(manager):
    vm, models, root = manager
    import asyncio
    voice_id = asyncio.run(vm.create("Narrador", "Texto de referencia",
                                     REFERENCE_WAV, language="es",
                                     description="Voz masculina cálida"))
    assert re.match(r"^voice_[0-9a-f]{8}$", voice_id), voice_id
    vdir = _voice_dir(root, voice_id)
    assert os.path.exists(os.path.join(vdir, "reference.wav"))
    assert os.path.exists(os.path.join(vdir, "reference.txt"))
    assert os.path.exists(os.path.join(vdir, "metadata.json"))

    meta = json.load(open(os.path.join(vdir, "metadata.json"), encoding="utf-8"))
    assert meta["id"] == voice_id
    assert meta["name"] == "Narrador"
    assert meta["language"] == "es"
    assert meta["description"] == "Voz masculina cálida"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", meta["created_at"])
    assert meta["reference_audio"] == "reference.wav"
    assert meta["reference_text"] == "Texto de referencia"

    with open(os.path.join(vdir, "reference.txt"), encoding="utf-8") as f:
        assert f.read() == "Texto de referencia"
    # Se aplica como clon activo
    assert vm.clone_active and vm.active_voice_id == voice_id
    assert len(models.clone_calls) == 1


def test_server_generated_ids_are_unique_and_safe(manager):
    """El id lo genera el servidor (voice_<hex>): el cliente no controla rutas."""
    vm, _, _ = manager
    import asyncio

    ids = {
        asyncio.run(vm.create(f"Nombre {i}", "texto", REFERENCE_WAV))
        for i in range(10)
    }
    assert len(ids) == 10
    for voice_id in ids:
        assert re.match(r"^voice_[0-9a-f]{8}$", voice_id), voice_id
        assert vm.get(voice_id)["name"].startswith("Nombre ")


def test_create_rejects_path_like_names(manager):
    vm, _, _ = manager
    import asyncio

    for evil in ("../x", "C:\\Windows", "/etc/passwd", "a/b", "..", "."):
        with pytest.raises(ValueError):
            asyncio.run(vm.create(evil, "texto", REFERENCE_WAV))


def test_create_requires_name_and_text(manager):
    vm, _, _ = manager
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(vm.create("", "texto", REFERENCE_WAV))
    with pytest.raises(ValueError):
        asyncio.run(vm.create("Voz", "", REFERENCE_WAV))


def test_list_and_get(manager):
    vm, _, _ = manager
    import asyncio

    ida = asyncio.run(vm.create("Alfa", "uno", REFERENCE_WAV))
    idb = asyncio.run(vm.create("Beta", "dos", REFERENCE_WAV))

    items = vm.list()
    assert {v["id"] for v in items} == {ida, idb}
    assert all(v["valid"] for v in items)
    by_name = {v["name"]: v for v in items}
    assert by_name["Alfa"]["id"] == ida

    meta = vm.get(idb)
    assert meta["id"] == idb and meta["language"] is None

    with pytest.raises(FileNotFoundError):
        vm.get("no-existe")


def test_update_metadata_and_reference(manager):
    vm, models, root = manager
    import asyncio

    vid = asyncio.run(vm.create("Vieja", "texto antiguo", REFERENCE_WAV))
    vm.unload_voice()

    asyncio.run(vm.update(vid, name="Nueva", language="en", description="d"))
    meta = vm.get(vid)
    assert meta["name"] == "Nueva" and meta["language"] == "en"
    assert meta["description"] == "d"
    assert meta["reference_text"] == "texto antiguo"

    asyncio.run(vm.update(vid, text="nuevo texto"))
    assert vm.get(vid)["reference_text"] == "nuevo texto"
    with open(os.path.join(root, vid, "reference.txt"), encoding="utf-8") as f:
        assert f.read() == "nuevo texto"
    # Sin clon activo, no se regenera el prompt.
    assert len(models.clone_calls) == 1

    # Con clon activo y referencia cambiada, se re-clona.
    asyncio.run(vm.load_voice(vid))
    assert len(models.clone_calls) == 2
    asyncio.run(vm.update(vid, audio_bytes=REFERENCE_WAV + b"X"))
    assert len(models.clone_calls) == 3

    with pytest.raises(FileNotFoundError):
        asyncio.run(vm.update("no-existe", name="x"))


def test_delete(manager):
    vm, _, root = manager
    import asyncio

    del_id = asyncio.run(vm.create("Borrar", "texto", REFERENCE_WAV))
    assert os.path.isdir(_voice_dir(root, del_id))

    assert vm.delete(del_id) is True
    assert not os.path.exists(_voice_dir(root, del_id))
    assert vm.delete(del_id) is False

    # Borrar el clon activo desactiva el voice cloning.
    del2_id = asyncio.run(vm.create("Borrar2", "texto", REFERENCE_WAV))
    assert vm.clone_active
    vm.delete(del2_id)
    assert not vm.clone_active


def test_get_reference_by_id_and_name(manager):
    vm, _, _ = manager
    import asyncio

    vid = asyncio.run(vm.create("Narrador", "texto", REFERENCE_WAV))
    ref_by_id = vm.get_reference(vid)
    ref_by_name = vm.get_reference("Narrador")
    assert ref_by_id is not None and ref_by_name is not None
    assert ref_by_id == ref_by_name
    assert os.path.basename(ref_by_id[0]) == "reference.wav"
    assert os.path.basename(ref_by_id[1]) == "reference.txt"
    assert vm.get_reference("no-existe") is None


def test_legacy_voice_compatibility(manager):
    """Directorio legado (voice.wav + text.txt, sin metadata.json)."""
    vm, _, root = manager
    legacy_dir = _voice_dir(root, "legacy")
    os.makedirs(legacy_dir)
    with open(os.path.join(legacy_dir, "voice.wav"), "wb") as f:
        f.write(REFERENCE_WAV)
    with open(os.path.join(legacy_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write("texto legado")

    items = vm.list()
    assert any(v["id"] == "legacy" for v in items)
    legacy = next(v for v in items if v["id"] == "legacy")
    assert legacy["valid"] is True
    assert legacy["name"] == "legacy"

    ref = vm.get_reference("legacy")
    assert ref is not None
    assert os.path.basename(ref[0]) == "voice.wav"

    # get() sintetiza la metadata legada.
    meta = vm.get("legacy")
    assert meta["reference_text"] == "texto legado"


def test_load_and_unload_voice(manager):
    vm, models, _ = manager
    import asyncio

    with pytest.raises(FileNotFoundError):
        asyncio.run(vm.load_voice("no-existe"))

    asyncio.run(vm.create("Voz", "texto", REFERENCE_WAV))
    assert vm.clone_active
    assert vm.unload_voice() is True
    assert not vm.clone_active and vm.active_voice_id is None
    assert vm.unload_voice() is False


@pytest.mark.parametrize("evil", [
    "/home/user/voices/myvoice/reference.wav",
    "/etc/passwd",
    "voice_001/../../x",
    "../voice_001",
    "..",
    ".",
    "a/b",
    "a\\b",
    "",
])
def test_path_like_voice_values_rejected(manager, evil):
    """El cliente no puede referirse a voces por rutas internas."""
    vm, _, _ = manager
    import asyncio

    asyncio.run(vm.create("Narrador", "texto", REFERENCE_WAV))
    assert vm.get_reference(evil) is None, evil
    with pytest.raises(FileNotFoundError):
        asyncio.run(vm.load_voice(evil))
    with pytest.raises(FileNotFoundError):
        vm.get(evil)


def test_storage_voice_dir_containment(manager):
    """La capa de almacenamiento también bloquea escapes de voices_dir."""
    vm, _, _ = manager
    for evil in ("../x", "/etc/passwd", "a/../../x"):
        with pytest.raises(voice_storage.VoiceNotFoundError):
            voice_storage.get_voice_files(evil)


def test_symlink_escape_rejected(manager):
    """Un enlace simbólico dentro de voices/ apuntando fuera no se resuelve."""
    vm, _, root = manager
    import asyncio

    outside = root + "_outside"
    os.makedirs(outside)
    asyncio.run(vm.create("Real", "texto", REFERENCE_WAV))
    os.symlink(outside, os.path.join(root, "link"))

    assert vm.get_reference("link") is None
    with pytest.raises(FileNotFoundError):
        vm.get("link")
    # Se lista como inválida (no resolvible), nunca escapa del directorio.
    link = next((v for v in vm.list() if v["id"] == "link"), None)
    assert link is not None and link["valid"] is False
