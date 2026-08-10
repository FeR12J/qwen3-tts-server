#!/usr/bin/env python3
"""Tests de claves API (almacenamiento con hash) y niveles de protección."""

import json

import pytest
from fastapi import HTTPException

from config.settings import settings
from security import auth
from services import apikey_service
from storage import config_storage


class FakeRequest:
    """Request mínimo con cabeceras para probar dependencias."""

    def __init__(self, headers=None):
        self.headers = headers or {}


@pytest.fixture
def keys_file(tmp_path, monkeypatch):
    """Aislar la persistencia de claves en un archivo temporal vacío."""
    import storage.api_key_storage as aks
    path = tmp_path / "apikeys.json"
    monkeypatch.setattr(aks, "APIKEYS_FILE", str(path))
    monkeypatch.setattr(apikey_service, "_keys", None)
    return path


# -- Almacenamiento con hash ---------------------------------------------


def test_key_stored_only_as_hash(keys_file):
    created = apikey_service.create_key("OpenWebUI")
    data = json.loads(keys_file.read_text())
    assert len(data) == 1
    entry = data[0]
    assert "key" not in entry or entry.get("key") is None
    assert entry["id"].startswith("key_")
    assert entry["name"] == "OpenWebUI"
    assert entry["key_hash"] != created["key"]
    assert entry["key_hash"] == apikey_service._hash(created["key"])
    assert entry["created_at"]
    assert entry.get("last_used_at") is None


def test_full_key_returned_only_on_create(keys_file):
    created = apikey_service.create_key("OpenWebUI")
    assert created["key"].startswith("qt-")
    listed = apikey_service.list_keys()
    assert len(listed) == 1
    out = listed[0]
    assert "key" not in out
    assert "key_hash" not in out
    assert created["key"] not in str(out)
    stored = keys_file.read_text()
    assert created["key"] not in stored
    assert "key_hash" in stored


def test_verify_key_and_last_used(keys_file):
    created = apikey_service.create_key("OpenWebUI")
    assert apikey_service.verify_key(created["key"]) is True
    entry = json.loads(keys_file.read_text())[0]
    assert entry["last_used_at"] is not None
    assert apikey_service.verify_key("qt-invalida") is False


def test_delete_and_toggle(keys_file):
    created = apikey_service.create_key("A")
    created2 = apikey_service.create_key("B")
    apikey_service.toggle_key(created["id"])
    assert apikey_service.verify_key(created["key"]) is False
    assert apikey_service.verify_key(created2["key"]) is True
    apikey_service.delete_key(created["id"])
    with pytest.raises(KeyError):
        apikey_service.delete_key(created["id"])
    assert len(apikey_service.list_keys()) == 1


# -- Niveles de protección ------------------------------------------------


@pytest.mark.asyncio
async def test_require_api_key_disabled_allows(keys_file, monkeypatch):
    monkeypatch.setattr(settings.runtime, "api_keys_enabled", False)
    await auth.require_api_key(FakeRequest())


@pytest.mark.asyncio
async def test_require_api_key_enabled_blocks_without_key(keys_file, monkeypatch):
    monkeypatch.setattr(settings.runtime, "api_keys_enabled", True)
    with pytest.raises(HTTPException) as e:
        await auth.require_api_key(FakeRequest())
    assert e.value.status_code == 401


@pytest.mark.asyncio
async def test_require_api_key_enabled_allows_valid_key(keys_file, monkeypatch):
    monkeypatch.setattr(settings.runtime, "api_keys_enabled", True)
    created = apikey_service.create_key("OpenWebUI")
    await auth.require_api_key(FakeRequest({"x-api-key": created["key"]}))


@pytest.mark.asyncio
async def test_require_admin_bootstrap_no_keys(keys_file):
    # Sin claves existentes se permite (para crear la primera)
    await auth.require_admin(FakeRequest())


@pytest.mark.asyncio
async def test_require_admin_requires_key_even_if_disabled(keys_file, monkeypatch):
    monkeypatch.setattr(settings.runtime, "api_keys_enabled", False)
    created = apikey_service.create_key("OpenWebUI")
    with pytest.raises(HTTPException) as e:
        await auth.require_admin(FakeRequest())
    assert e.value.status_code == 401
    await auth.require_admin(FakeRequest({"authorization": f"Bearer {created['key']}"}))


@pytest.mark.asyncio
async def test_require_admin_invalid_key(keys_file):
    apikey_service.create_key("OpenWebUI")
    with pytest.raises(HTTPException):
        await auth.require_admin(FakeRequest({"x-api-key": "qt-0000"}))
