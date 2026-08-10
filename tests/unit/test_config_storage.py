#!/usr/bin/env python3
"""Tests unitarios del almacenamiento de configuración."""

import json
import os

from storage import config_storage


def test_runtime_file_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config_storage, "RUNTIME_FILE", str(tmp_path / "runtime.json"))

    assert config_storage.load_runtime_file() == {}
    config_storage.save_runtime_file({"max_text_chars": 500, "device": "cpu"})

    data = config_storage.load_runtime_file()
    assert data["max_text_chars"] == 500
    assert data["device"] == "cpu"


def test_runtime_file_corrupt(tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    path.write_text("not json {{{")
    monkeypatch.setattr(config_storage, "RUNTIME_FILE", str(path))

    assert config_storage.load_runtime_file() == {}


def test_runtime_file_not_dict(tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps([1, 2, 3]))
    monkeypatch.setattr(config_storage, "RUNTIME_FILE", str(path))

    assert config_storage.load_runtime_file() == {}
