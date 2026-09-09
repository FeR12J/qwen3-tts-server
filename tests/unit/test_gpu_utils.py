#!/usr/bin/env python3
"""Tests de utilidades de GPU (conversión binaria de memoria)."""

from types import SimpleNamespace

from utils import gpu


def _fake_props(total_bytes: int):
    return SimpleNamespace(name="Fake GPU", total_memory=total_bytes)


def test_get_vram_available_usa_divisor_binario(monkeypatch):
    """97887 MiB reales => 95.6 GiB (no 102.6 GB con divisor decimal)."""
    monkeypatch.setattr(gpu, "_resolve_device", lambda: "cuda:0")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda i: _fake_props(97887 * 1024 * 1024),
    )
    assert gpu.get_vram_available() == 95.6


def test_get_vram_available_sin_cuda_cero(monkeypatch):
    monkeypatch.setattr(gpu, "_resolve_device", lambda: "cpu")
    assert gpu.get_vram_available() == 0.0


def test_list_devices_usa_divisor_binario(monkeypatch):
    """32 GiB exactos => 32.0 (nvidia-smi reporta 32607 MiB => 31.8)."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda i: _fake_props((32607 if i == 0 else 97887) * 1024 * 1024),
    )
    devices = gpu.list_devices()["devices"]
    assert devices[0]["vram_gb"] == 31.8
    assert devices[1]["vram_gb"] == 95.6


def test_list_devices_sin_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert gpu.list_devices() == {
        "cuda_available": False,
        "count": 0,
        "devices": [],
    }