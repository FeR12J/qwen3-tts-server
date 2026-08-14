#!/usr/bin/env python3
"""Tests de resolución y validación de dispositivo (cuda/cuda:N/cpu) y dtype."""

import contextlib

import pytest
import torch

from config.settings import settings
from services import config_service as cs


@pytest.fixture
def cuda_2_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device", lambda idx: contextlib.nullcontext())


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture
def bf16_supported(cuda_2_gpus, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)


@pytest.fixture
def bf16_unsupported(cuda_2_gpus, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)


# -- Resolución de dispositivo ----------------------------------------------


def test_resolve_device_cpu(monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cpu")
    assert cs.resolve_device() == "cpu"


def test_resolve_device_bare_cuda(cuda_2_gpus, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda")
    assert cs.resolve_device() == "cuda:0"


def test_resolve_device_cuda_1(cuda_2_gpus, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda:1")
    assert cs.resolve_device() == "cuda:1"


def test_resolve_device_missing_gpu_falls_back(cuda_2_gpus, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda:5")
    assert cs.resolve_device() == "cuda:0"


def test_resolve_device_cuda_without_cuda_falls_back(no_cuda, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda")
    assert cs.resolve_device() == "cpu"


# -- Validación de dispositivo ----------------------------------------------


def test_validate_device(cuda_2_gpus):
    assert cs.validate_device("auto") is True
    assert cs.validate_device("cpu") is True
    assert cs.validate_device("cuda") is True
    assert cs.validate_device("cuda:0") is True
    assert cs.validate_device("cuda:1") is True
    assert cs.validate_device("cuda:2") is False
    assert cs.validate_device("cuda:x") is False
    assert cs.validate_device("mps") is False


def test_validate_device_without_cuda(no_cuda):
    assert cs.validate_device("cuda") is False
    assert cs.validate_device("cuda:0") is False
    assert cs.validate_device("cpu") is True


# -- Validación estricta antes de cargar (validated_device) ------------------


def test_validated_device_existing_gpu(cuda_2_gpus, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda:1")
    assert cs.validated_device() == "cuda:1"


def test_validated_device_missing_gpu_raises(cuda_2_gpus, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda:3")
    with pytest.raises(ValueError, match="no existe"):
        cs.validated_device()


def test_validated_device_cuda_without_cuda_raises(no_cuda, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda")
    with pytest.raises(ValueError, match="no hay ninguna GPU CUDA"):
        cs.validated_device()


def test_validated_device_invalid_raises(no_cuda, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "mps")
    with pytest.raises(ValueError, match="inválido"):
        cs.validated_device()


def test_validated_device_auto_without_cuda(no_cuda, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "auto")
    assert cs.validated_device() == "cpu"


def test_validated_device_auto_with_cuda(cuda_2_gpus, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "auto")
    assert cs.validated_device() == "cuda:0"


# -- Resolución de dtype (auto, según hardware) ------------------------------


def test_resolve_dtype_auto_bf16_when_supported(bf16_supported, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "auto")
    monkeypatch.setattr(settings.runtime, "dtype", "auto")
    assert cs.resolve_dtype() == "bfloat16"
    assert cs.validated_dtype() == "bfloat16"


def test_resolve_dtype_auto_fp16_when_bf16_unsupported(bf16_unsupported, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "auto")
    monkeypatch.setattr(settings.runtime, "dtype", "auto")
    assert cs.resolve_dtype() == "float16"
    assert cs.validated_dtype() == "float16"


def test_resolve_dtype_auto_cpu(no_cuda, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "auto")
    monkeypatch.setattr(settings.runtime, "dtype", "auto")
    assert cs.resolve_dtype() == "float32"


def test_resolve_dtype_auto_in_gpu_without_bf16(bf16_unsupported, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda:1")
    monkeypatch.setattr(settings.runtime, "dtype", "auto")
    assert cs.resolve_dtype() == "float16"


# -- Validación estricta de dtype (no asumir bfloat16 en todas las GPUs) -----


def test_validated_dtype_explicit_bf16_unsupported_raises(bf16_unsupported, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda:0")
    monkeypatch.setattr(settings.runtime, "dtype", "bfloat16")
    with pytest.raises(ValueError, match="no soportado"):
        cs.validated_dtype()


def test_validated_dtype_explicit_bf16_supported_ok(bf16_supported, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda:0")
    monkeypatch.setattr(settings.runtime, "dtype", "bfloat16")
    assert cs.validated_dtype() == "bfloat16"


def test_validated_dtype_explicit_float16(bf16_unsupported, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cuda:0")
    monkeypatch.setattr(settings.runtime, "dtype", "float16")
    assert cs.validated_dtype() == "float16"


def test_validated_dtype_explicit_float32_cpu(no_cuda, monkeypatch):
    monkeypatch.setattr(settings.runtime, "device", "cpu")
    monkeypatch.setattr(settings.runtime, "dtype", "float32")
    assert cs.validated_dtype() == "float32"


# -- Runtime heredado de los grupos estáticos --------------------------------


def test_load_runtime_config_seeds_from_static_groups(monkeypatch):
    """Los ajustes editables se siembran desde los grupos estáticos (fuente
    de las variables de entorno), y el archivo persistido los reemplaza."""
    original_runtime = settings.runtime
    monkeypatch.setattr(settings.limits, "max_channels", 4)
    monkeypatch.setattr(settings.queue, "max_parallel_inference", 3)
    monkeypatch.setattr(
        "services.config_service.load_runtime_file",
        lambda: {"max_channels": 6},
    )
    cs.load_runtime_config()
    assert settings.runtime.max_channels == 6          # runtime.json gana al seed
    assert settings.runtime.max_parallel_inference == 3  # seed desde queue
    assert settings.runtime.max_voice_audio_bytes_mb == (
        settings.limits.max_voice_audio_bytes // (1024 * 1024)
    )
    assert settings.runtime.max_transcribe_audio_bytes_mb == (
        settings.limits.max_transcribe_audio_bytes // (1024 * 1024)
    )
    # Restaurar el singleton al estado previo (load_runtime_config lo reemplaza)
    settings.runtime = original_runtime


def test_load_runtime_config_env_beats_file(monkeypatch):
    """Precedencia documentada: una variable de entorno EXPLÍCITA gana al
    archivo persistido (QWEN_TTS_LIMITS__MAX_CHANNELS=4 > runtime.json=6)."""
    original_runtime = settings.runtime
    monkeypatch.setenv("QWEN_TTS_LIMITS__MAX_CHANNELS", "4")
    # Lo que pydantic resolvería desde la variable en un arranque real:
    monkeypatch.setattr(settings.limits, "max_channels", 4)
    monkeypatch.setattr(
        "services.config_service.load_runtime_file",
        lambda: {"max_channels": 6},
    )
    cs.load_runtime_config()
    assert settings.runtime.max_channels == 4
    settings.runtime = original_runtime


def test_load_runtime_config_fresh_install_seeds_port(monkeypatch):
    """Instalación limpia (sin runtime.json): runtime.port se siembra desde
    server.port (donde cae QWEN_TTS_PORT), en vez de quedarse en el default
    8001 e ignorar la variable de entorno."""
    original_runtime = settings.runtime
    monkeypatch.setattr(settings.server, "port", 9999)
    monkeypatch.setattr(
        "services.config_service.load_runtime_file",
        lambda: {},
    )
    cs.load_runtime_config()
    assert settings.runtime.port == 9999
    settings.runtime = original_runtime


def test_load_runtime_config_seeds_whisper_model_from_static(monkeypatch):
    """Instalación limpia: runtime.whisper_model se siembra desde el grupo
    estático whisper.whisper_model (editable después desde el panel)."""
    original_runtime = settings.runtime
    monkeypatch.setattr(settings.whisper, "whisper_model", "whisper-medium")
    monkeypatch.setattr(
        "services.config_service.load_runtime_file",
        lambda: {},
    )
    cs.load_runtime_config()
    assert settings.runtime.whisper_model == "whisper-medium"
    settings.runtime = original_runtime


def test_load_runtime_config_env_beats_file_whisper_model(monkeypatch):
    """Precedencia: QWEN_TTS_WHISPER__WHISPER_MODEL=whisper-small gana a un
    runtime.json con whisper-large-v3."""
    original_runtime = settings.runtime
    monkeypatch.setenv("QWEN_TTS_WHISPER__WHISPER_MODEL", "whisper-small")
    # Lo que pydantic resolvería desde la variable en un arranque real:
    monkeypatch.setattr(settings.whisper, "whisper_model", "whisper-small")
    monkeypatch.setattr(
        "services.config_service.load_runtime_file",
        lambda: {"whisper_model": "whisper-large-v3"},
    )
    cs.load_runtime_config()
    assert settings.runtime.whisper_model == "whisper-small"
    settings.runtime = original_runtime
