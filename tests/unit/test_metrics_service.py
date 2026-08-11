#!/usr/bin/env python3
"""Tests del contador de métricas (MetricsService + GET /metrics)."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.metrics_service import MetricsService


class DummyQueue:
    @property
    def queue_size(self):
        return 3


def _make(queue=None):
    config = SimpleNamespace(
        logging=SimpleNamespace(log_file=None, log_max_bytes=0)
    )
    return MetricsService(config, queue=queue)


def test_empty_metrics():
    svc = _make(DummyQueue())
    assert svc.get_metrics() == {
        "tts_requests": 0,
        "tts_errors": 0,
        "tts_active": 0,
        "tts_queue_size": 3,
        "whisper_requests": 0,
        "model_loads": 0,
        "model_unloads": 0,
        "average_tts_ms": 0,
        "average_queue_wait_ms": 0,
        "average_model_load_ms": 0,
        "average_rtf": 0.0,
        "average_ttfb_ms": 0,
        "average_vram_used_mb": 0,
    }


def test_tts_counters_and_average():
    svc = _make(DummyQueue())
    svc.tts_started()          # req 1
    svc.tts_started()          # req 2
    svc.tts_completed(1000, audio_duration_ms=5000, ttfb_ms=1000, vram_used_mb=4000)
    svc.tts_failed()           # req 2 falla
    svc.tts_started()          # req 3 (activa ahora)
    svc.tts_completed(2000, audio_duration_ms=10000, ttfb_ms=2000, vram_used_mb=4500)

    m = svc.get_metrics()
    assert m["tts_requests"] == 3
    assert m["tts_errors"] == 1
    assert m["tts_active"] == 0
    assert m["average_tts_ms"] == 1500  # (1000 + 2000) / 2


def test_audio_specific_metrics():
    """RTF, TTFB, VRAM, cola y carga del modelo (agregados por petición)."""
    svc = _make(DummyQueue())
    svc.tts_started()
    svc.queue_waited(250)
    svc.tts_completed(1000, audio_duration_ms=5000, ttfb_ms=700, vram_used_mb=4000)
    svc.tts_started()
    svc.queue_waited(750)
    svc.tts_completed(2000, audio_duration_ms=10000, ttfb_ms=900, vram_used_mb=4500)

    svc.model_loaded(3500)
    svc.model_loaded(4500)

    m = svc.get_metrics()
    assert m["average_queue_wait_ms"] == 500       # (250 + 750) / 2
    assert m["average_model_load_ms"] == 4000      # (3500 + 4500) / 2
    assert m["average_rtf"] == 0.2                 # 1000/5000 y 2000/10000
    assert m["average_ttfb_ms"] == 800             # (700 + 900) / 2
    assert m["average_vram_used_mb"] == 4250       # (4000 + 4500) / 2
    assert m["model_loads"] == 2


def test_ttfb_defaults_to_generation_when_omitted():
    """Sin ttfb explícito (síntesis no-streaming), TTFB = duración de generación."""
    svc = _make(DummyQueue())
    svc.tts_started()
    svc.tts_completed(842, audio_duration_ms=5100)
    m = svc.get_metrics()
    assert m["average_ttfb_ms"] == 842
    assert m["average_rtf"] == round(842 / 5100, 3)


def test_tts_active_tracks_in_flight():
    svc = _make(DummyQueue())
    svc.tts_started()
    svc.tts_started()
    assert svc.get_metrics()["tts_active"] == 2
    svc.tts_completed(500)
    assert svc.get_metrics()["tts_active"] == 1
    svc.tts_failed()
    assert svc.get_metrics()["tts_active"] == 0


def test_whisper_and_model_counters():
    svc = _make(DummyQueue())
    for _ in range(4):
        svc.whisper_requested()
    svc.model_loaded()
    svc.model_loaded()
    svc.model_unloaded()

    m = svc.get_metrics()
    assert m["whisper_requests"] == 4
    assert m["model_loads"] == 2
    assert m["model_unloads"] == 1


def test_queue_size_zero_without_queue():
    svc = _make()  # sin cola
    assert svc.get_metrics()["tts_queue_size"] == 0
