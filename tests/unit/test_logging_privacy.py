#!/usr/bin/env python3
"""Tests de privacidad en logs: por defecto solo se registra text_length,
nunca el texto enviado al TTS (log_input_text desactivado)."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logging import log_request


def _req():
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))


def test_default_logs_only_text_length(tmp_path):
    """Por defecto (log_input_text=False) el texto no aparece en el log."""
    log_file = str(tmp_path / "requests.log")
    text = "texto secreto y privado"
    log_request(_req(), text, log_file)
    line = open(log_file).read().strip()
    assert f"text_length={len(text)}" in line
    assert "texto secreto" not in line


def test_log_input_text_enabled_logs_truncated_text(tmp_path):
    """Con log_input_text=True se registra el texto truncado."""
    log_file = str(tmp_path / "requests.log")
    text = "x" * 500
    log_request(_req(), text, log_file, log_input_text=True)
    line = open(log_file).read().strip()
    assert "text_length=500" in line
    assert "x" * 100 in line
    assert "x" * 101 not in line  # truncado a 100 caracteres


def test_metrics_service_respects_runtime_flag(tmp_path, monkeypatch):
    """MetricsService lee log_input_text de la configuración en tiempo de
    ejecución: false por defecto (solo text_length)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.config_service import get_runtime_config
    from services.metrics_service import MetricsService

    assert get_runtime_config().get("log_input_text", False) is False

    log_file = str(tmp_path / "requests.log")
    config = SimpleNamespace(logging=SimpleNamespace(log_file=log_file, log_max_bytes=0))
    svc = MetricsService(config)
    svc.log_request(_req(), "contenido privado")

    line = open(log_file).read().strip()
    assert "text_length=17" in line
    assert "contenido privado" not in line

    # Al activar log_input_text en runtime, el texto sí se registra
    from services import config_service
    config_service.settings.runtime = config_service.settings.runtime.model_copy(
        update={"log_input_text": True}
    )
    try:
        visible = "contenido visible"
        svc.log_request(_req(), visible)
        line = open(log_file).read().strip().splitlines()[-1]
        assert f"text_length={len(visible)}" in line
        assert "contenido visible" in line
    finally:
        config_service.settings.runtime = config_service.settings.runtime.model_copy(
            update={"log_input_text": False}
        )
