#!/usr/bin/env python3
"""Métricas y registro de actividad del servidor."""

import logging
from datetime import datetime

from fastapi import Request

from utils.logging import log_request as _log_request
from utils.gpu import get_vram_available

logger = logging.getLogger("tts")


class MetricsService:
    """Contadores de actividad y métricas de recursos."""

    def __init__(self, config: dict):
        self._config = config
        self._request_count = 0
        self._last_request = None

    def log_request(self, req: Request, text: str):
        """Registrar una petición (contador + archivo de logs)."""
        self._request_count += 1
        self._last_request = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._config.get("log_file"):
            _log_request(
                req,
                text,
                self._config["log_file"],
                self._config.get("log_max_bytes", 5 * 1024 * 1024),
            )

    def vram_available_gb(self) -> float:
        """VRAM disponible (GB) de la GPU seleccionada."""
        return get_vram_available()

    def snapshot(self) -> dict:
        """Resumen de actividad del servidor."""
        return {
            "request_count": self._request_count,
            "last_request": self._last_request,
            "vram_available_gb": self.vram_available_gb(),
        }
