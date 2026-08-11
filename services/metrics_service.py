#!/usr/bin/env python3
"""Métricas y registro de actividad del servidor."""

import logging
from datetime import datetime

from fastapi import Request

from utils.logging import log_request as _log_request
from utils.gpu import get_vram_available
from services.config_service import get_runtime_config

logger = logging.getLogger("tts")


class MetricsService:
    """Contadores de actividad y métricas de recursos."""

    def __init__(self, config):
        """config: config.settings.Settings (objeto Settings único)."""
        self._config = config
        self._request_count = 0
        self._last_request = None

    def log_request(self, req: Request, text: str):
        """Registrar una petición (contador + archivo de logs).

        Privacidad por defecto: solo se registra text_length, no el texto
        (log_input_text desactivado). El texto completo solo se registra si
        la configuración en tiempo de ejecución lo permite.
        """
        self._request_count += 1
        self._last_request = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._config.logging.log_file:
            _log_request(
                req,
                text,
                self._config.logging.log_file,
                self._config.logging.log_max_bytes,
                get_runtime_config().get("log_input_text", False),
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
