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
    """Contadores de actividad y métricas de recursos.

    Métricas sencillas de uso para GET /metrics (sin Prometheus obligatorio):

        tts_requests / tts_errors / tts_active / tts_queue_size
        whisper_requests / model_loads / model_unloads / average_tts_ms

    Métricas específicas de audio (agregadas por petición):

        request latency / queue wait / model load / generation / audio duration
        RTF (generación / duración de audio) / TTFB / VRAM usada

    Los servicios llaman a los hooks (tts_started, tts_completed,
    tts_failed, ...); tts_queue_size se lee de QueueService en get_metrics.
    """

    def __init__(self, config, queue=None):
        """config: config.settings.Settings (objeto Settings único)."""
        self._config = config
        self._queue = queue
        self._request_count = 0
        self._last_request = None
        self._tts_requests = 0
        self._tts_errors = 0
        self._tts_active = 0
        self._tts_duration_sum_ms = 0
        self._tts_duration_count = 0
        self._whisper_requests = 0
        self._model_loads = 0
        self._model_unloads = 0
        # Métricas específicas de audio (sumas/recuentos -> medias)
        self._queue_wait_sum_ms = 0
        self._queue_wait_count = 0
        self._model_load_sum_ms = 0
        self._model_load_count = 0
        self._rtf_sum = 0.0
        self._rtf_count = 0
        self._ttfb_sum_ms = 0
        self._ttfb_count = 0
        self._vram_sum_mb = 0
        self._vram_count = 0

    # -- Hooks de eventos (llamados por los servicios) ---------------------

    def tts_started(self):
        """Una generación TTS (síntesis o streaming) ha comenzado."""
        self._tts_requests += 1
        self._tts_active += 1

    def tts_completed(self, duration_ms: int, audio_duration_ms: int = 0,
                      ttfb_ms: int = None, vram_used_mb: int = None):
        """Una generación TTS terminó correctamente.

        - duration_ms: tiempo real de generación (GPU).
        - audio_duration_ms: duración del audio generado (ms) -> RTF.
        - ttfb_ms: tiempo hasta el primer byte/audio (None -> duration_ms).
        - vram_used_mb: VRAM usada al terminar (None -> se consulta).
        """
        self._tts_active = max(0, self._tts_active - 1)
        self._tts_duration_sum_ms += duration_ms
        self._tts_duration_count += 1
        if audio_duration_ms > 0:
            self._rtf_sum += duration_ms / audio_duration_ms
            self._rtf_count += 1
        if ttfb_ms is None:
            ttfb_ms = duration_ms
        self._ttfb_sum_ms += ttfb_ms
        self._ttfb_count += 1
        if vram_used_mb is None:
            vram_used_mb = self.vram_used_mb()
        self._vram_sum_mb += vram_used_mb
        self._vram_count += 1

    def tts_failed(self):
        """Una generación TTS falló."""
        self._tts_active = max(0, self._tts_active - 1)
        self._tts_errors += 1

    def queue_waited(self, wait_ms: int):
        """Tiempo de espera en la cola de inferencia (hasta el turno de GPU)."""
        self._queue_wait_sum_ms += wait_ms
        self._queue_wait_count += 1

    def whisper_requested(self):
        """Una transcripción Whisper ha comenzado."""
        self._whisper_requests += 1

    def model_loaded(self, load_ms: int = None):
        """Un modelo TTS se cargó en GPU (opcional: duración de la carga)."""
        self._model_loads += 1
        if load_ms is not None:
            self._model_load_sum_ms += load_ms
            self._model_load_count += 1

    def model_unloaded(self):
        """Un modelo TTS se descargó (y liberó su VRAM)."""
        self._model_unloads += 1

    # -- Consulta ----------------------------------------------------------

    @staticmethod
    def vram_used_mb() -> int:
        """VRAM usada (MB) de la GPU en uso; 0 si no hay GPU/error."""
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                return int((total_b - free_b) // (1024 * 1024))
        except Exception:
            pass
        return 0

    def get_metrics(self) -> dict:
        """Resumen de métricas (GET /metrics)."""
        average_ms = 0
        if self._tts_duration_count > 0:
            average_ms = int(self._tts_duration_sum_ms / self._tts_duration_count)

        def avg(s, c):
            return s // c if c > 0 else 0
        return {
            "tts_requests": self._tts_requests,
            "tts_errors": self._tts_errors,
            "tts_active": self._tts_active,
            "tts_queue_size": self._queue.queue_size if self._queue else 0,
            "whisper_requests": self._whisper_requests,
            "model_loads": self._model_loads,
            "model_unloads": self._model_unloads,
            "average_tts_ms": average_ms,
            # Métricas específicas de audio (medias)
            "average_queue_wait_ms": avg(self._queue_wait_sum_ms, self._queue_wait_count),
            "average_model_load_ms": avg(self._model_load_sum_ms, self._model_load_count),
            "average_rtf": round(self._rtf_sum / self._rtf_count, 3) if self._rtf_count else 0.0,
            "average_ttfb_ms": avg(self._ttfb_sum_ms, self._ttfb_count),
            "average_vram_used_mb": avg(self._vram_sum_mb, self._vram_count),
        }

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
