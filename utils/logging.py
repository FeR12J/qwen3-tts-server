#!/usr/bin/env python3
"""Configuración de logging y registro de peticiones."""

import os
import logging
from datetime import datetime

from fastapi import Request

from utils.text import truncate_text

logger = logging.getLogger("tts")


def setup_logging(level: str = "INFO"):
    """Configurar el logging raíz una sola vez."""
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def rotate_log_if_needed(log_file: str, max_bytes: int):
    """Rotar el archivo de logs si supera el tamaño máximo."""
    if max_bytes <= 0:
        return
    try:
        if os.path.exists(log_file) and os.path.getsize(log_file) > max_bytes:
            os.replace(log_file, log_file + ".old")
    except OSError as e:
        logger.warning(f"No se pudo rotar el log {log_file}: {e}")


def log_request(req: Request, text: str, log_file: str, max_bytes: int = 5 * 1024 * 1024):
    """Registrar solicitud en archivo de logs."""
    rotate_log_if_needed(log_file, max_bytes)
    ip = req.client.host if req.client else "unknown"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = truncate_text(text, 100)
    with open(log_file, "a") as f:
        f.write(f"[{ts}] {ip}: {log_text}\n")
