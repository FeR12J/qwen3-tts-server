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


def log_request(req: Request, text: str, log_file: str, max_bytes: int = 5 * 1024 * 1024,
                log_input_text: bool = False):
    """Registrar solicitud en archivo de logs.

    Privacidad por defecto: solo se registra ``text_length`` (longitud del
    texto), nunca el contenido. Solo si ``log_input_text`` está activado se
    registra el texto (truncado a 100 caracteres).
    """
    rotate_log_if_needed(log_file, max_bytes)
    ip = req.client.host if req.client else "unknown"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if log_input_text:
        log_text = truncate_text(text, 100)
        with open(log_file, "a") as f:
            f.write(f"[{ts}] {ip} text_length={len(text)}: {log_text}\n")
    else:
        with open(log_file, "a") as f:
            f.write(f"[{ts}] {ip} text_length={len(text)}\n")


def log_event(logger_, event: str, request_id: str, **fields):
    """Log estructurado de un evento: `request_id=... event=... k1=v1 ...`.

    Cada petición es trazable por su request_id a través de los eventos
    (tts_started, tts_completed, tts_failed...).
    """
    parts = [f"request_id={request_id}", f"event={event}"]
    parts += [f"{k}={v}" for k, v in fields.items()]
    logger_.info(" ".join(parts))
