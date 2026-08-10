#!/usr/bin/env python3
"""Persistencia de la configuración en tiempo de ejecución."""

import os
import json
import logging

from utils.paths import DATA_DIR

logger = logging.getLogger("tts")

RUNTIME_FILE = os.path.join(DATA_DIR, "runtime.json")


def load_runtime_file() -> dict:
    """Cargar la configuración persistida desde disco (si existe)."""
    try:
        if os.path.exists(RUNTIME_FILE):
            with open(RUNTIME_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.warning(f"No se pudo cargar la configuración de {RUNTIME_FILE}: {e}")
    return {}


def save_runtime_file(runtime: dict):
    """Persistir la configuración en disco."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(RUNTIME_FILE, "w", encoding="utf-8") as f:
            json.dump(runtime, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"No se pudo guardar la configuración de {RUNTIME_FILE}: {e}")
