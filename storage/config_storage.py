#!/usr/bin/env python3
"""Persistencia de la configuración en tiempo de ejecución."""

import os
import json
import logging

from config.settings import settings

logger = logging.getLogger("tts")

RUNTIME_FILE = os.path.join(settings.paths.data_dir, "runtime.json")


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
    """Persistir la configuración en disco (escritura atómica).

    Escribe a un archivo temporal y lo mueve con ``os.replace``: un crash a
    mitad de escritura no puede corromper runtime.json (y un archivo
    corrupto haría perder la configuración persistida).
    """
    try:
        os.makedirs(settings.paths.data_dir, exist_ok=True)
        tmp_path = RUNTIME_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(runtime, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, RUNTIME_FILE)
    except Exception as e:
        logger.warning(f"No se pudo guardar la configuración de {RUNTIME_FILE}: {e}")
