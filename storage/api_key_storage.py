#!/usr/bin/env python3
"""Persistencia de las claves API."""

import os
import json
import logging

from config.settings import settings

logger = logging.getLogger("tts")

APIKEYS_FILE = settings.auth.keys_file


def load_keys_file() -> list:
    """Cargar las claves API desde disco (si existe)."""
    try:
        if os.path.exists(APIKEYS_FILE):
            with open(APIKEYS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning(f"No se pudieron cargar las claves API de {APIKEYS_FILE}: {e}")
    return []


def save_keys_file(keys: list):
    """Persistir las claves API en disco."""
    try:
        os.makedirs(os.path.dirname(APIKEYS_FILE), exist_ok=True)
        with open(APIKEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"No se pudieron guardar las claves API en {APIKEYS_FILE}: {e}")
