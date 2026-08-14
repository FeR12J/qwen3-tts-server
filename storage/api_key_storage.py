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
    """Persistir las claves API en disco (escritura atómica).

    Escribe a un archivo temporal y lo mueve con ``os.replace``: un crash a
    mitad de escritura no puede corromper el archivo (y un archivo corrupto
    reactivaría el modo bootstrap de admin). Permisos 0600: los hashes de
    claves no deben ser legibles por otros usuarios.
    """
    try:
        os.makedirs(os.path.dirname(APIKEYS_FILE), exist_ok=True)
        tmp_path = APIKEYS_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, APIKEYS_FILE)
        os.chmod(APIKEYS_FILE, 0o600)
    except Exception as e:
        logger.warning(f"No se pudieron guardar las claves API en {APIKEYS_FILE}: {e}")
