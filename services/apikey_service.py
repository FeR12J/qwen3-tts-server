#!/usr/bin/env python3
"""Gestión de claves API del servidor TTS."""

import hashlib
import secrets
import logging
from datetime import datetime

from storage.api_key_storage import load_keys_file, save_keys_file
from services.config_service import get_runtime_config

logger = logging.getLogger("tts")

_keys = None


def _load():
    global _keys
    _keys = load_keys_file()


def _save():
    save_keys_file(_keys)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def global_enabled() -> bool:
    """¿Está activada la exigencia de clave API para el servicio?"""
    return bool(get_runtime_config().get("api_keys_enabled"))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_key(name: str) -> dict:
    """Crear una clave API. La clave completa solo se devuelve una vez."""
    if not name:
        raise ValueError("El nombre de la clave no puede estar vacío")
    _load()
    key = "qt-" + secrets.token_hex(16)
    entry = {
        "id": "key_" + secrets.token_hex(8),
        "name": name,
        "key_hash": _hash(key),
        "prefix": key[:8],
        "enabled": True,
        "created_at": _now(),
        "last_used_at": None,
    }
    _keys.append(entry)
    _save()
    logger.info(f"Clave API creada: {name}")
    return {"id": entry["id"], "name": name, "key": key}


def list_keys() -> list:
    """Listar claves sin exponer el hash ni la clave completa."""
    _load()
    return [
        {
            "id": k["id"],
            "name": k["name"],
            "masked": k.get("prefix", k["key_hash"][:8]) + "...",
            "enabled": k["enabled"],
            "created_at": k["created_at"],
            "last_used_at": k.get("last_used_at"),
        }
        for k in _keys
    ]


def count_keys() -> int:
    """Número de claves existentes (para el modo bootstrap de admin)."""
    _load()
    return len(_keys)


def delete_key(key_id: str):
    _load()
    for i, k in enumerate(_keys):
        if k["id"] == key_id:
            del _keys[i]
            _save()
            logger.info(f"Clave API eliminada: {k['name']}")
            return
    raise KeyError(f"No existe la clave {key_id}")


def toggle_key(key_id: str) -> dict:
    """Activar/desactivar una clave individual."""
    _load()
    for k in _keys:
        if k["id"] == key_id:
            k["enabled"] = not k["enabled"]
            _save()
            logger.info(f"Clave API '{k['name']}' {'activada' if k['enabled'] else 'desactivada'}")
            return {"id": k["id"], "enabled": k["enabled"]}
    raise KeyError(f"No existe la clave {key_id}")


def verify_key(key: str) -> bool:
    """Verificar una clave API y registrar el último uso."""
    if not key:
        return False
    _load()
    digest = _hash(key)
    for k in _keys:
        if k["key_hash"] == digest:
            if not k["enabled"]:
                return False
            k["last_used_at"] = _now()
            _save()
            return True
    return False
