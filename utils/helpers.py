#!/usr/bin/env python3
"""Funciones auxiliares y utilidades."""

import torch
import gc
import os
import time
import logging
from datetime import datetime
import soundfile as sf
from fastapi import HTTPException, Request

logger = logging.getLogger("tts")


def _resolve_device() -> str:
    """Import perezoso para evitar el ciclo: helpers -> services -> helpers."""
    from services.config_service import resolve_device
    return resolve_device()


def get_vram_available():
    """Obtener VRAM disponible en GB de la GPU seleccionada en la configuración."""
    device = _resolve_device()
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
            if torch.cuda.is_available() and idx < torch.cuda.device_count():
                return round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1)
        except (ValueError, IndexError, RuntimeError):
            pass
    return 0.0


def get_dtype():
    """Determinar dtype óptimo para el dispositivo configurado."""
    if _resolve_device().startswith("cuda:"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def clear_models(model_registry):
    """Limpia VRAM y elimina referencias de modelos correctamente."""
    logger.info("Limpiando VRAM...")

    for key in list(model_registry.keys()):
        entry = model_registry[key]
        try:
            if "model" in entry:
                del entry["model"]
            if "type" in entry:
                del entry["type"]
        except Exception as e:
            logger.warning(f"Error eliminando modelo {key}: {e}")
    model_registry.clear()
    
    # Limpiar cache de CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def validate_text(text: str, max_chars: int):
    """Validar texto para TTS."""
    if not text or not text.strip():
        raise HTTPException(400, "Texto vacío")
    if len(text) > max_chars:
        raise HTTPException(400, f"Texto demasiado largo ({len(text)} chars). Máximo: {max_chars}")



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
    # Truncar texto para logs largos
    log_text = text[:100] + "..." if len(text) > 100 else text
    with open(log_file, "a") as f:
        f.write(f"[{ts}] {ip}: {log_text}\n")


def cleanup_old_audios(audios_dir: str, max_age_days: int):
    """Eliminar audios generados hace más de max_age_days días."""
    if max_age_days <= 0:
        return
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    try:
        for fname in os.listdir(audios_dir):
            path = os.path.join(audios_dir, fname)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError as e:
                logger.warning(f"No se pudo eliminar {path}: {e}")
    except OSError as e:
        logger.warning(f"Error leyendo directorio de audios {audios_dir}: {e}")
    if removed:
        logger.info(f"🗑  Limpieza: {removed} audio(s) antiguo(s) eliminado(s) de {audios_dir}")


def save_audio(wav, sr, prefix, audios_dir: str):
    """Guardar audio WAV en disco."""
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{audios_dir}/{prefix}_{dt}.wav"
    sf.write(path, wav, sr)
    logger.info(f"💾 Audio guardado: {path}")
    return path
