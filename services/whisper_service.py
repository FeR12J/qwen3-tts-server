#!/usr/bin/env python3
"""Servicio de transcripción de audio con Whisper (transformers)."""

import os
import re
import io
import gc
import wave
import shutil
import logging
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch

from config.settings import CONFIG
from services.config_service import resolve_device
from utils.gpu import get_dtype

logger = logging.getLogger("tts")

_model = None
_processor = None
_model_name = None


def is_loaded() -> bool:
    """¿Está cargado el modelo Whisper?"""
    return _model is not None


def get_device() -> str:
    """Dispositivo en el que está el modelo (o el que se usaría)."""
    if _model is not None:
        return str(_model.device)
    return resolve_device()


def _ensure_loaded():
    """Cargar el modelo Whisper (bloqueante, ejecutar en hilo)."""
    global _model, _processor, _model_name
    if _model is not None:
        return

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_name = CONFIG.get("whisper_model", "whisper-large-v3")
    model_path = os.path.join(CONFIG["local_models_dir"], model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo Whisper no encontrado en {model_path}")

    device = resolve_device()
    dtype = get_dtype()
    logger.info(f"Cargando Whisper desde {model_path} (device: {device}, dtype: {dtype})")

    _processor = WhisperProcessor.from_pretrained(model_path)
    _model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _model_name = model_name
    logger.info("Whisper cargado correctamente")


def unload():
    """Liberar el modelo Whisper de memoria."""
    global _model, _processor
    _model = None
    _processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Modelo Whisper descargado")


def unload_if_loaded() -> bool:
    """Descargar Whisper solo si está cargado (no-op en caso contrario).

    Devuelve True si se descargó algo. Sirve para liberar VRAM antes de
    generar TTS en GPUs pequeñas.
    """
    if _model is None:
        return False
    unload()
    return True


def _decode_audio(audio_bytes: bytes) -> tuple:
    """Decodificar audio a mono float32 a 16 kHz (wav, mp3, flac, ogg, m4a...)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        try:
            return _decode_with_ffmpeg(audio_bytes, ffmpeg)
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"ffmpeg falló ({e}); probando soundfile...")
    try:
        return _decode_with_soundfile(audio_bytes)
    except Exception as e:
        raise ValueError(f"No se pudo decodificar el audio ({e}). Formatos soportados: wav, mp3, flac, ogg, m4a")


def _decode_with_ffmpeg(audio_bytes: bytes, ffmpeg: str) -> tuple:
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", tmp.name, "-f", "wav", "-ar", "16000", "-ac", "1", "-"],
            capture_output=True,
        )
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(proc.stderr.decode(errors="ignore").strip() or "error de ffmpeg")
    wav = wave.open(io.BytesIO(proc.stdout), "rb")
    audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("float32") / 32768.0
    return audio, float(audio.shape[0]) / 16000.0


def _decode_with_soundfile(audio_bytes: bytes) -> tuple:
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 16000, sr).astype("float32")
    return audio, float(audio.shape[0]) / 16000.0


def transcribe(audio_bytes: bytes, language: Optional[str] = None) -> dict:
    """Transcribir audio a texto (bloqueante, ejecutar en hilo)."""
    audio, duration = _decode_audio(audio_bytes)
    _ensure_loaded()

    device = _model.device
    dtype = _model.dtype
    language = (language or "").strip().lower() or "auto"

    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=dtype)

    forced_language = language if language != "auto" else None
    forced_decoder_ids = None
    if forced_language is not None:
        try:
            forced_decoder_ids = _processor.get_decoder_prompt_ids(language=forced_language, task="transcribe")
        except ValueError as e:
            raise ValueError(str(e))

    with torch.inference_mode():
        try:
            if forced_decoder_ids is not None:
                generated = _model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                detected_language = forced_language
            else:
                generated = _model.generate(input_features)
                full = _processor.tokenizer.decode(generated[0].tolist())
                match = re.search(r"<\|([a-z]{2,3})\|>", full)
                detected_language = match.group(1) if match else "auto"
        except torch.cuda.OutOfMemoryError as e:
            # CUDA OOM: limpiar referencias temporales y cache, y elevar un
            # error controlado para que la capa HTTP no filtre detalles.
            from services.model_manager import GPUOutOfMemoryError
            logger.error(f"CUDA OOM en transcripción Whisper: {e}")
            input_features = None
            generated = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise GPUOutOfMemoryError() from e

    text = _processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return {
        "text": text,
        "language": detected_language,
        "duration_seconds": round(duration, 2),
        "model": _model_name,
        "device": str(device),
    }
