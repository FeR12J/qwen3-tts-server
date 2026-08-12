#!/usr/bin/env python3
"""Servicio de transcripción de audio con Whisper (transformers).

Interfaz pública (lo que usan los endpoints):

    result = await whisper_service.transcribe(audio, language="spanish")
    whisper_service.is_loaded()
    whisper_service.get_device()
    await whisper_service.unload()

Los endpoints NO conocen detalles de transformers, ffmpeg, torch ni de la
carga del modelo: la decodificación la hace AudioService (inyectado) y la
carga del modelo es interna al servicio.
"""

import asyncio
import gc
import logging
import os
import re
from typing import Optional

import torch

from config.settings import settings
from services.audio_service import SPEECH_SAMPLE_RATE, AudioService
from services.config_service import (
    resolve_device,
    validated_device,
    validated_dtype,
)

logger = logging.getLogger("tts")

# Modos de marcas de tiempo en la transcripción
VALID_TIMESTAMP_MODES = ("off", "segment", "word")

# Whisper: cada token de tiempo representa 20 ms.
_TIME_PRECISION = 0.02

_WORD_RE = re.compile(r"\S+")


def _resolve_timestamp_mode(override: Optional[str]) -> str:
    """Resolver el modo de marcas de tiempo de una transcripción.

    Prioridad: override por petición > configuración en tiempo de ejecución
    (whisper_timestamps) > "off". Cualquier valor inválido cae a "off".
    """
    mode = (override or "").strip().lower()
    if mode in VALID_TIMESTAMP_MODES:
        return mode
    configured = getattr(settings.runtime, "whisper_timestamps", "off")
    if configured in VALID_TIMESTAMP_MODES:
        return configured
    return "off"


def _extract_segments(tokenizer, token_ids, duration_seconds: float) -> list:
    """Convertir los tokens generados con timestamps en segmentos.

    - Los ids >= ``tokenizer.timestamp_begin`` son marcas de tiempo
      (segundos = (id - timestamp_begin) * 20 ms).
    - Los ids de texto entre dos marcas pertenecen al segmento que abre la
      primera. Una marca consecutiva sin texto no crea segmento vacío.
    - El último tramo de texto (sin marca de cierre) termina en la última
      marca conocida (o en la duración del audio si no hubo ninguna).

    Devuelve [{"start": s, "end": e, "text": t}, ...] con s/e en segundos.
    """
    timestamp_begin = tokenizer.timestamp_begin
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)

    segments = []
    seg_start = None
    last_time = None
    text_ids = []

    for tid in ids:
        if tid >= timestamp_begin:
            t = round((tid - timestamp_begin) * _TIME_PRECISION, 2)
            last_time = t
            if seg_start is None:
                seg_start = t
            elif text_ids:
                text = tokenizer.decode(text_ids, skip_special_tokens=True).strip()
                if text:
                    segments.append({"start": seg_start, "end": t, "text": text})
                seg_start = t
                text_ids = []
            else:
                seg_start = t
        else:
            text_ids.append(tid)

    if text_ids:
        text = tokenizer.decode(text_ids, skip_special_tokens=True).strip()
        if text:
            segments.append({
                "start": seg_start if seg_start is not None else 0.0,
                "end": last_time if last_time is not None else round(float(duration_seconds), 2),
                "text": text,
            })
    return segments


def _extract_words(segments: list) -> list:
    """Palabras con marcas de tiempo por interpolación proporcional.

    Whisper no emite un token por palabra: dentro de un segmento la posición
    de cada palabra se estima proporcionalmente a su desplazamiento de
    caracteres entre el inicio y el fin del segmento. Para transcripción los
    segmentos son cortos (pocas palabras), por lo que la aproximación es
    razonable y no requiere atención cruzada.
    """
    words = []
    for seg in segments:
        text = seg["text"]
        if not text:
            continue
        span = seg["end"] - seg["start"]
        total = len(text)
        if span < 0 or total == 0:
            continue
        for match in _WORD_RE.finditer(text):
            words.append({
                "word": match.group(),
                "start": round(seg["start"] + span * match.start() / total, 2),
                "end": round(seg["start"] + span * match.end() / total, 2),
            })
    return words


class WhisperService:
    """Transcripción de audio con Whisper (transformers).

    Encapsula la decodificación del audio (AudioService), la carga del
    modelo, torch y el resto de detalles internos. ``transcribe`` es
    asíncrona y no bloquea el event loop.
    """

    def __init__(self, audio_service: AudioService, metrics=None):
        self._audio = audio_service
        self._metrics = metrics
        self._model = None
        self._processor = None
        self._model_name = None

    # -- Estado ------------------------------------------------------------

    def is_loaded(self) -> bool:
        """¿Está cargado el modelo Whisper?"""
        return self._model is not None

    def get_device(self) -> str:
        """Dispositivo del modelo (o el que se usaría si no está cargado)."""
        if self._model is not None:
            return str(self._model.device)
        return resolve_device()

    def status(self) -> dict:
        """Estado del servicio para los endpoints de status."""
        return {
            "model_loaded": self.is_loaded(),
            "model": settings.whisper.whisper_model,
            "device": self.get_device(),
            "timestamps": _resolve_timestamp_mode(None),
        }

    # -- Ciclo de vida del modelo ------------------------------------------

    def unload(self) -> None:
        """Liberar el modelo Whisper de memoria (bloqueante)."""
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Modelo Whisper descargado")

    def unload_if_loaded(self) -> bool:
        """Descargar Whisper solo si está cargado (no-op en caso contrario).

        Devuelve True si se descargó algo. Sirve para liberar VRAM antes de
        generar TTS en GPUs pequeñas.
        """
        if self._model is None:
            return False
        self.unload()
        return True

    # -- Transcripción -----------------------------------------------------

    async def transcribe(self, audio, language: Optional[str] = None,
                         task: str = "transcribe",
                         timestamps: Optional[str] = None) -> dict:
        """Transcribir audio (bytes, ruta o file-like) a texto.

        ``timestamps`` sobreescribe el modo configurado (whisper_timestamps):
        "off" (solo texto), "segment" (segmentos con inicio/fin) o "word"
        (segmentos + palabras).

        Devuelve {text, language, duration_seconds, model, device,
        timestamps} y, según el modo, segments[] y words[]. La inferencia se
        ejecuta en un hilo (no bloquea el event loop); la carga del modelo
        es lazy y automática.
        """
        if self._metrics is not None:
            self._metrics.whisper_requested()
        return await asyncio.to_thread(
            self._transcribe_sync, audio, language, task, timestamps
        )

    def _transcribe_sync(self, audio, language, task,
                         timestamps: Optional[str] = None) -> dict:
        """Implementación bloqueante (se ejecuta en un hilo)."""
        wav, sr = self._audio.load(audio, target_sr=SPEECH_SAMPLE_RATE)
        duration = float(wav.shape[0]) / sr
        self._ensure_loaded()

        device = self._model.device
        dtype = self._model.dtype
        language = (language or "").strip().lower() or "auto"
        mode = _resolve_timestamp_mode(timestamps)
        log = logging.getLogger("tts")
        if mode != "off":
            log.info(f"whisper: marcas de tiempo habilitadas (modo: {mode})")

        inputs = self._processor(wav, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        forced_language = language if language != "auto" else None
        forced_decoder_ids = None
        if forced_language is not None:
            try:
                forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                    language=forced_language, task=task
                )
            except ValueError as e:
                raise ValueError(str(e))

        generate_kwargs = {}
        if mode != "off":
            generate_kwargs["return_timestamps"] = True

        with torch.inference_mode():
            try:
                if forced_decoder_ids is not None:
                    generated = self._model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        **generate_kwargs,
                    )
                    detected_language = forced_language
                else:
                    generated = self._model.generate(
                        input_features, **generate_kwargs
                    )
                    full = self._processor.tokenizer.decode(generated[0].tolist())
                    match = re.search(r"<\|([a-z]{2,3})\|>", full)
                    detected_language = match.group(1) if match else "auto"
            except torch.cuda.OutOfMemoryError as e:
                # CUDA OOM: limpiar referencias temporales y cache, y elevar un
                # error controlado para que la capa HTTP no filtre detalles.
                from services.model_manager import GPUOutOfMemoryError
                logger.error(f"CUDA OOM en transcripción Whisper: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise GPUOutOfMemoryError() from e

        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        result = {
            "text": text,
            "language": detected_language,
            "duration_seconds": round(duration, 2),
            "model": self._model_name,
            "device": str(device),
            "timestamps": mode,
        }
        if mode in ("segment", "word"):
            segments = _extract_segments(
                self._processor.tokenizer, generated[0], duration
            )
            result["segments"] = segments
        if mode == "word":
            result["words"] = _extract_words(result["segments"])
        return result

    def _ensure_loaded(self) -> None:
        """Cargar el modelo Whisper (bloqueante, ejecutar en hilo)."""
        if self._model is not None:
            return

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_name = settings.whisper.whisper_model
        model_path = os.path.join(settings.paths.models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo Whisper no encontrado en {model_path}")

        device = validated_device()
        dtype = validated_dtype()
        logger.info(f"Cargando Whisper desde {model_path} (device: {device}, dtype: {dtype})")

        self._processor = WhisperProcessor.from_pretrained(model_path)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self._model_name = model_name
        logger.info("Whisper cargado correctamente")


# -- Singleton del módulo ---------------------------------------------------
#
# Compatibilidad con gpu_management y las rutas (que importan el módulo).
# El AudioService se inyecta en build_context() vía configure().

_instance: Optional[WhisperService] = None


def _get() -> WhisperService:
    global _instance
    if _instance is None:
        _instance = WhisperService(audio_service=None)
    return _instance


def configure(audio_service: AudioService, metrics=None) -> WhisperService:
    """Inyectar el AudioService (y métricas opcionales) en el singleton.

    Devuelve la instancia configurada (la misma que usan las rutas).
    """
    global _instance
    _instance = WhisperService(audio_service, metrics=metrics)
    return _instance


def is_loaded() -> bool:
    return _get().is_loaded()


def get_device() -> str:
    return _get().get_device()


def status() -> dict:
    return _get().status()


async def transcribe(audio, language: Optional[str] = None,
                     task: str = "transcribe",
                     timestamps: Optional[str] = None) -> dict:
    """Transcribir audio a texto (interfaz asíncrona sencilla)."""
    return await _get().transcribe(audio, language, task, timestamps)


async def unload() -> None:
    """Descargar el modelo Whisper (asíncrono: libera el hilo)."""
    await asyncio.to_thread(_get().unload)


def unload_if_loaded() -> bool:
    """Descargar solo si está cargado (síncrono, para gpu_management)."""
    return _get().unload_if_loaded()
