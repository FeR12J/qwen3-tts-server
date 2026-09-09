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
import threading
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

# Whisper: el encoder produce 100 features por segundo; con más de 3000
# features (~30 s) transformers activa el algoritmo secuencial long-form,
# que exige return_timestamps=True (con False lanza ValueError).
_LONG_FORM_FRAMES = 3000


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


def _timestamp_begin(tokenizer) -> Optional[int]:
    """Primer id de token de tiempo (compatible transformers >= 4.50).

    ``WhisperTokenizerFast.timestamp_begin`` (atributo) se eliminó en
    transformers moderno y se sustituyó por el método ``timestamp_ids()``.
    """
    ids = getattr(tokenizer, "timestamp_ids", None)
    if ids is not None:
        ids = ids() if callable(ids) else ids
        if ids:
            return ids[0]
    return getattr(tokenizer, "timestamp_begin", None)


def _extract_segments(tokenizer, token_ids, duration_seconds: float) -> list:
    """Convertir los tokens generados con timestamps en segmentos.

    - Los ids >= ``timestamp_begin`` son marcas de tiempo
      (segundos = (id - timestamp_begin) * 20 ms).
    - Los ids de texto entre dos marcas pertenecen al segmento que abre la
      primera. Una marca consecutiva sin texto no crea segmento vacío.
    - El último tramo de texto (sin marca de cierre) termina en la última
      marca conocida (o en la duración del audio si no hubo ninguna).

    Devuelve [{"start": s, "end": e, "text": t}, ...] con s/e en segundos.
    """
    timestamp_begin = _timestamp_begin(tokenizer)
    if timestamp_begin is None:
        raise ValueError("El tokenizer no expone marcas de tiempo")
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


def _sequences_from_generate(outputs) -> object:
    """Secuencia(s) de ids generadas: tensor plano o ``dict["sequences"]``.

    ``generate(return_segments=True)`` devuelve un dict con "sequences" y
    "segments"; sin ``return_segments`` devuelve directamente el tensor.
    """
    if isinstance(outputs, dict) and "sequences" in outputs:
        return outputs["sequences"]
    return outputs


def _segments_from_generate(outputs, tokenizer):
    """Segmentos calculados por el modelo (long-form, ``return_segments``).

    transformers 4.57.3 devuelve la segmentación exacta del algoritmo
    secuencial de Whisper (más fiable que reconstruirla del flujo de tokens):
    ``segments`` es una lista por elemento del batch; cada segmento incluye
    ``start``/``end`` (segundos, posiblemente tensores) y ``tokens``.

    Devuelve una lista [{start, end, text}] o None si el output no incluye
    segmentos (el llamador entonces usa la extracción manual).
    """
    if not isinstance(outputs, dict) or not outputs.get("segments"):
        return None
    batch = outputs["segments"]
    if not batch:
        return []
    out = []
    for seg in batch[0]:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        tokens = seg.get("tokens")
        if hasattr(start, "__float__"):
            start = float(start)
        if hasattr(end, "__float__"):
            end = float(end)
        token_ids = tokens.tolist() if hasattr(tokens, "tolist") else tokens
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if not text:
            continue
        out.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text,
        })
    return out


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
        # La carga lazy ocurre en hilos (asyncio.to_thread): un lock de hilo
        # evita que dos transcripciones concurrentes (N > 1) carguen dos
        # instancias del modelo a la vez.
        self._load_lock = threading.Lock()

    # -- Estado ------------------------------------------------------------

    def is_loaded(self) -> bool:
        """¿Está cargado el modelo Whisper?"""
        return self._model is not None

    def get_device(self) -> str:
        """Dispositivo del modelo (o el que se usaría si no está cargado)."""
        if self._model is not None:
            return str(self._model.device)
        return resolve_device()

    def _configured_model_name(self) -> str:
        """Nombre del modelo Whisper configurado.

        El runtime (editable desde el panel) tiene prioridad sobre el grupo
        estático ``whisper.whisper_model``; si por cualquier razón el runtime
        no lo tuviera, se cae al default estático.
        """
        return getattr(settings.runtime, "whisper_model", None) or settings.whisper.whisper_model

    def status(self) -> dict:
        """Estado del servicio para los endpoints de status."""
        return {
            "model_loaded": self.is_loaded(),
            "model": self._configured_model_name(),
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

    async def load(self) -> None:
        """Cargar el modelo Whisper de forma explícita (bloqueante en hilo).

        Equivale a la carga lazy de la primera transcripción: garantiza que
        el modelo cargado es el configurado y no bloquea el event loop.
        """
        await asyncio.to_thread(self._ensure_loaded)

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
        """Implementación bloqueante (se ejecuta en un hilo).

        Time sin truncar (long-form): para audio > 30 s se pasan todas las
        features a generate() para que transformers ejecute el algoritmo
        secuencial de Whisper (ventanas de 30 s encadenadas). Ese algoritmo
        exige ``return_timestamps=True``, así que en modo "off" los timestamps
        se generan internamente y se eliminan al decodificar el texto final.
        """
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

        # NUNCA truncar: un audio largo debe llegar entero a generate() y
        # transformers decide short/long-form por el número de features. El
        # attention_mask se pasa para que el encoder ignore el padding.
        inputs = self._processor(
            wav,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)

        is_long = input_features.shape[-1] > _LONG_FORM_FRAMES
        if is_long:
            log.info(
                f"whisper: transcripción long-form "
                f"({duration:.1f}s > 30 s, {input_features.shape[-1]} features)"
            )

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
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        # Long-form exige return_timestamps=True (transformers lanza
        # ValueError con False). También se usa el algoritmo secuencial para
        # pedir segmentos/palabras. En modo "off" los timestamps son un medio
        # (necesario para audios largos): se eliminan en el texto final.
        generate_kwargs["return_timestamps"] = (is_long or mode != "off")
        if mode in ("segment", "word"):
            # Segmentos calculados por el propio modelo (land-mark de
            # precisión): preferibles a reconstruirlos del flujo de tokens.
            generate_kwargs["return_segments"] = True

        with torch.inference_mode():
            try:
                if forced_decoder_ids is not None:
                    outputs = self._model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        **generate_kwargs,
                    )
                    detected_language = forced_language
                else:
                    outputs = self._model.generate(
                        input_features, **generate_kwargs
                    )
                    sequences = _sequences_from_generate(outputs)
                    full = self._processor.tokenizer.decode(
                        sequences[0].tolist()
                        if hasattr(sequences[0], "tolist")
                        else sequences[0]
                    )
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

        # El texto final nunca contiene marcas de tiempo: skip_special_tokens
        # elimina los tokens de tiempo, incluidos los usados internamente en
        # long-form con modo "off".
        sequences = _sequences_from_generate(outputs)
        text = self._processor.batch_decode(
            sequences, skip_special_tokens=True
        )[0].strip()
        result = {
            "text": text,
            "language": detected_language,
            "duration_seconds": round(duration, 2),
            "model": self._model_name,
            "device": str(device),
            "timestamps": mode,
        }
        if mode in ("segment", "word"):
            segments = _segments_from_generate(
                outputs, self._processor.tokenizer
            )
            if segments is None:
                segments = _extract_segments(
                    self._processor.tokenizer, sequences[0], duration
                )
            result["segments"] = segments
        if mode == "word":
            result["words"] = _extract_words(result["segments"])
        return result

    def _ensure_loaded(self) -> None:
        """Cargar el modelo Whisper (bloqueante, ejecutar en hilo).

        Garantiza que el modelo cargado sea el configurado: si ya está
        cargado pero el configurado ha cambiado (p.ej. desde el panel), se
        descarga el anterior y se carga el nuevo. Así el cambio de modelo se
        aplica de forma lazy en la siguiente transcripción, incluso si no se
        descargó explícitamente al guardar la configuración.
        """
        if self._model is not None and self._model_name == self._configured_model_name():
            return
        with self._load_lock:
            if self._model is not None:
                if self._model_name == self._configured_model_name():
                    return
                logger.info(
                    f"Cambiando modelo Whisper: {self._model_name} -> "
                    f"{self._configured_model_name()}"
                )
                self.unload()
            self._load_model()

    def _load_model(self) -> None:
        """Carga real del modelo (bajo _load_lock)."""
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_name = self._configured_model_name()
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


async def load() -> None:
    """Cargar el modelo Whisper de forma explícita (asíncrono)."""
    await _get().load()


def unload_if_loaded() -> bool:
    """Descargar solo si está cargado (síncrono, para gpu_management)."""
    return _get().unload_if_loaded()
