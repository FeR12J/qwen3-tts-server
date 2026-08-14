#!/usr/bin/env python3
"""Servicio de generación TTS (capa de orquestación).

Único punto de síntesis del servidor: todos los endpoints TTS terminan en
``synthesize(request)``. Aquí se valida la petición según el tipo de modelo
activo (antes de consumir GPU), se resuelven las referencias de voz y se
despacha a la librería Qwen3-TTS a través de ModelManager (este servicio
nunca manipula la instancia del modelo directamente).
"""

import os
import re
import uuid
import time
import logging

from dataclasses import dataclass

from schemas.tts import TTSRequest
from security.validation import is_safe_voice_ref, resolve_contained_path
from services.audio_service import AudioValidationError
from services.config_service import get_runtime_config, get_limits
from services.errors import APIError, ModelLoadingError, ModelNotLoadedError
from services.model_manager import ModelInfo
from utils.chunker import TextChunker, TextChunkerError
from utils.logging import log_event

logger = logging.getLogger("tts")

# Ritmo medio de habla usado para estimar la duración del audio generado
# a partir de la longitud del texto (antes de usar GPU).
CHARS_PER_SECOND = 16.0


class TTSValidationError(APIError):
    """Petición TTS inválida (400). Formato:
    {"error": {"code": "INVALID_TTS_REQUEST", "message": "..."}}."""

    def __init__(self, message: str, code: str = "INVALID_TTS_REQUEST"):
        super().__init__(code, message, 400)


@dataclass
class SynthesisResult:
    """Resultado de synthesize: audio ya codificado y metadatos."""
    audio: bytes
    sample_rate: int
    model_id: str
    model_type: str


@dataclass
class StreamPlan:
    """Plan de streaming validado (fragmentos + modelo resuelto).

    Se construye antes de abrir la respuesta HTTP para poder devolver
    errores 400 con un status HTTP real.
    """
    sentences: list
    info: ModelInfo


class TTSService:
    """Orquestación de la generación de audio TTS."""

    def __init__(self, config, queue, model_manager, voice_manager, audio_service, metrics=None):
        self._config = config
        self._queue = queue
        self._models = model_manager
        self._voices = voice_manager
        self._audio = audio_service
        self._metrics = metrics

    # -- Referencias de voz -------------------------------------------------

    def _check_ref_audio_size(self, wav_path: str):
        """Validar el audio de referencia (tamaño, duración, sample rate,
        canales y contenido completo) antes de usar GPU. Delega en AudioService."""
        limits = get_limits()
        try:
            self._audio.validate(
                wav_path,
                max_bytes=limits.max_reference_audio_mb * 1024 * 1024,
                max_duration=limits.max_reference_duration_seconds,
                min_sample_rate=limits.min_sample_rate,
                max_sample_rate=limits.max_sample_rate,
                max_channels=limits.max_channels,
                decode=True,
            )
        except AudioValidationError as e:
            raise TTSValidationError(str(e))

    def _local_voice_ref(self, name: str):
        """(wav, text) de una voz local por id o nombre, o None si no existe."""
        if not name:
            return None
        ref = self._voices.get_reference(name)
        if ref is None:
            return None
        wav, txt = ref
        self._check_ref_audio_size(wav)
        with open(txt, "r", encoding="utf-8") as f:
            return wav, f.read().strip()

    def _default_ref_voice(self):
        """Voz local de referencia por defecto (def_voice o la primera válida)."""
        rc = get_runtime_config()
        candidates = [rc.get("def_voice", "")]
        try:
            candidates += [item["id"] for item in self._voices.list() if item.get("valid")]
        except Exception:
            pass
        for name in candidates:
            ref = self._local_voice_ref(name)
            if ref:
                return ref
        return None

    # -- Validación previa a GPU -------------------------------------------

    def _validate_custom_voice(self, request: TTSRequest, info) -> str:
        """Validar y resolver speaker para modelos CustomVoice."""
        rc = get_runtime_config()
        speaker = request.speaker or rc["def_voice"]
        if not speaker:
            raise TTSValidationError("speaker es obligatorio para CustomVoice.")
        if info.supported_speakers is not None and speaker.lower() not in {
            s.lower() for s in info.supported_speakers
        }:
            raise TTSValidationError(
                f"speaker '{speaker}' no válido. Válidos: {', '.join(info.supported_speakers)}"
            )
        return speaker

    def _validate_voice_design(self, request: TTSRequest) -> str:
        """Validar y resolver la descripción de voz para VoiceDesign."""
        rc = get_runtime_config()
        description = (
            request.voice_description
            or request.instruct
            or rc.get("def_instruct")
            or ""
        ).strip()
        if not description:
            raise TTSValidationError(
                "voice_description es obligatorio para VoiceDesign."
            )
        return description

    def _resolve_reference_audio(self, raw: str) -> str:
        """Resolver reference_audio conteniéndola al directorio del proyecto.

        El campo es una ruta de servidor controlada por el cliente: sin
        contención permitiría forzar la lectura/decodificación de cualquier
        archivo del sistema. Se resuelve la ruta real (symlinks incluidos)
        y se exige que quede dentro de settings.paths.base_dir.
        """
        try:
            return resolve_contained_path(
                raw, self._config.paths.base_dir, "reference_audio"
            )
        except ValueError as e:
            raise TTSValidationError(str(e))

    def _validate_base(self, request: TTSRequest):
        """Validar y resolver la referencia de clonación para modelos Base.

        Orden: reference_audio -> voice local -> voz cargada -> voz por defecto.
        Devuelve None si se usará voice_clone_prompt, o (wav, text).

        El tamaño/dureación del audio de referencia ya se validó sin GPU en
        _validate_input_limits(): aquí solo se resuelve la referencia.
        """
        if request.reference_audio:
            ref_path = self._resolve_reference_audio(request.reference_audio)
            if not os.path.exists(ref_path):
                raise TTSValidationError(
                    f"reference_audio no existe: {request.reference_audio}"
                )
            ref_text = (request.reference_text or "").strip()
            if not ref_text:
                raise TTSValidationError(
                    "reference_text es obligatorio cuando se proporciona reference_audio."
                )
            return ref_path, ref_text

        if request.voice:
            voice = request.voice or ""
            if not is_safe_voice_ref(voice):
                raise TTSValidationError(
                    "'voice' debe ser el id o nombre de una voz local "
                    "(ej: 'voice_7f32a1'), no una ruta de archivo"
                )
            ref = self._local_voice_ref(request.voice)
            if ref is None:
                raise TTSValidationError(f"Voz local '{request.voice}' no encontrada.")
            return ref

        if self._voices.clone_prompt:
            return None  # usa la voz cargada (voice_clone_prompt)

        ref = self._default_ref_voice()
        if ref is None:
            raise TTSValidationError(
                "reference_audio es obligatorio para voice cloning: el modelo Base "
                "requiere una voz de referencia (parámetro reference_audio o una "
                "voz local cargada)."
            )
        return ref

    def _validate_language(self, request: TTSRequest, info) -> str:
        rc = get_runtime_config()
        lang = request.language or rc["def_language"]
        if info.supported_languages is not None and lang.lower() not in {
            item.lower() for item in info.supported_languages
        }:
            raise TTSValidationError(
                f"language '{lang}' no soportado. Soportados: {', '.join(info.supported_languages)}"
            )
        return lang

    # -- Síntesis -----------------------------------------------------------

    def _require_text(self, request: TTSRequest) -> str:
        """Validar el texto de entrada contra los límites (antes de GPU).

        Aplica limits.max_text_characters (límite absoluto de entrada) y una
        HEURÍSTICA de duración estimada (limits.max_estimated_audio_duration_seconds
        a partir de la longitud del texto, ritmo medio de habla ~16 caracteres/segundo).
        """
        text = (request.text or request.input or "").strip()
        if not text:
            raise TTSValidationError("Campo 'text' o 'input' requerido")

        limits = get_limits()
        if len(text) > limits.max_text_characters:
            raise TTSValidationError(
                f"Texto demasiado largo: {len(text)} caracteres "
                f"(máximo configurado: {limits.max_text_characters})."
            )

        estimated = len(text) / CHARS_PER_SECOND
        if estimated > limits.max_estimated_audio_duration_seconds:
            raise TTSValidationError(
                f"Audio estimado demasiado largo: ~{estimated:.1f}s "
                f"(máximo configurado: {limits.max_estimated_audio_duration_seconds}s). "
                "Reduce la longitud del texto."
            )
        return text

    def _validate_input_limits(self, request: TTSRequest):
        """Límites de entrada independientes del modelo (antes de GPU/lock).

        Comprueba el tamaño del audio de referencia indicado en la petición
        (reference_audio o voz local) contra limits.max_reference_audio_mb.
        """
        if request.reference_audio:
            # Si no existe o escapa del directorio del proyecto, el error lo
            # dará la validación de la referencia (_validate_base).
            try:
                ref_path = self._resolve_reference_audio(request.reference_audio)
            except TTSValidationError:
                ref_path = None
            if ref_path is not None and os.path.exists(ref_path):
                self._check_ref_audio_size(ref_path)
        if request.voice:
            ref = self._voices.get_reference(request.voice)
            if ref:
                self._check_ref_audio_size(ref[0])

    def _make_chunker(self) -> TextChunker:
        """Chunker de textos largos según la configuración.

        - ``max_text_characters`` (runtime): límite del texto de entrada.
        - ``chunking`` (runtime): modo de división (sentence | paragraph).
        - ``max_text_chars`` (runtime): tamaño máximo de cada fragmento
          generado (editable desde el panel).
        """
        rc = get_runtime_config()
        limits = get_limits()
        return TextChunker(
            max_characters=limits.max_text_characters,
            chunking=rc["chunking"],
            chunk_size=rc["max_text_chars"],
        )

    def _chunk_text(self, text: str) -> list:
        """Dividir el texto en fragmentos, traduciendo errores a 400."""
        try:
            return self._make_chunker().chunk(text)
        except TextChunkerError as e:
            raise TTSValidationError(str(e))

    def _build_kwargs(self, request: TTSRequest, info: ModelInfo) -> dict:
        """Validar según el tipo de modelo y construir kwargs de generación.

        No toca GPU: solo resuelve/valida referencias de voz, idioma y
        parámetros según el modelo activo.
        """
        kwargs = {"language": self._validate_language(request, info)}
        if request.temperature is not None:
            # Parámetro real de generación de la librería Qwen3-TTS.
            kwargs["temperature"] = request.temperature

        model_type = info.model_type
        if model_type == "voice_design":
            kwargs["instruct"] = self._validate_voice_design(request)
        elif model_type == "base":
            ref = self._validate_base(request)
            if ref is None:
                kwargs["voice_clone_prompt"] = self._voices.clone_prompt
            else:
                kwargs["ref_audio"], kwargs["ref_text"] = ref
        else:
            kwargs["speaker"] = self._validate_custom_voice(request, info)
            instruct = (
                request.voice_description
                or request.instruct
                or get_runtime_config().get("def_instruct")
            ) or None
            if instruct:
                kwargs["instruct"] = instruct
        return kwargs

    async def _generate_one(self, request: TTSRequest, text: str, info: ModelInfo):
        """Generar audio para un texto con el modelo activo (usa GPU)."""
        kwargs = self._build_kwargs(request, info)
        kwargs["text"] = text
        model_type = info.model_type

        if model_type == "voice_design":
            logger.info(f"Usando voice design (description: {kwargs['instruct'][:60]})")
            wavs, sr = await self._models.generate_voice_design(**kwargs)
        elif model_type == "base":
            if "voice_clone_prompt" in kwargs:
                logger.info("Usando voice cloning (voz cargada)")
            else:
                logger.info(f"Usando voz de referencia: {kwargs['ref_audio']}")
            wavs, sr = await self._models.generate_voice_clone(**kwargs)
        else:
            logger.info(f"Usando voz por defecto (speaker={kwargs['speaker']})")
            wavs, sr = await self._models.generate_custom_voice(**kwargs)
        return wavs, sr

    def _encode(self, request: TTSRequest, wav, sr, request_id: str = None) -> bytes:
        """Guardar copia y codificar el audio según output_format."""
        self._audio.save(wav, sr, "tts", request_id=request_id)
        if request.output_format == "pcm":
            return self._audio.encode_pcm(wav, sr)
        return self._audio.encode_wav(wav, sr)

    @staticmethod
    def _request_id(http_request) -> str:
        """Id de trazabilidad de la petición (x-request-id o uno generado).

        Se usa en los nombres de los audios guardados (AudioService.save)
        y en los logs estructurados. La cabecera es controlada por el
        cliente: se sanitizan los caracteres de control (evita inyección de
        líneas en los logs) y se limita la longitud.
        """
        if http_request is None:
            return uuid.uuid4().hex
        raw = http_request.headers.get("x-request-id") or ""
        sanitized = re.sub(r"[\x00-\x1f\x7f]+", "", raw)[:64]
        return sanitized or uuid.uuid4().hex

    async def synthesize(self, request: TTSRequest, http_request=None) -> SynthesisResult:
        """Pipeline completo de síntesis: validación + generación + codificación.

        Único punto de generación no-streaming del servidor; todos los
        endpoints TTS pasan por aquí. Ejecuta dentro de queue.inference_lock().

        Textos largos (hasta limits.max_text_characters) se dividen en fragmentos
        (TextChunker) y se generan secuencialmente, concatenando el audio.
        """
        from services import whisper_service
        from services.gpu_management import prepare_for_tts

        request_id = self._request_id(http_request)
        t_request = time.perf_counter()
        text = self._require_text(request)
        self._validate_input_limits(request)
        rc = get_runtime_config()

        t_lock = time.perf_counter()
        async with self._queue.inference_lock():
            # Tiempo de espera en la cola de inferencia (hasta el turno de GPU)
            queue_wait_ms = int((time.perf_counter() - t_lock) * 1000)
            # Regla arquitectónica: la validación cara (texto, audio de
            # referencia) ya ocurrió ANTES del lock (_require_text /
            # _validate_input_limits). Dentro del lock solo queda: resolver
            # el modelo (puede cargarlo = GPU), validaciones baratas de
            # metadatos (idioma/speaker) y la generación.
            info = await self._resolve_model(request)
            self._build_kwargs(request, info)

            await prepare_for_tts(self._models, self._voices, whisper_service,
                                  queue=self._queue)

            if rc.get("log_requests", True) and http_request is not None and self._metrics:
                self._metrics.log_request(http_request, text)

            chunks = self._chunk_text(text)
            if not chunks:
                raise TTSValidationError(
                    "No se pudo dividir el texto en fragmentos para generar."
                )

            log_event(
                logger, "tts_started", request_id,
                model=info.model_id, model_type=info.model_type,
                text_length=len(text), chunks=len(chunks),
                queue_wait_ms=queue_wait_ms,
            )
            if self._metrics:
                self._metrics.tts_started()
                self._metrics.queue_waited(queue_wait_ms)
            started = time.perf_counter()
            try:
                if len(chunks) == 1:
                    wavs, sr = await self._generate_one(request, chunks[0], info)
                    wav = wavs[0]
                else:
                    logger.info(
                        f"Texto largo: {len(text)} caracteres -> {len(chunks)} fragmentos"
                    )
                    parts = []
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Generando fragmento {i + 1}/{len(chunks)}")
                        wavs, sr = await self._generate_one(request, chunk, info)
                        parts.append(wavs[0])
                    wav = self._concat_wavs(parts)

                audio = self._encode(request, wav, sr, request_id=request_id)
            except BaseException:
                # BaseException (no solo Exception): una cancelación
                # (cliente desconectado, shutdown) también debe decrementar
                # tts_active, o la métrica se fuga en cada petición caída.
                duration_ms = int((time.perf_counter() - started) * 1000)
                log_event(
                    logger, "tts_failed", request_id,
                    model=info.model_id, duration_ms=duration_ms,
                    queue_wait_ms=queue_wait_ms,
                )
                if self._metrics:
                    self._metrics.tts_failed()
                raise
            duration_ms = int((time.perf_counter() - started) * 1000)
            audio_duration_ms = int(wav.shape[0] / sr * 1000)
            if self._metrics:
                # En síntesis no-streaming el primer byte llega al terminar
                # la generación: TTFB = duración de la generación.
                self._metrics.tts_completed(
                    duration_ms,
                    audio_duration_ms=audio_duration_ms,
                    ttfb_ms=duration_ms,
                )
            log_event(
                logger, "tts_completed", request_id,
                model=info.model_id,
                request_latency_ms=int((time.perf_counter() - t_request) * 1000),
                duration_ms=duration_ms,
                audio_duration_ms=audio_duration_ms,
                rtf=round(duration_ms / audio_duration_ms, 3) if audio_duration_ms else 0.0,
                ttfb_ms=duration_ms,
                vram_used_mb=self._metrics.vram_used_mb() if self._metrics else 0,
            )
            logger.info(
                f"Generación completada (model: {info.model_id}, "
                f"{info.model_type}, {len(chunks)} fragmento(s))"
            )
            return SynthesisResult(
                audio=audio,
                sample_rate=sr,
                model_id=info.model_id,
                model_type=info.model_type,
            )

    @staticmethod
    def _concat_wavs(parts: list):
        """Concatenar arrays de onda (misma frecuencia de muestreo)."""
        import numpy as np
        return np.concatenate(parts)

    async def stream_plan(self, request: TTSRequest) -> StreamPlan:
        """Validar la petición y dividir el texto en fragmentos ANTES del response.

        Se ejecuta antes de abrir el stream: cualquier error de validación
        (texto, modelo, idioma, voz...) devuelve un 400 HTTP real, no un
        stream truncado a mitad.
        """
        text = self._require_text(request)
        self._validate_input_limits(request)
        chunks = self._chunk_text(text)
        if not chunks:
            raise TTSValidationError(
                "No se pudo dividir el texto en fragmentos para chunked streaming."
            )
        logger.info(f"Streaming: {len(chunks)} fragmentos, {len(text)} caracteres")

        async with self._queue.inference_lock():
            info = await self._resolve_model(request)
            self._build_kwargs(request, info)  # validación sin GPU

        return StreamPlan(sentences=chunks, info=info)

    async def stream_synthesize(self, request: TTSRequest, plan: StreamPlan,
                                http_request=None):
        """Generador de chunked streaming: audio por fragmentos (frases o
        párrafos, según text.chunking) en cuanto terminan.

        IMPORTANTE: cada fragmento es una generación independiente.
        - Each chunk is synthesized independently.
        - Chunking reduces memory usage and enables incremental delivery,
          but does not preserve acoustic/prosodic state between chunks.
        El modelo se reinicia entre fragmentos: no hay continuidad
        prosódica/entonativa entre chunks (no es "true streaming").

        Cada yield es el PCM 16-bit LE de una frase (sin cabecera WAV: el
        ensamblado del formato es responsabilidad de la capa HTTP). Mantiene
        queue.inference_lock() durante todo el stream (GPU exclusiva) y solo
        devuelve tras generar la frase: el primer audio llega cuando acaba
        el primer fragmento, no al terminar todo el texto.
        """
        from services import whisper_service
        from services.gpu_management import prepare_for_tts

        rc = get_runtime_config()
        request_id = self._request_id(http_request)
        text = (request.text or request.input or "").strip()
        t_request = time.perf_counter()

        t_lock = time.perf_counter()
        async with self._queue.inference_lock():
            # Tiempo de espera en la cola de inferencia (hasta el turno de GPU)
            queue_wait_ms = int((time.perf_counter() - t_lock) * 1000)
            await prepare_for_tts(self._models, self._voices, whisper_service,
                                  queue=self._queue)

            if rc.get("log_requests", True) and http_request is not None and self._metrics:
                self._metrics.log_request(http_request, text)

            if self._metrics:
                self._metrics.tts_started()
                self._metrics.queue_waited(queue_wait_ms)
            started = time.perf_counter()
            audio_ms_total = 0
            ttfb_ms = None
            try:
                for index, sentence in enumerate(plan.sentences):
                    info = await self._models.get_active_model()
                    if info is None or info.model_id != plan.info.model_id:
                        # El modelo cambió mientras se preparaba el stream
                        info = await self._resolve_model(request)
                    if index == 0:
                        # TTFB: tiempo desde la petición hasta el primer audio
                        # disponible (la métrica clave para evaluar streaming)
                        ttfb_ms = int((time.perf_counter() - t_request) * 1000)
                        log_event(
                            logger, "tts_started", request_id,
                            model=info.model_id, model_type=info.model_type,
                            text_length=len(text), chunks=len(plan.sentences),
                            streaming=True, queue_wait_ms=queue_wait_ms,
                        )
                    chunk_started = time.perf_counter()
                    wavs, sr = await self._generate_one(request, sentence, info)
                    audio = self._audio.encode_pcm(wavs[0], sr)
                    # request_id + chunk_index: nombre único y trazable por
                    # fragmento (el timestamp solo tiene precisión de 1 s).
                    self._audio.save(
                        wavs[0], sr, "tts_stream",
                        request_id=request_id,
                        chunk_index=index,
                    )
                    log_event(
                        logger, "tts_chunk_emitted", request_id,
                        chunk_index=index + 1, total_chunks=len(plan.sentences),
                        text_length=len(sentence),
                        duration_ms=int((time.perf_counter() - chunk_started) * 1000),
                        audio_duration_ms=int(wavs[0].shape[0] / sr * 1000),
                        ttfb_ms=ttfb_ms,
                    )
                    audio_ms_total += int(wavs[0].shape[0] / sr * 1000)
                    yield SynthesisResult(
                        audio=audio,
                        sample_rate=sr,
                        model_id=info.model_id,
                        model_type=info.model_type,
                    )
            except BaseException:
                # BaseException: GeneratorExit (cliente desconectado a mitad
                # de stream) y CancelledError también deben decrementar
                # tts_active, o la métrica se fuga en cada stream caído.
                if self._metrics:
                    self._metrics.tts_failed()
                raise
            duration_ms = int((time.perf_counter() - started) * 1000)
            if self._metrics:
                self._metrics.tts_completed(
                    duration_ms,
                    audio_duration_ms=audio_ms_total,
                    ttfb_ms=ttfb_ms,
                )
            log_event(
                logger, "tts_completed", request_id,
                model=plan.info.model_id,
                request_latency_ms=int((time.perf_counter() - t_request) * 1000),
                duration_ms=duration_ms,
                audio_duration_ms=audio_ms_total,
                rtf=round(duration_ms / audio_ms_total, 3) if audio_ms_total else 0.0,
                ttfb_ms=ttfb_ms,
                vram_used_mb=self._metrics.vram_used_mb() if self._metrics else 0,
                streaming=True,
            )

    async def _resolve_model(self, request: TTSRequest):
        """Modelo activo, o el solicitado en `request.model` (cambiando si procede).

        Clientes OpenWebUI envían un `model` genérico (p.ej. "tts-1") que no
        corresponde a ningún modelo local: en ese caso se ignora con un aviso
        y se usa el modelo activo.
        """
        active = await self._models.get_active_model()
        if request.model:
            if active is not None and active.model_id == request.model:
                return active
            known = set(self._models.list_local_models())
            known.update(id_ for id_ in (await self._models.registry_ids()))
            if request.model not in known:
                logger.warning(
                    f"model '{request.model}' ignorado: no es un modelo local "
                    f"(compatible con OpenWebUI). Usando el activo."
                )
                if active is None:
                    raise ModelNotLoadedError(
                        "No hay modelo cargado. Usa /model/load primero."
                    )
                return active
            try:
                return await self._models.switch_model(request.model)
            except FileNotFoundError as e:
                raise TTSValidationError(f"Modelo '{request.model}' no encontrado") from e
            except Exception as e:
                logger.error(f"Error cargando modelo '{request.model}': {e}")
                raise ModelLoadingError(
                    f"No se pudo cargar el modelo '{request.model}'"
                ) from e
        if active is None:
            raise ModelNotLoadedError("No hay modelo cargado. Usa /model/load primero.")
        return active
