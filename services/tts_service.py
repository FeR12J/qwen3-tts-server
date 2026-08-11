#!/usr/bin/env python3
"""Servicio de generación TTS (capa de orquestación).

Único punto de síntesis del servidor: todos los endpoints TTS terminan en
``synthesize(request)``. Aquí se valida la petición según el tipo de modelo
activo (antes de consumir GPU), se resuelven las referencias de voz y se
despacha a la librería Qwen3-TTS a través de ModelManager (este servicio
nunca manipula la instancia del modelo directamente).
"""

import os
import logging

from dataclasses import dataclass

from schemas.tts import TTSRequest
from services.audio_service import AudioValidationError
from services.config_service import get_runtime_config
from services.model_manager import ModelInfo
from utils.chunker import TextChunker, TextChunkerError

logger = logging.getLogger("tts")

# Ritmo medio de habla usado para estimar la duración del audio generado
# a partir de la longitud del texto (antes de usar GPU).
CHARS_PER_SECOND = 16.0


class TTSValidationError(Exception):
    """Petición TTS inválida (400). Se traduce en:
    {"error": {"code": "INVALID_TTS_REQUEST", "message": "..."}}."""

    def __init__(self, message: str, code: str = "INVALID_TTS_REQUEST"):
        super().__init__(message)
        self.code = code
        self.message = message


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
        from config.settings import settings
        limits = settings.limits
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
        """(wav, text) de una voz local por nombre, o None si no existe."""
        if not name:
            return None
        voice_dir = os.path.join(self._config.paths.voices_dir, name)
        wav = os.path.join(voice_dir, "voice.wav")
        txt = os.path.join(voice_dir, "text.txt")
        if not (os.path.exists(wav) and os.path.exists(txt)):
            return None
        self._check_ref_audio_size(wav)
        with open(txt, "r", encoding="utf-8") as f:
            return wav, f.read().strip()

    def _default_ref_voice(self):
        """Voz local de referencia por defecto (def_voice o la primera válida)."""
        rc = get_runtime_config()
        candidates = [rc.get("def_voice", "")]
        try:
            for name in sorted(os.listdir(self._config.paths.voices_dir)):
                candidates.append(name)
        except OSError:
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

    def _validate_base(self, request: TTSRequest):
        """Validar y resolver la referencia de clonación para modelos Base.

        Orden: reference_audio -> voice local -> voz cargada -> voz por defecto.
        Devuelve None si se usará voice_clone_prompt, o (wav, text).
        """
        if request.reference_audio:
            if not os.path.exists(request.reference_audio):
                raise TTSValidationError(
                    f"reference_audio no existe: {request.reference_audio}"
                )
            self._check_ref_audio_size(request.reference_audio)
            ref_text = (request.reference_text or "").strip()
            if not ref_text:
                raise TTSValidationError(
                    "reference_text es obligatorio cuando se proporciona reference_audio."
                )
            return request.reference_audio, ref_text

        if request.voice:
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
            l.lower() for l in info.supported_languages
        }:
            raise TTSValidationError(
                f"language '{lang}' no soportado. Soportados: {', '.join(info.supported_languages)}"
            )
        return lang

    # -- Síntesis -----------------------------------------------------------

    def _require_text(self, request: TTSRequest) -> str:
        """Validar el texto de entrada contra los límites (antes de GPU).

        Aplica limits.max_text_characters (longitud) y una estimación de
        limits.max_audio_duration_seconds a partir de la longitud del texto
        (ritmo medio de habla ~16 caracteres/segundo).
        """
        text = (request.text or request.input or "").strip()
        if not text:
            raise TTSValidationError("Campo 'text' o 'input' requerido")

        from config.settings import settings
        if len(text) > settings.limits.max_text_characters:
            raise TTSValidationError(
                f"Texto demasiado largo: {len(text)} caracteres "
                f"(máximo configurado: {settings.limits.max_text_characters})."
            )

        estimated = len(text) / CHARS_PER_SECOND
        if estimated > settings.limits.max_audio_duration_seconds:
            raise TTSValidationError(
                f"Audio estimado demasiado largo: ~{estimated:.1f}s "
                f"(máximo configurado: {settings.limits.max_audio_duration_seconds}s). "
                "Reduce la longitud del texto."
            )
        return text

    def _validate_input_limits(self, request: TTSRequest):
        """Límites de entrada independientes del modelo (antes de GPU/lock).

        Comprueba el tamaño del audio de referencia indicado en la petición
        (reference_audio o voz local) contra limits.max_reference_audio_mb.
        """
        if request.reference_audio:
            # Si no existe, el error lo dará la validación de la referencia.
            if os.path.exists(request.reference_audio):
                self._check_ref_audio_size(request.reference_audio)
        if request.voice:
            wav = os.path.join(self._config.paths.voices_dir, request.voice, "voice.wav")
            if os.path.exists(wav):
                self._check_ref_audio_size(wav)

    def _make_chunker(self) -> TextChunker:
        """Chunker de textos largos según la configuración.

        - ``limits.max_text_characters``: límite del texto de entrada.
        - ``text.chunking``: modo de división (sentence | paragraph).
        - ``max_text_chars`` (runtime): tamaño máximo de cada fragmento
          generado (editable desde el panel).
        """
        from config.settings import settings
        rc = get_runtime_config()
        return TextChunker(
            max_characters=settings.limits.max_text_characters,
            chunking=settings.text.chunking,
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

    def _encode(self, request: TTSRequest, wav, sr) -> bytes:
        """Guardar copia y codificar el audio según output_format."""
        self._audio.save(wav, sr, "tts")
        if request.output_format == "pcm":
            return self._audio.encode_pcm(wav, sr)
        return self._audio.encode_wav(wav, sr)

    async def synthesize(self, request: TTSRequest, http_request=None) -> SynthesisResult:
        """Pipeline completo de síntesis: validación + generación + codificación.

        Único punto de generación no-streaming del servidor; todos los
        endpoints TTS pasan por aquí. Ejecuta dentro de queue.inference_lock().

        Textos largos (hasta limits.max_text_characters) se dividen en fragmentos
        (TextChunker) y se generan secuencialmente, concatenando el audio.
        """
        from services import whisper_service
        from services.gpu_management import prepare_for_tts

        text = self._require_text(request)
        self._validate_input_limits(request)
        rc = get_runtime_config()

        async with self._queue.inference_lock():
            # Validación completa (idioma, speaker, refs de voz, tamaños)
            # ANTES de preparar/consumir GPU.
            info = await self._resolve_model(request)
            self._build_kwargs(request, info)

            await prepare_for_tts(self._models, self._voices, whisper_service)

            if rc.get("log_requests", True) and http_request is not None and self._metrics:
                self._metrics.log_request(http_request, text)

            chunks = self._chunk_text(text)
            if not chunks:
                raise TTSValidationError(
                    "No se pudo dividir el texto en fragmentos para generar."
                )

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

            audio = self._encode(request, wav, sr)
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
                "No se pudo dividir el texto en fragmentos para streaming."
            )
        logger.info(f"Streaming: {len(chunks)} fragmentos, {len(text)} caracteres")

        async with self._queue.inference_lock():
            info = await self._resolve_model(request)
            self._build_kwargs(request, info)  # validación sin GPU

        return StreamPlan(sentences=chunks, info=info)

    async def stream_synthesize(self, request: TTSRequest, plan: StreamPlan,
                                http_request=None):
        """Generador de streaming real: audio por fragmentos (frases o párrafos, según text.chunking) en cuanto terminan.

        Cada yield es el PCM 16-bit LE de una frase (sin cabecera WAV: el
        ensamblado del formato es responsabilidad de la capa HTTP). Mantiene
        queue.inference_lock() durante todo el stream (GPU exclusiva) y solo
        devuelve tras generar la frase: el primer audio llega cuando acaba
        el primer fragmento, no al terminar todo el texto.
        """
        from services import whisper_service
        from services.gpu_management import prepare_for_tts

        rc = get_runtime_config()

        async with self._queue.inference_lock():
            await prepare_for_tts(self._models, self._voices, whisper_service)

            if rc.get("log_requests", True) and http_request is not None and self._metrics:
                self._metrics.log_request(
                    http_request, (request.text or request.input or "")
                )

            for sentence in plan.sentences:
                info = await self._models.get_active_model()
                if info is None or info.model_id != plan.info.model_id:
                    # El modelo cambió mientras se preparaba el stream
                    info = await self._resolve_model(request)
                wavs, sr = await self._generate_one(request, sentence, info)
                audio = self._audio.encode_pcm(wavs[0], sr)
                self._audio.save(wavs[0], sr, "tts_stream")
                logger.info(f"Fragmento emitido ({len(sentence)} chars, model: {info.model_id})")
                yield SynthesisResult(
                    audio=audio,
                    sample_rate=sr,
                    model_id=info.model_id,
                    model_type=info.model_type,
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
                    raise TTSValidationError(
                        "No hay modelo cargado. Usa /model/load primero."
                    )
                return active
            try:
                return await self._models.switch_model(request.model)
            except FileNotFoundError as e:
                raise TTSValidationError(f"Modelo '{request.model}' no encontrado") from e
            except Exception as e:
                logger.error(f"Error cargando modelo '{request.model}': {e}")
                raise TTSValidationError(f"No se pudo cargar el modelo '{request.model}'") from e
        if active is None:
            raise TTSValidationError("No hay modelo cargado. Usa /model/load primero.")
        return active
