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

from fastapi import HTTPException

from schemas.tts import TTSRequest
from security.validation import validate_text
from services.config_service import get_runtime_config

logger = logging.getLogger("tts")


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

    def _local_voice_ref(self, name: str):
        """(wav, text) de una voz local por nombre, o None si no existe."""
        if not name:
            return None
        voice_dir = os.path.join(self._config.paths.voices_dir, name)
        wav = os.path.join(voice_dir, "voice.wav")
        txt = os.path.join(voice_dir, "text.txt")
        if not (os.path.exists(wav) and os.path.exists(txt)):
            return None
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
        if info.supported_speakers is not None and speaker not in info.supported_speakers:
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
        if info.supported_languages is not None and lang not in info.supported_languages:
            raise TTSValidationError(
                f"language '{lang}' no soportado. Soportados: {', '.join(info.supported_languages)}"
            )
        return lang

    # -- Síntesis -----------------------------------------------------------

    async def synthesize(self, request: TTSRequest, http_request=None) -> SynthesisResult:
        """Pipeline completo de síntesis: validación + generación + codificación.

        Único punto de generación del servidor; todos los endpoints TTS pasan
        por aquí. Ejecuta dentro de queue.inference_lock().
        """
        from services import whisper_service
        from services.gpu_management import prepare_for_tts

        text = (request.text or request.input or "").strip()
        if not text:
            raise TTSValidationError("Campo 'text' o 'input' requerido")

        rc = get_runtime_config()
        try:
            validate_text(text, rc["max_text_chars"])
        except HTTPException as e:
            raise TTSValidationError(e.detail)

        async with self._queue.inference_lock():
            await prepare_for_tts(self._models, self._voices, whisper_service)

            if rc.get("log_requests", True) and http_request is not None and self._metrics:
                self._metrics.log_request(http_request, text)

            info = await self._resolve_model(request)
            model_type = info.model_type
            lang = self._validate_language(request, info)

            # Validación según el tipo de modelo (antes de consumir GPU)
            generation_kwargs = {}
            if request.temperature is not None:
                # Parámetro real de generación de la librería Qwen3-TTS.
                generation_kwargs["temperature"] = request.temperature

            if model_type == "voice_design":
                description = self._validate_voice_design(request)
                logger.info(f"Usando voice design (description: {description[:60]})")
                generation_kwargs.update(
                    {"text": text, "language": lang, "instruct": description}
                )
                wavs, sr = await self._models.generate_voice_design(**generation_kwargs)
            elif model_type == "base":
                ref = self._validate_base(request)
                if ref is None:
                    logger.info("Usando voice cloning (voz cargada)")
                    generation_kwargs.update(
                        {"text": text, "language": lang,
                         "voice_clone_prompt": self._voices.clone_prompt}
                    )
                else:
                    wav_path, ref_text = ref
                    logger.info(f"Usando voz de referencia: {wav_path}")
                    generation_kwargs.update(
                        {"text": text, "language": lang,
                         "ref_audio": wav_path, "ref_text": ref_text}
                    )
                wavs, sr = await self._models.generate_voice_clone(**generation_kwargs)
            else:
                speaker = self._validate_custom_voice(request, info)
                rc = get_runtime_config()
                instruct = (
                    request.voice_description or request.instruct or rc.get("def_instruct")
                ) or None
                logger.info(f"Usando voz por defecto (speaker={speaker})")
                generation_kwargs.update({"text": text, "language": lang, "speaker": speaker})
                if instruct:
                    generation_kwargs["instruct"] = instruct
                wavs, sr = await self._models.generate_custom_voice(**generation_kwargs)

            # Codificar audio
            self._audio.save(wavs[0], sr, "tts")
            if request.output_format == "pcm":
                audio = self._audio.encode_pcm(wavs[0], sr)
            else:
                audio = self._audio.encode_wav(wavs[0], sr)

            logger.info(f"Generación completada (model: {info.model_id}, {model_type})")
            return SynthesisResult(
                audio=audio,
                sample_rate=sr,
                model_id=info.model_id,
                model_type=model_type,
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
