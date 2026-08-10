#!/usr/bin/env python3
"""Servicio de generación TTS."""

import os
import logging

from services.config_service import get_runtime_config

logger = logging.getLogger("tts")


class TTSService:
    """Orquestación de la generación de audio TTS.

    Toda la interacción con el modelo pasa por ModelManager:
    este servicio nunca manipula la instancia del modelo directamente.
    """

    def __init__(self, config: dict, model_manager, voice_manager, audio_service):
        self._config = config
        self._models = model_manager
        self._voices = voice_manager
        self._audio = audio_service

    def _default_ref_voice(self):
        """Voz local de referencia para el modelo base (wav + texto)."""
        rc = get_runtime_config()
        candidates = [rc.get("def_voice", "")]
        try:
            for name in sorted(os.listdir(self._config.paths.voices_dir)):
                candidates.append(name)
        except OSError:
            pass
        for name in candidates:
            if not name:
                continue
            voice_dir = os.path.join(self._config.paths.voices_dir, name)
            wav = os.path.join(voice_dir, "voice.wav")
            txt = os.path.join(voice_dir, "text.txt")
            if os.path.exists(wav) and os.path.exists(txt):
                with open(txt, "r", encoding="utf-8") as f:
                    return wav, f.read().strip()
        return None

    async def generate(self, text: str, language: str, speaker: str, instruct: str) -> tuple:
        """Generar audio TTS según el tipo de modelo activo.

        Devuelve (bytes_wav, sample_rate).
        """
        info = await self._models.get_active_model()
        if info is None:
            raise ValueError("No hay modelo cargado. Usa /model/load primero.")

        rc = get_runtime_config()
        model_type = info.model_type
        lang = language or rc["def_language"]
        clone_prompt = self._voices.clone_prompt

        if model_type == "voice_design":
            instruct_text = instruct or rc["def_instruct"] or ""
            logger.info(f"Usando voice design (instruct: {instruct_text[:60] or '(vacío)'})")
            wavs, sr = await self._models.generate_voice_design(
                text=text,
                language=lang,
                instruct=instruct_text,
            )
        elif model_type == "base":
            if clone_prompt:
                logger.info("Usando voice cloning (modelo base)")
                wavs, sr = await self._models.generate_voice_clone(
                    text=text,
                    language=lang,
                    voice_clone_prompt=clone_prompt,
                )
            else:
                ref = self._default_ref_voice()
                if ref is None:
                    raise ValueError(
                        "El modelo Base requiere una voz de referencia. "
                        "Usa 'Voz -> cargar' en el panel o crea una voz local."
                    )
                wav_path, ref_text = ref
                logger.info(f"Usando voz local de referencia: {wav_path}")
                wavs, sr = await self._models.generate_voice_clone(
                    text=text,
                    language=lang,
                    ref_audio=wav_path,
                    ref_text=ref_text,
                )
        else:
            logger.info(f"Usando voz por defecto (speaker={speaker or rc['def_voice']})")
            wavs, sr = await self._models.generate_custom_voice(
                text=text,
                language=lang,
                speaker=speaker or rc["def_voice"],
                instruct=instruct or rc["def_instruct"] or None,
            )

        # Guardar audio en disco
        self._audio.save(wavs[0], sr, "tts")

        # Preparar respuesta HTTP
        logger.info("Generación completada")
        return self._audio.encode_wav(wavs[0], sr), sr
