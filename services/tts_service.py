#!/usr/bin/env python3
"""Servicio de generación TTS."""

import logging
import asyncio
import io
import os
import soundfile as sf
from config.settings import CONFIG
from services.config_service import get_runtime_config
from utils.helpers import save_audio

logger = logging.getLogger("tts")


def _get_model_type(model, entry_type: str) -> str:
    """Resolver el tipo de modelo (custom_voice, voice_design, base...)."""
    if entry_type and entry_type != "unknown":
        return entry_type
    return getattr(
        getattr(model, "model", None),
        "tts_model_type",
        getattr(model, "tts_model_type", "custom_voice"),
    )


def _default_ref_voice():
    """Voz local de referencia para el modelo base (wav + texto)."""
    rc = get_runtime_config()
    candidates = [rc.get("def_voice", "")]
    try:
        for name in sorted(os.listdir(CONFIG["local_voices_dir"])):
            candidates.append(name)
    except OSError:
        pass
    for name in candidates:
        if not name:
            continue
        voice_dir = os.path.join(CONFIG["local_voices_dir"], name)
        wav = os.path.join(voice_dir, "voice.wav")
        txt = os.path.join(voice_dir, "text.txt")
        if os.path.exists(wav) and os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                return wav, f.read().strip()
    return None


async def generate_tts(
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    model_registry: dict,
    current_model_id: str,
    clone_prompt
):
    """Generar audio TTS según el tipo de modelo y su estado."""
    
    entry = model_registry[current_model_id]
    model = entry["model"]
    rc = get_runtime_config()
    model_type = _get_model_type(model, entry.get("type", ""))
    lang = language or rc["def_language"]
    
    if model_type == "voice_design":
        instruct_text = instruct or rc["def_instruct"] or ""
        logger.info(f"Usando voice design (instruct: {instruct_text[:60] or '(vacío)'})")
        wavs, sr = await asyncio.to_thread(
            model.generate_voice_design,
            text=text,
            language=lang,
            instruct=instruct_text
        )
    elif model_type == "base":
        if clone_prompt:
            logger.info("Usando voice cloning (modelo base)")
            wavs, sr = await asyncio.to_thread(
                model.generate_voice_clone,
                text=text,
                language=lang,
                voice_clone_prompt=clone_prompt
            )
        else:
            ref = _default_ref_voice()
            if ref is None:
                raise ValueError(
                    "El modelo Base requiere una voz de referencia. "
                    "Usa 'Voz -> cargar' en el panel o crea una voz local."
                )
            wav_path, ref_text = ref
            logger.info(f"Usando voz local de referencia: {wav_path}")
            wavs, sr = await asyncio.to_thread(
                model.generate_voice_clone,
                text=text,
                language=lang,
                ref_audio=wav_path,
                ref_text=ref_text
            )
    else:
        logger.info(f"Usando voz por defecto (speaker={speaker or rc['def_voice']})")
        wavs, sr = await asyncio.to_thread(
            model.generate_custom_voice,
            text=text,
            language=lang,
            speaker=speaker or rc["def_voice"],
            instruct=instruct or rc["def_instruct"] or None
        )
    
    # Guardar audio en disco
    save_audio(wavs[0], sr, "tts", CONFIG["audios_dir"])
    
    # Preparar respuesta HTTP
    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="wav")
    logger.info("Generación completada")
    
    return buffer.getvalue(), sr
