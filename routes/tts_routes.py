#!/usr/bin/env python3
"""Rutas del servidor TTS."""

import os
import re
import shutil
import asyncio
import logging
import subprocess
import tempfile
import traceback
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Depends, Form, File, UploadFile
from fastapi.responses import Response
from config.settings import CONFIG
from utils.auth import require_api_key
from services.config_service import get_runtime_config, resolve_device
from utils.helpers import validate_text, log_request, get_vram_available, clear_models
from schemas.schemas import TTSRequest, TTSRequestOpenWebUI, LoadModelRequest, LoadVoiceRequest
from services.tts_service import generate_tts

logger = logging.getLogger("tts")

# Tamaño máximo del audio subido para crear voces (50 MB)
MAX_VOICE_AUDIO_BYTES = 50 * 1024 * 1024

# Reproductores de audio disponibles en el sistema para reproducir en este equipo
PLAYERS = [p for p in ["mpv", "ffplay", "paplay", "aplay", "play"] if shutil.which(p)]

# Estado de reproducción: lock para serializar reproducciones y proceso actual
playback_lock = asyncio.Lock()
playback_proc = [None]  # Wrapper para mutabilidad


async def _create_voice_clone(model, wav_path: str, txt_path: str):
    """Crear el prompt de voz clonada a partir de un audio de referencia y su transcripción."""
    with open(txt_path, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    return await asyncio.to_thread(
        model.create_voice_clone_prompt,
        ref_audio=wav_path,
        ref_text=ref_text,
        x_vector_only_mode=False
    )


def create_tts_routes(app: FastAPI, model_registry, current_model_id_var, clone_prompt_var, semaphore):
    """Crear todas las rutas del servidor TTS."""
    
    @app.get("/")
    async def root():
        return {
            "status": "ok",
            "current_model": current_model_id_var[0],
            "clone_active": clone_prompt_var[0] is not None,
            "vram_available_gb": get_vram_available(),
            "device": resolve_device()
        }

    @app.post("/tts", dependencies=[Depends(require_api_key)])
    async def tts_endpoint(req_body: TTSRequest, req: Request):
        async with semaphore:
            rc = get_runtime_config()
            validate_text(req_body.text, rc["max_text_chars"])
            if rc.get("log_requests", True):
                log_request(req, req_body.text, CONFIG["log_file"], CONFIG["log_max_bytes"])
            
            if current_model_id_var[0] is None:
                raise HTTPException(400, "No hay modelo cargado. Usa /model/load primero.")
            
            try:
                audio_bytes, sr = await generate_tts(
                    text=req_body.text,
                    language=req_body.language,
                    speaker=req_body.speaker,
                    instruct=req_body.instruct,
                    model_registry=model_registry,
                    current_model_id=current_model_id_var[0],
                    clone_prompt=clone_prompt_var[0]
                )
                return Response(audio_bytes, media_type="audio/wav")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error en generación TTS: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error generando audio: {str(e)}")

    @app.post("/tts/play", dependencies=[Depends(require_api_key)])
    async def tts_play_endpoint(req_body: TTSRequest, req: Request):
        """Generar TTS y reproducirlo directamente en este equipo, esperando a que la reproducción anterior termine."""
        async with semaphore:
            rc = get_runtime_config()
            validate_text(req_body.text, rc["max_text_chars"])
            if rc.get("log_requests", True):
                log_request(req, req_body.text, CONFIG["log_file"], CONFIG["log_max_bytes"])

            if current_model_id_var[0] is None:
                raise HTTPException(400, "No hay modelo cargado. Usa /model/load primero.")

            if not PLAYERS:
                raise HTTPException(500, "No hay reproductor de audio disponible (mpv, ffplay, paplay, aplay o play)")

            async with playback_lock:
                proc = playback_proc[0]
                if proc is not None and proc.poll() is None:
                    timeout = rc["playback_wait_timeout"]
                    logger.info(f"Esperando a que termine la reproducción anterior (máx. {timeout}s)...")
                    try:
                        await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout esperando a la reproducción anterior ({timeout}s)")
                        proc.kill()
                        await asyncio.to_thread(proc.wait)
                        raise HTTPException(
                            504,
                            f"Timeout esperando a que termine la reproducción anterior ({timeout}s)"
                        )
                    playback_proc[0] = None

                try:
                    audio_bytes, sr = await generate_tts(
                        text=req_body.text,
                        language=req_body.language,
                        speaker=req_body.speaker,
                        instruct=req_body.instruct,
                        model_registry=model_registry,
                        current_model_id=current_model_id_var[0],
                        clone_prompt=clone_prompt_var[0]
                    )

                    tmp_path = os.path.join(
                        tempfile.gettempdir(),
                        f"tts_play_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
                    )
                    with open(tmp_path, "wb") as f:
                        f.write(audio_bytes)

                    player = PLAYERS[0]
                    if player == "ffplay":
                        cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
                    else:
                        cmd = [player, tmp_path]

                    playback_proc[0] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.info(f"Audio en reproducción en este equipo (player={player}): {tmp_path}")

                    return {
                        "status": "ok",
                        "message": "Audio generado y reproduciéndose en este equipo",
                        "player": player,
                        "temp_file": tmp_path,
                        "sample_rate": sr
                    }

                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error en reproducción TTS: {e}")
                    logger.debug(traceback.format_exc())
                    raise HTTPException(500, f"Error generando o reproduciendo audio: {str(e)}")
    @app.post("/tts/audio/speech", dependencies=[Depends(require_api_key)])
    async def openwebui_tts(req_body: TTSRequestOpenWebUI, req: Request):
        async with semaphore:
            rc = get_runtime_config()
            text = req_body.text or req_body.input
            
            if not text:
                raise HTTPException(400, "Campo 'text' o 'input' requerido")
                
            validate_text(text, rc["max_text_chars"])
            if rc.get("log_requests", True):
                log_request(req, text, CONFIG["log_file"], CONFIG["log_max_bytes"])
            
            if current_model_id_var[0] is None:
                raise HTTPException(500, "No hay modelo cargado. Usa /model/load primero.")
            
            try:
                audio_bytes, sr = await generate_tts(
                    text=text,
                    language=req_body.language,
                    speaker=req_body.speaker,
                    instruct=req_body.instruct,
                    model_registry=model_registry,
                    current_model_id=current_model_id_var[0],
                    clone_prompt=clone_prompt_var[0]
                )
                logger.info("Generación completada (OpenWebUI)")
                return Response(audio_bytes, media_type="audio/wav")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error en generación OpenWebUI: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error generando audio: {str(e)}")

    @app.post("/model/load", dependencies=[Depends(require_api_key)])
    async def load_model_endpoint(req_body: LoadModelRequest):
        async with semaphore:
            # Resetear voice cloning al cargar nuevo modelo
            clone_prompt_var[0] = None
            
            model_id = req_body.model_id.strip()
            
            if not model_id:
                raise HTTPException(400, "model_id vacío")
            
            model_path = os.path.join(CONFIG["local_models_dir"], model_id)
            if not os.path.exists(model_path):
                raise HTTPException(404, f"Modelo '{model_id}' no encontrado en {CONFIG['local_models_dir']}")
            
            try:
                from services.model_service import load_model
                model = await load_model(model_id, model_registry)
                
                # Obtener info del modelo cargado
                entry = model_registry[model_id]
                model_type = entry["type"]
                current_model_id_var[0] = model_id
                
                return {
                    "status": "ok",
                    "loaded_model": model_id,
                    "model_type": model_type,
                    "vram_available_gb": get_vram_available()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error cargando modelo: {str(e)}")

    @app.post("/model/unload", dependencies=[Depends(require_api_key)])
    async def unload_model_endpoint():
        async with semaphore:
            if current_model_id_var[0] is None:
                return {"status": "ok", "message": "No hay modelo cargado"}
            
            model_id = current_model_id_var[0]
            await asyncio.to_thread(clear_models, model_registry)
            current_model_id_var[0] = None
            clone_prompt_var[0] = None
            
            logger.info(f"Modelo descargado: {model_id}")
            return {
                "status": "ok",
                "unloaded_model": model_id,
                "message": f"Modelo '{model_id}' descargado y VRAM liberada"
            }

    @app.get("/models")
    @app.get("/tts/audio/models")
    async def list_models():
        try:
            local_models = [
                d for d in os.listdir(CONFIG["local_models_dir"])
                if os.path.isdir(os.path.join(CONFIG["local_models_dir"], d))
            ]
            
            return {
                "available_models": sorted(local_models),
                "current_model": current_model_id_var[0],
                "models_dir": CONFIG["local_models_dir"]
            }
        except Exception as e:
            logger.error(f"Error listando modelos: {e}")
            raise HTTPException(500, f"Error leyendo directorio de modelos: {str(e)}")

    @app.post("/voice/load", dependencies=[Depends(require_api_key)])
    async def load_voice(req_body: LoadVoiceRequest):
        async with semaphore:
            
            if current_model_id_var[0] is None:
                raise HTTPException(400, "No hay modelo cargado. Usa /model/load primero.")
            
            voice_name = req_body.voice_name.strip()
            if not voice_name:
                raise HTTPException(400, "voice_name vacío")
            
            voice_path = os.path.join(CONFIG["local_voices_dir"], voice_name)
            
            if not os.path.exists(voice_path):
                raise HTTPException(404, f"Voz '{voice_name}' no encontrada en {CONFIG['local_voices_dir']}")
            
            wav_path = os.path.join(voice_path, "voice.wav")
            txt_path = os.path.join(voice_path, "text.txt")
            
            if not os.path.exists(wav_path):
                raise HTTPException(400, f"Falta voice.wav en {voice_path}")
            if not os.path.exists(txt_path):
                raise HTTPException(400, f"Falta text.txt en {voice_path}")
            
            entry = model_registry[current_model_id_var[0]]
            model = entry["model"]
            model_type = entry["type"]
            
            # Verificar compatibilidad del modelo con voice cloning (solo modelos base)
            if model_type not in ("base", "unknown"):
                raise HTTPException(
                    400,
                    f"El modelo actual ({current_model_id_var[0]}) no soporta voice cloning (tipo: {model_type})"
                )
            
            try:
                logger.info(f"Creando voz clonada para: {voice_name}")
                
                clone_prompt_var[0] = await _create_voice_clone(model, wav_path, txt_path)
                
                logger.info(f"Voz clonada creada y aplicada: {voice_name}")
                
                return {
                    "status": "ok",
                    "voice": voice_name,
                    "model": current_model_id_var[0],
                    "message": f"Voz '{voice_name}' lista para usar"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error creando voz clonada: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error creando voz clonada: {str(e)}")

    @app.post("/voice/create", dependencies=[Depends(require_api_key)])
    async def create_voice(
        voice_name: str = Form(...),
        text: str = Form(...),
        audio: UploadFile = File(...),
    ):
        """Crear una voz subiendo un WAV y su transcripción. La guarda y la clona."""
        async with semaphore:
            if current_model_id_var[0] is None:
                raise HTTPException(400, "No hay modelo cargado. Usa /model/load primero.")
            
            voice_name = voice_name.strip()
            if not voice_name:
                raise HTTPException(400, "voice_name vacío")
            if not re.match(r"^[A-Za-z0-9_\-]+$", voice_name):
                raise HTTPException(400, "El nombre de la voz solo puede contener letras, números, guiones y guiones bajos")
            
            text = text.strip()
            if not text:
                raise HTTPException(400, "La transcripción no puede estar vacía")
            
            if not audio.filename or not audio.filename.lower().endswith(".wav"):
                raise HTTPException(400, "El archivo debe ser un WAV (.wav)")
            
            entry = model_registry[current_model_id_var[0]]
            model = entry["model"]
            
            # Verificar compatibilidad del modelo con voice cloning
            if entry["type"] == "custom_voice":
                raise HTTPException(
                    400,
                    f"El modelo actual ({current_model_id_var[0]}) no soporta voice cloning (tipo: {entry['type']})"
                )
            
            data = await audio.read()
            if len(data) > MAX_VOICE_AUDIO_BYTES:
                raise HTTPException(400, f"El archivo de audio excede {MAX_VOICE_AUDIO_BYTES // (1024 * 1024)} MB")
            
            # Guardar archivos en voices/<nombre>/
            voice_dir = os.path.join(CONFIG["local_voices_dir"], voice_name)
            os.makedirs(voice_dir, exist_ok=True)
            wav_path = os.path.join(voice_dir, "voice.wav")
            txt_path = os.path.join(voice_dir, "text.txt")
            
            with open(wav_path, "wb") as f:
                f.write(data)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            logger.info(f"Voz '{voice_name}' guardada en {voice_dir}")
            
            try:
                logger.info(f"Creando voz clonada para: {voice_name}")
                clone_prompt_var[0] = await _create_voice_clone(model, wav_path, txt_path)
                logger.info(f"Voz clonada creada y aplicada: {voice_name}")
                
                return {
                    "status": "ok",
                    "voice": voice_name,
                    "model": current_model_id_var[0],
                    "message": f"Voz '{voice_name}' creada, guardada y aplicada"
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error creando voz clonada: {e}")
                logger.debug(traceback.format_exc())
                raise HTTPException(500, f"Error creando voz clonada: {str(e)}")

    @app.post("/voice/unload", dependencies=[Depends(require_api_key)])
    async def unload_voice():
        async with semaphore:
            if clone_prompt_var[0] is None:
                return {
                    "status": "ok",
                    "message": "Voice cloning ya estaba desactivado"
                }
            
            clone_prompt_var[0] = None
            
            logger.info("Voice cloning desactivado")
            
            return {
                "status": "ok",
                "message": "Voice cloning desactivado"
            }

    @app.get("/voices")
    @app.get("/tts/audio/voices")
    async def list_voices():
        try:
            voices = [
                d for d in os.listdir(CONFIG["local_voices_dir"])
                if os.path.isdir(os.path.join(CONFIG["local_voices_dir"], d))
            ]
            
            # Verificar estructura de cada voz
            voice_info = []
            for v in voices:
                voice_path = os.path.join(CONFIG["local_voices_dir"], v)
                has_wav = os.path.exists(os.path.join(voice_path, "voice.wav"))
                has_txt = os.path.exists(os.path.join(voice_path, "text.txt"))
                voice_info.append({
                    "name": v,
                    "valid": has_wav and has_txt,
                    "has_voice_wav": has_wav,
                    "has_text_txt": has_txt
                })
            
            return {
                "available_voices": [v["name"] for v in voice_info],
                "clone_active": clone_prompt_var[0] is not None,
                "voices_detail": voice_info
            }
        except Exception as e:
            logger.error(f"Error listando voces: {e}")
            raise HTTPException(500, f"Error leyendo directorio de voces: {str(e)}")
