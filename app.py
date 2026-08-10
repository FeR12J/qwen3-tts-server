#!/usr/bin/env python3
"""Aplicación principal del servidor TTS."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import sys

# Agregar ruta para imports relativos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import CONFIG
from utils.helpers import get_vram_available, cleanup_old_audios
from routes.tts_routes import create_tts_routes
from routes.webui_routes import create_webui_routes
from routes.whisper_routes import create_whisper_routes
from services.model_service import load_model
from services import config_service

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tts")

# Variables globales (usando listas para mutabilidad en async)
model_registry = {}
current_model_id_var = [None]  # Wrapper para mutabilidad
clone_prompt_var = [None]  # Prompt único de clonación
semaphore = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación."""
    config_service.load_runtime_config()
    config_service.apply_log_level()
    create_tts_routes(app, model_registry, current_model_id_var, clone_prompt_var, semaphore)
    create_webui_routes(app, model_registry, current_model_id_var, clone_prompt_var)
    create_whisper_routes(app, semaphore)
    await startup_procedure()
    yield


# Inicializar aplicación FastAPI
app = FastAPI(title="Qwen3-TTS API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def startup_procedure():
    """Procedimiento de inicialización del servidor."""
    
    print("\n" + "."*60)
    print("Iniciando qwen3-tts server")
    print("."*60)
    
    # Limpiar audios antiguos
    cleanup_old_audios(CONFIG["audios_dir"], CONFIG["audios_max_age_days"])
    
    # Modelos locales
    try:
        local_models = [d for d in os.listdir(CONFIG["local_models_dir"]) 
                       if os.path.isdir(os.path.join(CONFIG["local_models_dir"], d))]
        
        print(f"\n📦 Modelos locales disponibles ({len(local_models)}):")
        for i, m in enumerate(sorted(local_models), 1):
            print(f"   {i}. {m}")
            
    except FileNotFoundError:
        print(f"\nDirectorio de modelos no encontrado: {CONFIG['local_models_dir']}")
        local_models = []
    except Exception as e:
        print(f"\nError leyendo directorio de modelos: {e}")
        local_models = []
    
    # Seleccionar modelo por defecto (configurable, fallback al primero disponible)
    if len(local_models) > 0:
        selected_model = None
        if CONFIG.get("default_model") and CONFIG["default_model"] in local_models:
            selected_model = CONFIG["default_model"]
        else:
            selected_model = local_models[0]
            if CONFIG.get("default_model"):
                print(f"\n⚠️  Modelo por defecto '{CONFIG['default_model']}' no encontrado, usando '{selected_model}'")
        
        print(f"\nModelo seleccionado por defecto: {selected_model}")
        
        try:
            model = await load_model(selected_model, model_registry)
            entry = model_registry[selected_model]
            current_model_id_var[0] = selected_model
            print(f"   Tipo: {entry['type']}")
            print(f"   VRAM disponible: {get_vram_available()} GB")
        except Exception as e:
            print(f"\nError cargando modelo por defecto: {e}")
    else:
        print("\nNo hay modelos disponibles. El servidor funcionará sin modelo inicial.")
    
    # Voces locales
    try:
        local_voices = [d for d in os.listdir(CONFIG["local_voices_dir"]) 
                       if os.path.isdir(os.path.join(CONFIG["local_voices_dir"], d))]
        
        print(f"\nVoces locales disponibles ({len(local_voices)}):")
        for i, v in enumerate(sorted(local_voices), 1):
            voice_path = os.path.join(CONFIG["local_voices_dir"], v)
            has_wav = os.path.exists(os.path.join(voice_path, "voice.wav"))
            has_txt = os.path.exists(os.path.join(voice_path, "text.txt"))
            status = "OK" if (has_wav and has_txt) else ("KO" if not has_wav else "!?")
            print(f"   {i}. {v} {status}")
            
    except FileNotFoundError:
        print(f"\nDirectorio de voces no encontrado: {CONFIG['local_voices_dir']}")
        local_voices = []
    except Exception as e:
        print(f"\nError leyendo directorio de voces: {e}")
        local_voices = []
    
    # Intentar clonar voz por defecto si hay modelos y voces disponibles
    if len(local_models) > 0 and len(local_voices) >= 1 and current_model_id_var[0] is not None:
        
        selected_voice = None
        if CONFIG.get("default_voice") and CONFIG["default_voice"] in local_voices:
            selected_voice = CONFIG["default_voice"]
        else:
            selected_voice = local_voices[0]
        
        wav_path = os.path.join(CONFIG["local_voices_dir"], selected_voice, "voice.wav")
        txt_path = os.path.join(CONFIG["local_voices_dir"], selected_voice, "text.txt")
        
        if os.path.exists(wav_path) and os.path.exists(txt_path):
            print(f"\n🎤 Intentando clonar voz por defecto: {selected_voice}")
            
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
                
                entry = model_registry[current_model_id_var[0]]
                model = entry["model"]
                
                clone_prompt_var[0] = await asyncio.to_thread(
                    model.create_voice_clone_prompt,
                    ref_audio=wav_path,
                    ref_text=ref_text,
                    x_vector_only_mode=False
                )
                
                print(f"✅ Voz '{selected_voice}' clonada correctamente y aplicada por defecto")
            except Exception as e:
                print(f"Error creando voz clonada (continuar sin voice cloning): {e}")
        else:
            print(f"\nFaltan archivos para voz '{selected_voice}'")
    
    print("\n" + "."*30)
    print("Escuchando...")
    print("."*30 + "\n")
    print(f"INFO:     WebUI disponible en: http://localhost:{CONFIG['port']}/webui")
    print(f"INFO:     Documentación de la API: http://localhost:{CONFIG['port']}/webui/docs\n")


# Punto de entrada principal
if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
