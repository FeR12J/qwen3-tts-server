#!/usr/bin/env python3
"""Servicio de gestión de modelos TTS."""

import os
import logging
import asyncio
from qwen_tts import Qwen3TTSModel
from config.settings import CONFIG
from services.config_service import resolve_device
from utils.helpers import clear_models, get_dtype, get_vram_available

logger = logging.getLogger("tts")


async def load_model(model_id: str, model_registry: dict):
    """Cargar un modelo TTS desde el directorio local."""
    
    if model_id in model_registry:
        logger.info(f"Modelo {model_id} ya cargado en memoria")
        return model_registry[model_id]["model"]
    
    model_path = os.path.join(CONFIG["local_models_dir"], model_id)
    if not os.path.exists(model_path):
        raise ValueError(f"Modelo '{model_id}' no existe en {model_path}")
    
    await asyncio.to_thread(clear_models, model_registry)
    logger.info(f"Cargando modelo: {model_id} desde {model_path}")
    
    try:
        # Carga pesada y bloqueante: ejecutar fuera del event loop
        def _load():
            dtype = get_dtype()
            device = resolve_device()
            logger.info(f"Cargando en dispositivo: {device} (dtype: {dtype})")
            return Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device,
                dtype=dtype
            )

        model = await asyncio.to_thread(_load)
        
        model_type = getattr(
            getattr(model, "model", None),
            "tts_model_type",
            getattr(model, "tts_model_type", "unknown"),
        )
        logger.info(f"Modelo cargado correctamente. Tipo: {model_type}")
        
        model_registry[model_id] = {
            "model": model,
            "type": model_type
        }
        
        return model
        
    except Exception as e:
        logger.error(f"Error cargando modelo {model_id}: {e}")
        raise
