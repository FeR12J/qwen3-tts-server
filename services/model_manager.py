#!/usr/bin/env python3
"""Autoridad única sobre el ciclo de vida de los modelos TTS."""

import os
import gc
import asyncio
import logging
from dataclasses import dataclass

import torch
from qwen_tts import Qwen3TTSModel

from config.settings import CONFIG
from services.config_service import resolve_device
from utils.gpu import get_dtype

logger = logging.getLogger("tts")


@dataclass(frozen=True)
class ModelInfo:
    """Información pública de un modelo cargado.

    No expone la instancia del modelo: solo ModelManager la manipula.
    """
    model_id: str
    model_type: str


class ModelManager:
    """Única autoridad sobre el ciclo de vida de los modelos.

    Gestiona carga, descarga, cambio y consulta del modelo activo, evita
    cargas duplicadas, libera VRAM, controla estados y coordina las
    operaciones de inferencia sobre la GPU.
    """

    def __init__(self):
        self._registry: dict = {}  # model_id -> {"model": ..., "type": ...}
        self._active_id = None

    # -- Consultas ---------------------------------------------------------

    async def get_active_model(self) -> ModelInfo | None:
        """Modelo activo, o None si no hay ninguno cargado."""
        if self._active_id is None:
            return None
        entry = self._registry.get(self._active_id)
        if entry is None:
            return None
        return ModelInfo(self._active_id, entry["type"])

    async def get_loaded_models(self) -> list:
        """Modelos actualmente en memoria."""
        return [ModelInfo(mid, e["type"]) for mid, e in self._registry.items()]

    def is_loaded(self) -> bool:
        """¿Hay un modelo activo?"""
        return self._active_id is not None and self._active_id in self._registry

    def list_local_models(self) -> list:
        """Modelos disponibles en el directorio local (no requieren carga)."""
        try:
            return sorted(
                d for d in os.listdir(CONFIG["local_models_dir"])
                if os.path.isdir(os.path.join(CONFIG["local_models_dir"], d))
            )
        except OSError as e:
            logger.warning(f"Error leyendo directorio de modelos: {e}")
            return []

    # -- Ciclo de vida -----------------------------------------------------

    async def load_model(self, model_id: str) -> ModelInfo:
        """Cargar un modelo local y activarlo.

        Si ya está en memoria solo lo activa (sin cargas duplicadas). Si hay
        otros modelos en memoria, se descargan para liberar VRAM.
        """
        model_id = (model_id or "").strip()
        if not model_id:
            raise ValueError("model_id vacío")

        if model_id in self._registry:
            logger.info(f"Modelo {model_id} ya cargado en memoria")
            self._active_id = model_id
            return ModelInfo(model_id, self._registry[model_id]["type"])

        model_path = os.path.join(CONFIG["local_models_dir"], model_id)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo '{model_id}' no existe en {model_path}")

        # Cargar uno nuevo: liberar VRAM primero
        self._free_all()
        logger.info(f"Cargando modelo: {model_id} desde {model_path}")

        try:
            def _load():
                dtype = get_dtype()
                device = resolve_device()
                logger.info(f"Cargando en dispositivo: {device} (dtype: {dtype})")
                return Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=device,
                    dtype=dtype,
                )

            model = await asyncio.to_thread(_load)

            model_type = getattr(
                getattr(model, "model", None),
                "tts_model_type",
                getattr(model, "tts_model_type", "unknown"),
            )
            logger.info(f"Modelo cargado correctamente. Tipo: {model_type}")

            self._registry[model_id] = {"model": model, "type": model_type}
            self._active_id = model_id
            return ModelInfo(model_id, model_type)

        except Exception as e:
            logger.error(f"Error cargando modelo {model_id}: {e}")
            raise

    async def switch_model(self, model_id: str) -> ModelInfo:
        """Activar otro modelo.

        Si ya está en memoria solo cambia el activo; si no, lo carga
        (descargando el actual para liberar VRAM).
        """
        model_id = (model_id or "").strip()
        if model_id == self._active_id and self.is_loaded():
            return await self.get_active_model()
        if model_id in self._registry:
            self._active_id = model_id
            return ModelInfo(model_id, self._registry[model_id]["type"])
        return await self.load_model(model_id)

    async def unload_model(self, model_id: str) -> None:
        """Descargar un modelo concreto y liberar su VRAM (no-op si no existe)."""
        entry = self._registry.pop(model_id, None)
        if entry is None:
            logger.info(f"Modelo {model_id} no estaba cargado")
            return
        if self._active_id == model_id:
            self._active_id = None
        self._free_entry(entry)
        logger.info(f"Modelo descargado: {model_id}")

    # -- Inferencia (solo ModelManager toca la instancia del modelo) -------

    async def create_voice_clone_prompt(self, wav_path: str, txt_path: str):
        """Crear el prompt de voz clonada del modelo activo."""
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        model = self._active_model()
        return await asyncio.to_thread(
            model.create_voice_clone_prompt,
            ref_audio=wav_path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    async def generate_voice_design(self, **kwargs) -> tuple:
        """Generar audio con voice design usando el modelo activo."""
        return await asyncio.to_thread(self._active_model().generate_voice_design, **kwargs)

    async def generate_voice_clone(self, **kwargs) -> tuple:
        """Generar audio con voice cloning usando el modelo activo."""
        return await asyncio.to_thread(self._active_model().generate_voice_clone, **kwargs)

    async def generate_custom_voice(self, **kwargs) -> tuple:
        """Generar audio con voz por defecto usando el modelo activo."""
        return await asyncio.to_thread(self._active_model().generate_custom_voice, **kwargs)

    # -- Internos ----------------------------------------------------------

    def _active_model(self):
        """Instancia del modelo activo (uso interno exclusivo del manager)."""
        if not self.is_loaded():
            raise ValueError("No hay modelo cargado")
        return self._registry[self._active_id]["model"]

    def _free_entry(self, entry: dict):
        """Eliminar referencias de un modelo y liberar VRAM."""
        try:
            entry.pop("model", None)
            entry.pop("type", None)
        except Exception as e:
            logger.warning(f"Error liberando modelo: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _free_all(self):
        """Liberar VRAM de todos los modelos en memoria."""
        logger.info("Liberando VRAM...")
        for entry in self._registry.values():
            self._free_entry(entry)
        self._registry.clear()
        self._active_id = None
