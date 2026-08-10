#!/usr/bin/env python3
"""Autoridad única sobre el ciclo de vida de los modelos TTS."""

import os
import gc
import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

import torch
from qwen_tts import Qwen3TTSModel

from config.settings import settings
from services.config_service import resolve_device
from utils.gpu import get_dtype

logger = logging.getLogger("tts")


class ModelState(str, Enum):
    """Estados explícitos del modelo activo."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    UNLOADING = "unloading"
    ERROR = "error"


# Mensaje público: nunca se exponen detalles internos al cliente
GPU_OOM_MESSAGE = "Not enough GPU memory to process this request."


class GPUOutOfMemoryError(RuntimeError):
    """Error controlado de memoria de GPU (CUDA OOM).

    Se eleva tras capturar torch.cuda.OutOfMemoryError para que la capa HTTP
    pueda devolver una respuesta controlada sin filtrar detalles internos.
    """

    def __init__(self):
        super().__init__(GPU_OOM_MESSAGE)


@dataclass(frozen=True)
class ModelInfo:
    """Información pública de un modelo cargado.

    No expone la instancia del modelo: solo ModelManager la manipula.
    """
    model_id: str
    model_type: str


# Método de generación requerido para considerar el modelo válido (READY)
_GENERATION_METHODS = {
    "voice_design": "generate_voice_design",
    "base": "generate_voice_clone",
}
_DEFAULT_GENERATION_METHOD = "generate_custom_voice"


def _resolve_model_type(model) -> str:
    """Resolver el tipo de modelo (custom_voice, voice_design, base...)."""
    return getattr(
        getattr(model, "model", None),
        "tts_model_type",
        getattr(model, "tts_model_type", "unknown"),
    )


def _required_generation_method(model_type: str) -> str:
    return _GENERATION_METHODS.get(model_type, _DEFAULT_GENERATION_METHOD)


class ModelManager:
    """Única autoridad sobre el ciclo de vida de los modelos.

    Gestiona carga, descarga, cambio y consulta del modelo activo, evita
    cargas duplicadas, libera VRAM y controla los estados explícitos:
    UNLOADED, LOADING, READY, GENERATING, UNLOADING, ERROR.

    Nunca se indica READY hasta que el modelo ha terminado de cargarse
    y puede realizar una inferencia válida.
    """

    def __init__(self):
        # model_id -> {"model", "type", "state", "error", "loaded_at", "device", "dtype"}
        self._registry: dict = {}
        self._active_id = None
        # Último modelo que estuvo activo (para restaurarlo tras descargas
        # automáticas de VRAM, p.ej. al ceder la GPU a Whisper)
        self._last_active_id = None
        # Cargas en curso: model_id -> Future. Coalesce peticiones simultáneas
        # del mismo modelo para no crear nunca dos instancias en GPU. Cada
        # future se resuelve con (ok, ModelInfo|None, Exception|None). La
        # entrada queda hasta la siguiente carga del mismo modelo.
        self._pending_loads: dict = {}

    # -- Consultas ---------------------------------------------------------

    async def get_active_model(self) -> ModelInfo | None:
        """Modelo activo listo para inferencia, o None."""
        entry = self._registry.get(self._active_id)
        if entry is None or entry["state"] != ModelState.READY.value:
            return None
        return ModelInfo(self._active_id, entry["type"])

    async def get_loaded_models(self) -> list:
        """Modelos actualmente en memoria (READY)."""
        return [
            ModelInfo(mid, e["type"])
            for mid, e in self._registry.items()
            if e["state"] == ModelState.READY.value
        ]

    async def get_model_status(self) -> dict:
        """Estado completo del modelo activo (o del último intento)."""
        entry = self._registry.get(self._active_id)
        if entry is None:
            return {
                "model_id": None,
                "type": None,
                "state": ModelState.UNLOADED.value,
                "device": None,
                "dtype": None,
                "loaded_at": None,
                "error": None,
            }
        return {
            "model_id": self._active_id,
            "type": entry.get("type"),
            "state": entry["state"],
            "device": entry.get("device"),
            "dtype": entry.get("dtype"),
            "loaded_at": entry.get("loaded_at"),
            "error": entry.get("error"),
        }

    def is_loaded(self) -> bool:
        """¿Hay un modelo activo listo para inferencia?"""
        entry = self._registry.get(self._active_id)
        return entry is not None and entry["state"] == ModelState.READY.value

    def list_local_models(self) -> list:
        """Modelos disponibles en el directorio local (no requieren carga)."""
        try:
            return sorted(
                d for d in os.listdir(settings.paths.models_dir)
                if os.path.isdir(os.path.join(settings.paths.models_dir, d))
            )
        except OSError as e:
            logger.warning(f"Error leyendo directorio de modelos: {e}")
            return []

    # -- Ciclo de vida -----------------------------------------------------

    async def load_model(self, model_id: str) -> ModelInfo:
        """Cargar un modelo local y activarlo.

        Estados: UNLOADED -> LOADING -> READY (o ERROR si falla).
        Si ya está READY en memoria solo lo activa (sin cargas duplicadas).

        Si el modelo ya se está cargando (peticiones simultáneas), espera a
        que termine esa carga: nunca se crean dos instancias del mismo modelo
        en GPU. Al terminar devuelve el mismo resultado o propaga el error.
        """
        model_id = (model_id or "").strip()
        if not model_id:
            raise ValueError("model_id vacío")

        entry = self._registry.get(model_id)
        if entry is not None and entry["state"] == ModelState.READY.value:
            logger.info(f"Modelo {model_id} ya cargado en memoria")
            self._active_id = model_id
            return ModelInfo(model_id, entry["type"])

        # Carga en curso para este modelo: esperarla (coalescing). El future
        # nunca se elimina mientras los esperadores puedan consultarlo: se
        # resuelve con (ok, ModelInfo|None, Exception|None) y la entrada se
        # reemplaza en la siguiente carga del mismo modelo.
        pending = self._pending_loads.get(model_id)
        if pending is not None and not pending.done():
            logger.info(f"Modelo {model_id} ya se está cargando; esperando...")
            ok, result, error = await pending
            if ok:
                self._active_id = model_id
                return result
            raise error

        model_path = os.path.join(settings.paths.models_dir, model_id)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo '{model_id}' no existe en {model_path}")

        # Preparar la entrada y marcarla como LOADING
        previous_error = entry.get("error") if entry is not None else None
        if entry is None:
            entry = self._new_entry()
            self._registry[model_id] = entry
        else:
            self._free_entry(entry)  # descartar instancia anterior (ej. ERROR)
        entry.update(
            model=None,
            type=None,
            state=ModelState.LOADING.value,
            error=None,
            loaded_at=None,
            device=None,
            dtype=None,
        )
        self._active_id = model_id

        # Registrar la carga en curso ANTES del primer await (reemplaza una
        # entrada anterior ya resuelta: success/error del intento previo)
        future = asyncio.get_running_loop().create_future()
        self._pending_loads[model_id] = future

        # Liberar VRAM de otros modelos (nunca del que se está cargando)
        self._free_all(except_id=model_id)
        logger.info(f"Cargando modelo: {model_id} desde {model_path}")

        try:
            def _load():
                dtype = get_dtype()
                device = resolve_device()
                logger.info(f"Cargando en dispositivo: {device} (dtype: {dtype})")
                model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=device,
                    dtype=dtype,
                )
                return model, device, dtype

            model, device, dtype = await asyncio.to_thread(_load)

            # Validar que el modelo puede realizar inferencia antes de READY
            model_type = _resolve_model_type(model)
            method = _required_generation_method(model_type)
            if not hasattr(model, method):
                raise RuntimeError(
                    f"El modelo '{model_id}' no expone el método de generación "
                    f"'{method}' (tipo: {model_type})"
                )

            entry["model"] = model
            entry["type"] = model_type
            entry["device"] = device
            entry["dtype"] = self._dtype_name(dtype)
            entry["loaded_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            entry["state"] = ModelState.READY.value
            logger.info(f"Modelo cargado correctamente. Tipo: {model_type}")
            info = ModelInfo(model_id, model_type)

        except asyncio.CancelledError:
            # Una carga cancelada (p.ej. shutdown o cliente desconectado) no
            # puede dejar el estado atascado en LOADING: si era un reintento,
            # se restaura el error anterior; si no, queda "carga cancelada".
            logger.warning(f"Carga del modelo {model_id} cancelada")
            cancel_error = previous_error or "Carga cancelada"
            entry["state"] = ModelState.ERROR.value
            entry["error"] = cancel_error
            if not future.done():
                future.set_result((False, None, RuntimeError(cancel_error)))
            raise

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM cargando modelo {model_id}: {e}")
            entry["state"] = ModelState.ERROR.value
            entry["error"] = GPU_OOM_MESSAGE
            if not future.done():
                future.set_result((False, None, GPUOutOfMemoryError()))
            self._free_cache()
            raise GPUOutOfMemoryError() from e

        except Exception as e:
            logger.error(f"Error cargando modelo {model_id}: {e}")
            entry["state"] = ModelState.ERROR.value
            entry["error"] = str(e)
            if not future.done():
                future.set_result((False, None, e))
            raise  # el originador propaga el error original

        if not future.done():
            future.set_result((True, info, None))
        return info

    async def switch_model(self, model_id: str) -> ModelInfo:
        """Activar otro modelo.

        Si ya está READY en memoria solo cambia el activo; si no, lo carga
        (descargando el actual para liberar VRAM).
        """
        model_id = (model_id or "").strip()
        entry = self._registry.get(model_id)
        if entry is not None and entry["state"] == ModelState.READY.value:
            self._active_id = model_id
            return ModelInfo(model_id, entry["type"])
        return await self.load_model(model_id)

    async def unload_model(self, model_id: str) -> None:
        """Descargar un modelo concreto y liberar su VRAM (no-op si no existe).

        Estados: READY/ERROR -> UNLOADING -> UNLOADED.
        """
        entry = self._registry.get(model_id)
        if entry is None:
            logger.info(f"Modelo {model_id} no estaba cargado")
            return

        entry["state"] = ModelState.UNLOADING.value
        self._registry.pop(model_id, None)
        if self._active_id == model_id:
            self._last_active_id = model_id
            self._active_id = None
        self._free_entry(entry)
        logger.info(f"Modelo descargado: {model_id}")

    def last_active_id(self) -> str | None:
        """Último modelo que estuvo activo (o None si nunca hubo)."""
        return self._last_active_id

    # -- Inferencia (solo ModelManager toca la instancia del modelo) -------

    async def create_voice_clone_prompt(self, wav_path: str, txt_path: str):
        """Crear el prompt de voz clonada del modelo activo (GENERATING)."""
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        return await self._infer(
            "create_voice_clone_prompt",
            {"ref_audio": wav_path, "ref_text": ref_text, "x_vector_only_mode": False},
        )

    async def generate_voice_design(self, **kwargs) -> tuple:
        """Generar audio con voice design usando el modelo activo."""
        return await self._infer("generate_voice_design", kwargs)

    async def generate_voice_clone(self, **kwargs) -> tuple:
        """Generar audio con voice cloning usando el modelo activo."""
        return await self._infer("generate_voice_clone", kwargs)

    async def generate_custom_voice(self, **kwargs) -> tuple:
        """Generar audio con voz por defecto usando el modelo activo."""
        return await self._infer("generate_custom_voice", kwargs)

    # -- Internos ----------------------------------------------------------

    @staticmethod
    def _new_entry() -> dict:
        return {
            "model": None,
            "type": None,
            "state": ModelState.UNLOADED.value,
            "error": None,
            "loaded_at": None,
            "device": None,
            "dtype": None,
        }

    @staticmethod
    def _dtype_name(dtype) -> str:
        """Nombre corto del dtype (bfloat16, float16, float32...)."""
        s = str(dtype)
        if s.startswith("torch."):
            s = s[len("torch."):]
        return s

    async def _infer(self, method_name: str, kwargs: dict) -> tuple:
        """Ejecutar inferencia sobre el modelo activo (READY -> GENERATING -> READY/ERROR)."""
        entry = self._registry.get(self._active_id)
        if entry is None or entry["state"] != ModelState.READY.value:
            raise ValueError("No hay modelo cargado listo para inferencia")
        model = entry["model"]
        entry["state"] = ModelState.GENERATING.value
        try:
            result = await asyncio.to_thread(getattr(model, method_name), **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            # CUDA OOM: estado controlado, limpiar referencias y cache, y
            # elevar un error tipado que la capa HTTP traduce sin filtrar
            # detalles internos.
            logger.error(f"CUDA OOM en inferencia ({method_name})")
            result = None
            kwargs = None
            entry["state"] = ModelState.ERROR.value
            entry["error"] = GPU_OOM_MESSAGE
            self._free_cache()
            raise GPUOutOfMemoryError() from e
        except Exception as e:
            logger.error(f"Error en inferencia ({method_name}): {e}")
            entry["state"] = ModelState.ERROR.value
            entry["error"] = str(e)
            raise
        entry["state"] = ModelState.READY.value
        return result

    @staticmethod
    def _free_cache():
        """Liberar caché de CUDA tras un fallo de GPU (OOM)."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _free_entry(self, entry: dict):
        """Eliminar referencias de un modelo y liberar VRAM."""
        try:
            entry.pop("model", None)
        except Exception as e:
            logger.warning(f"Error liberando modelo: {e}")
        self._free_cache()

    def _free_all(self, except_id=None):
        """Liberar VRAM de todos los modelos salvo except_id."""
        logger.info("Liberando VRAM...")
        for mid in list(self._registry.keys()):
            if mid == except_id:
                continue
            entry = self._registry.pop(mid, None)
            if entry is not None:
                self._free_entry(entry)
        if self._active_id not in self._registry:
            self._active_id = None
