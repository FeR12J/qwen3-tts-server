#!/usr/bin/env python3
"""Gestión de VRAM compartida entre el modelo TTS y Whisper.

En GPUs pequeñas solo cabe un modelo a la vez:

    TTS request        -> Qwen3-TTS loaded, Whisper unloaded
    Whisper request    -> Qwen3-TTS unloaded, Whisper loaded

En GPUs grandes, con ``unload_tts_for_whisper=false`` y
``unload_whisper_for_tts=false``, ambos modelos se mantienen cargados.

Todas las funciones se ejecutan dentro del inference_lock de QueueService,
de modo que no pueden interrumpir una generación en curso.
"""

import logging

from services.config_service import get_runtime_config

logger = logging.getLogger("tts")


async def prepare_for_tts(models, voices, whisper_service, restore_model: bool = True,
                          queue=None):
    """Liberar VRAM antes de usar el modelo TTS.

    - Descarga Whisper si la config ``unload_whisper_for_tts`` lo exige.
    - Si ``restore_model`` y el modelo TTS no está cargado, restaura el
      último modelo activo (descargado al ceder la GPU a Whisper).

    Con ``max_parallel_inference > 1``, si hay OTRA inferencia en curso se
    omite el intercambio de modelos (descargar Whisper con una transcripción
    activa lo dejaría a medio usar).
    """
    rc = get_runtime_config()
    if queue is not None and queue.inference_active > 1:
        # Con otra inferencia en curso no se intercambian modelos: descargar
        # Whisper con una transcripción activa (o restaurar el TTS bajo una
        # transcripción en una GPU pequeña) es inseguro. Si el modelo TTS no
        # está cargado, la petición fallará con un error claro y se reintenta.
        logger.warning(
            "VRAM: se omite el intercambio TTS/Whisper porque hay otra "
            "inferencia en curso (max_parallel_inference > 1)"
        )
        return
    if rc.get("unload_whisper_for_tts", True) and whisper_service.unload_if_loaded():
        logger.info("VRAM liberada: Whisper descargado antes de usar el modelo TTS")

    if restore_model and not models.is_loaded():
        last = models.last_active_id()
        if last is not None:
            logger.info(f"Restaurando modelo TTS tras ceder la GPU: {last}")
            try:
                # Ya estamos serializados por el inference_lock de la ruta;
                # load_model gestiona internamente el coalescing de cargas.
                await models.load_model(last)
            except Exception as e:
                logger.warning(f"No se pudo restaurar el modelo TTS '{last}': {e}")


async def prepare_for_whisper(models, voices, whisper_service, queue=None):
    """Liberar VRAM antes de transcribir.

    Descarga el modelo TTS si la config ``unload_tts_for_whisper`` lo exige,
    para dejar sitio al modelo Whisper (carga lazy al transcribir).

    Con ``max_parallel_inference > 1``, si hay OTRA inferencia en curso se
    omite la descarga (podría ser una generación TTS usando ese modelo).
    """
    rc = get_runtime_config()
    if not rc.get("unload_tts_for_whisper", True):
        return
    if queue is not None and queue.inference_active > 1:
        logger.warning(
            "VRAM: se omite la descarga del modelo TTS para Whisper porque "
            "hay otra inferencia en curso (max_parallel_inference > 1)"
        )
        return
    active = await models.get_active_model()
    if active is None:
        return
    logger.info(f"VRAM liberada: descargando modelo TTS '{active.model_id}' antes de Whisper")
    try:
        await models.unload_model(active.model_id)
        voices.unload_voice()
    except Exception as e:
        logger.warning(f"No se pudo descargar el modelo TTS antes de Whisper: {e}")
