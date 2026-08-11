#!/usr/bin/env python3
"""Errores unificados de la API.

Todas las respuestas de error tienen el mismo formato JSON:

    {
      "error": {
        "code": "MODEL_NOT_LOADED",
        "message": "The requested TTS model is not loaded.",
        "request_id": "abc123"
      }
    }

El handler global (app.api_error_handler) rellena request_id desde la
cabecera x-request-id o generando uno nuevo.
"""

GPU_OOM_MESSAGE = "Not enough GPU memory to process this request."


class APIError(Exception):
    """Error de API con código, mensaje y estado HTTP unificados."""

    def __init__(self, code: str, message: str, status_code: int = 400,
                 request_id: str = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.request_id = request_id


class ModelNotLoadedError(APIError):
    """Modelo TTS solicitado no está cargado (409)."""

    def __init__(self, message: str = "The requested TTS model is not loaded.",
                 request_id: str = None):
        super().__init__("MODEL_NOT_LOADED", message, 409, request_id)


class ModelLoadingError(APIError):
    """Falló la carga de un modelo TTS (503)."""

    def __init__(self, message: str = "Failed to load the TTS model.",
                 request_id: str = None):
        super().__init__("MODEL_LOADING_ERROR", message, 503, request_id)


class InvalidVoiceError(APIError):
    """Referencia de voz inválida (400)."""

    def __init__(self, message: str = "Invalid voice.", request_id: str = None):
        super().__init__("INVALID_VOICE", message, 400, request_id)


class InvalidAudioError(APIError):
    """Audio de entrada inválido (400)."""

    def __init__(self, message: str = "Invalid audio.", request_id: str = None):
        super().__init__("INVALID_AUDIO", message, 400, request_id)


class GPUOutOfMemoryError(APIError):
    """CUDA OOM: la petición no puede procesarse por falta de VRAM (503)."""

    def __init__(self, message: str = GPU_OOM_MESSAGE, request_id: str = None):
        super().__init__("GPU_OUT_OF_MEMORY", message, 503, request_id)


class QueueFullError(APIError):
    """Cola de inferencia llena (429): reintentar más tarde."""

    def __init__(self, queue_size: int, request_id: str = None):
        super().__init__(
            "QUEUE_FULL",
            f"Queue full ({queue_size} requests waiting). Try again in a few seconds.",
            429,
            request_id,
        )


class AuthenticationError(APIError):
    """Clave API inválida o no proporcionada (401)."""

    def __init__(self, message: str = "Invalid or missing API key.",
                 request_id: str = None):
        super().__init__("AUTHENTICATION_ERROR", message, 401, request_id)