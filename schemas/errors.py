#!/usr/bin/env python3
"""Esquemas de errores de la API."""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Cuerpo estándar de respuesta de error."""
    detail: str
