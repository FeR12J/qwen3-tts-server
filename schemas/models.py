#!/usr/bin/env python3
"""Esquemas de las solicitudes de modelos."""

from pydantic import BaseModel


class LoadModelRequest(BaseModel):
    """Solicitud para cargar modelo."""
    model_id: str
