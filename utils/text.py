#!/usr/bin/env python3
"""Utilidades de procesado de texto."""


def truncate_text(text: str, max_chars: int = 100) -> str:
    """Truncar texto para logs y mensajes cortos."""
    if not text:
        return ""
    return text[:max_chars] + "..." if len(text) > max_chars else text
