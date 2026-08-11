#!/usr/bin/env python3
"""Utilidades de procesado de texto."""

import re

_ABBREVIATIONS = {
    "sr", "sra", "sres", "dr", "dra", "dres", "etc", "ej", "p.ej", "aprox",
    "pág", "vol", "no", "nº", "tel", "lic", "ing", "av", "c", "d", "f", "e",
    "a", "p", "mr", "mrs", "ms", "prof", "st", "jr", "srta", "v", "vs",
}

_DECIMAL_RE = re.compile(r"\d+(?:[.,]\d+)*$")


def truncate_text(text: str, max_chars: int = 100) -> str:
    """Truncar texto para logs y mensajes cortos."""
    if not text:
        return ""
    return text[:max_chars] + "..." if len(text) > max_chars else text


def split_sentences(text: str, max_chars: int = 500) -> list:
    """Dividir texto en frases para streaming TTS.

    - Frontera: ``. ! ? ; …`` seguida de espacio/salto de línea, o un salto
      de línea en sí.
    - No parte abreviaturas comunes ("Sr.", "etc.") ni decimales ("3.14").
    - Trozos muy cortos se fusionan con el siguiente.
    - Frases que exceden ``max_chars`` se parten en el espacio más cercano.
    """
    if not text:
        return []

    sentences = []
    current = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        current += ch

        if ch == "\n" and current.strip():
            sentences.append(current.strip())
            current = ""
        elif ch in ".!?;…" and i + 1 < n and text[i + 1] in " \t\n":
            if not _is_boundary(text, i):
                pass
            else:
                sentences.append(current.strip())
                current = ""
                i += 1  # consumir el espacio que sigue a la puntuación
        i += 1

    if current.strip():
        sentences.append(current.strip())

    sentences = [s for s in sentences if s]
    return _enforce_max_chars(sentences, max_chars)


def _is_boundary(text: str, i: int) -> bool:
    """¿La puntuación en i cierra una frase (no es abreviatura ni decimal)?"""
    ch = text[i]
    if ch != ".":
        return True

    prev_word = _prev_word(text, i)
    if prev_word and prev_word.lower() in _ABBREVIATIONS:
        return False

    # Decimal: "3.14" o "3,14" -> no es frontera. "2024. Sigue" sí lo es
    # (el carácter tras el espacio no es un dígito).
    if i > 0 and text[i - 1].isdigit():
        j = i + 1
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j].isdigit():
            return False
    return True


def _prev_word(text: str, i: int) -> str:
    """Palabra inmediatamente anterior a la posición i."""
    start = i - 1
    while start >= 0 and (text[start].isalnum() or text[start] in ".'’"):
        start -= 1
    return text[start + 1:i]


def _enforce_max_chars(parts: list, max_chars: int) -> list:
    result = []
    for part in parts:
        chunk = part.replace("\n", " ")
        if len(chunk) > max_chars:
            result.extend(_hard_split(chunk, max_chars))
        else:
            result.append(chunk)
    return result


def _hard_split(chunk: str, max_chars: int) -> list:
    pieces = []
    while len(chunk) > max_chars:
        cut = chunk.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(chunk[:cut].strip())
        chunk = chunk[cut:].strip()
    if chunk:
        pieces.append(chunk)
    return pieces
