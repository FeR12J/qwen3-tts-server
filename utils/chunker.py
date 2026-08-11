#!/usr/bin/env python3
"""Componente de división de textos largos (chunking) para TTS.

Divide un texto en fragmentos que respetan, en orden de prioridad:

1. Párrafos (nunca se corta un párrafo si cabe en el fragmento).
2. Frases (nunca se corta una frase si cabe en el fragmento).
3. Palabras (solo cuando un fragmento debe partirse: corte por espacios).
4. Caracteres (último recurso: solo una palabra sin espacios lo exige).

Modos configurables:

- ``sentence``: cada fragmento es un grupo de frases completas del mismo
  párrafo (los párrafos nunca se mezclan dentro de un fragmento). Es el modo
  recomendado para chunked streaming: el audio llega frase a frase.
- ``paragraph``: cada fragmento es uno o más párrafos completos; solo los
  párrafos que exceden el tamaño máximo se subdividen por frases/palabras.

``max_characters`` es la longitud máxima del texto de entrada: si se excede,
chunk() lanza TextChunkerError. ``chunk_size`` es la longitud máxima de cada
fragmento de salida.

IMPORTANTE (streaming por fragmentos):
- Each chunk is synthesized independently.
- Chunking reduces memory usage and enables incremental delivery,
  but does not preserve acoustic/prosodic state between chunks.
- El modelo se reinicia entre fragmentos: la voz/entonação es coherente
  dentro de cada chunk, pero no hay continuidad prosódica entre ellos.
  No es "true streaming" continuo, sino "chunked streaming".

``chunk_size`` es un trade-off CALIDAD <-> LATENCIA <-> VRAM (no solo
rendimiento):

- Larger chunk_size generally improves:
  - voice consistency
  - prosodic continuity
  - sentence-level intonation
- Smaller chunk_size generally improves:
  - time to first audio chunk
  - memory usage

Regla práctica: elegir el chunk_size más grande que cumpla el objetivo de
latencia/VRAM, porque la calidad de continuidad mejora con chunks grandes.
"""

import re
from typing import Literal

from utils.text import split_sentences

ChunkingMode = Literal["sentence", "paragraph"]

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


class TextChunkerError(ValueError):
    """Texto inválido o configuración de chunking inválida."""


class TextChunker:
    """Divide textos largos en fragmentos respetando párrafos y frases."""

    def __init__(
        self,
        max_characters: int = 10000,
        chunking: ChunkingMode = "sentence",
        chunk_size: int = 1000,
    ):
        if chunking not in ("sentence", "paragraph"):
            raise TextChunkerError(
                f"chunking inválido: '{chunking}'. Válidos: sentence, paragraph"
            )
        if max_characters <= 0 or chunk_size <= 0:
            raise TextChunkerError("max_characters y chunk_size deben ser > 0")
        self.max_characters = max_characters
        self.chunking = chunking
        self.chunk_size = chunk_size

    # -- API pública -------------------------------------------------------

    def chunk(self, text: str) -> list:
        """Dividir el texto en fragmentos (list[str]), sin cortes artificiales.

        Lanza TextChunkerError si el texto excede max_characters.
        """
        text = (text or "").strip()
        if not text:
            return []
        if len(text) > self.max_characters:
            raise TextChunkerError(
                f"Texto demasiado largo: {len(text)} caracteres "
                f"(máximo {self.max_characters})"
            )

        paragraphs = self._split_paragraphs(text)

        if self.chunking == "paragraph":
            units = []
            for para in paragraphs:
                if len(para) <= self.chunk_size:
                    units.append(para)
                else:
                    units.extend(split_sentences(para, max_chars=self.chunk_size))
            return self._pack(units, joiner="\n\n")

        # Modo "sentence": los párrafos son fronteras duras de empaquetado.
        chunks = []
        for para in paragraphs:
            sentences = split_sentences(para, max_chars=self.chunk_size)
            chunks.extend(self._pack(sentences, joiner=" "))
        return chunks

    # -- Internos ----------------------------------------------------------

    @staticmethod
    def _split_paragraphs(text: str) -> list:
        """Párrafos del texto (separados por una o más líneas en blanco)."""
        return [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text) if p.strip()]

    def _pack(self, units: list, joiner: str = " ") -> list:
        """Empaquetar unidades (frases o párrafos) en fragmentos <= chunk_size.

        Las unidades ya vienen limitadas a chunk_size por el splitter; este
        método solo agrupa las que caben juntas, sin partirlas. ``joiner``
        preserva la estructura entre unidades (espacio entre frases, salto
        de párrafo entre párrafos).
        """
        chunks = []
        buffer = ""
        for unit in units:
            if not unit:
                continue
            if len(unit) > self.chunk_size:
                # Defensa: unidad mayor que el máximo (palabra única sin
                # espacios). Corte por palabras, último recurso por caracteres.
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._wrap(unit))
                continue
            candidate = (buffer + joiner + unit).strip() if buffer else unit
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = unit
        if buffer:
            chunks.append(buffer)
        return chunks

    def _wrap(self, unit: str) -> list:
        """Cortar una unidad por palabras (último recurso: caracteres)."""
        pieces = []
        while len(unit) > 0:
            cut = unit.rfind(" ", 0, min(len(unit), self.chunk_size))
            if cut <= 0:
                # palabra sin espacios: corte por caracteres
                cut = min(len(unit), self.chunk_size)
            piece = unit[:cut].strip()
            if piece:
                pieces.append(piece)
            unit = unit[cut:].strip()
        return pieces
