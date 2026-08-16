#!/usr/bin/env python3
"""Tests del parseo de marcas de tiempo en las transcripciones Whisper."""

import pytest

from config.settings import settings
from security.validation import validate_config_update
from services import whisper_service as ws

# Whisper v3: los ids >= timestamp_begin son tokens de tiempo (20 ms/token)
_TIMESTAMP_BEGIN = 50363


def _ts(seconds: float) -> int:
    """Id de token de tiempo para una marca en segundos."""
    return _TIMESTAMP_BEGIN + int(round(seconds / ws._TIME_PRECISION))


class FakeTokenizer:
    """Tokenizer mínimo con timestamp_begin y decode por mapeo de ids."""

    timestamp_begin = _TIMESTAMP_BEGIN

    def __init__(self, pieces: dict):
        self._pieces = pieces

    def decode(self, token_ids, skip_special_tokens=True):
        out = []
        for tid in token_ids:
            piece = self._pieces.get(tid)
            if piece is not None:
                out.append(piece)
        return "".join(out)


def _pieces(texts: list) -> tuple:
    """Mapear una lista de piezas de texto a ids secuenciales (< timestamp_begin)."""
    mapping = {}
    for i, piece in enumerate(texts):
        mapping[100 + i] = piece
    return mapping


# -- Resolución del modo ----------------------------------------------------


def test_resolve_mode_override_prioritario(monkeypatch):
    monkeypatch.setattr(settings.runtime, "whisper_timestamps", "segment")
    assert ws._resolve_timestamp_mode("word") == "word"
    assert ws._resolve_timestamp_mode("off") == "off"


def test_resolve_mode_usa_config_sin_override(monkeypatch):
    monkeypatch.setattr(settings.runtime, "whisper_timestamps", "segment")
    assert ws._resolve_timestamp_mode(None) == "segment"
    assert ws._resolve_timestamp_mode("") == "segment"


def test_resolve_mode_invalido_cae_a_config(monkeypatch):
    monkeypatch.setattr(settings.runtime, "whisper_timestamps", "segment")
    assert ws._resolve_timestamp_mode("bogus") == "segment"


def test_resolve_mode_config_invalido_cae_a_off(monkeypatch):
    monkeypatch.setattr(settings.runtime, "whisper_timestamps", "bogus")
    assert ws._resolve_timestamp_mode(None) == "off"


def test_resolve_mode_default_off(monkeypatch):
    monkeypatch.setattr(settings.runtime, "whisper_timestamps", "off")
    assert ws._resolve_timestamp_mode(None) == "off"


# -- Segmentos --------------------------------------------------------------


def test_segments_basico():
    m = _pieces(["Hola", " ", "mundo", "mundo"])
    tk = FakeTokenizer(m)
    ids = [_ts(0.0), 100, 101, 102, _ts(2.5), 103, _ts(5.0)]
    segs = ws._extract_segments(tk, ids, duration_seconds=10.0)
    assert segs == [
        {"start": 0.0, "end": 2.5, "text": "Hola mundo"},
        {"start": 2.5, "end": 5.0, "text": "mundo"},
    ]


def test_segments_con_api_nueva_sin_timestamp_begin():
    """Transformers >= 4.50: solo existe timestamp_ids() (sin atributo)."""
    m = _pieces(["Hola", "mundo"])

    class ModernTokenizer(FakeTokenizer):
        @property
        def timestamp_begin(self):
            raise AttributeError("timestamp_begin ya no existe en transformers 4.57")

        def timestamp_ids(self):
            return list(range(_TIMESTAMP_BEGIN, _TIMESTAMP_BEGIN + 1501))

    tk = ModernTokenizer(m)
    ids = [_ts(0.5), 100, 101, _ts(3.0)]
    segs = ws._extract_segments(tk, ids, duration_seconds=10.0)
    assert segs == [{"start": 0.5, "end": 3.0, "text": "Holamundo"}]


def test_segments_ignora_tokens_especiales_iniciales():
    m = _pieces(["Hola"])
    tk = FakeTokenizer(m)
    # startoftranscript, idioma es, task transcribe, texto, cierre
    ids = [50258, 50262, 50359, _ts(0.0), 100, _ts(3.0), 50257]
    segs = ws._extract_segments(tk, ids, duration_seconds=10.0)
    assert segs == [{"start": 0.0, "end": 3.0, "text": "Hola"}]


def test_segments_marcas_consecutivas_no_crean_vacios():
    m = _pieces(["solo"])
    tk = FakeTokenizer(m)
    # 0.0 -> 1.0 sin texto (vacío), luego texto entre 1.0 y 2.0
    ids = [_ts(0.0), _ts(1.0), 100, _ts(2.0)]
    segs = ws._extract_segments(tk, ids, duration_seconds=10.0)
    assert segs == [{"start": 1.0, "end": 2.0, "text": "solo"}]


def test_segments_texto_final_sin_marca_de_cierre():
    m = _pieces(["Hola"])
    tk = FakeTokenizer(m)
    ids = [_ts(1.0), 100]
    segs = ws._extract_segments(tk, ids, duration_seconds=7.5)
    assert segs == [{"start": 1.0, "end": 1.0, "text": "Hola"}]


def test_segments_sin_ninguna_marca():
    m = _pieces(["Hola"])
    tk = FakeTokenizer(m)
    ids = [100]
    segs = ws._extract_segments(tk, ids, duration_seconds=7.5)
    assert segs == [{"start": 0.0, "end": 7.5, "text": "Hola"}]


def test_segments_acepta_tensor():
    import torch

    m = _pieces(["Hola", " ", "mundo"])
    tk = FakeTokenizer(m)
    ids = torch.tensor([_ts(0.0), 100, 101, 102, _ts(5.0)])
    segs = ws._extract_segments(tk, ids, duration_seconds=10.0)
    assert segs == [{"start": 0.0, "end": 5.0, "text": "Hola mundo"}]


# -- Palabras ---------------------------------------------------------------


def test_words_interpolacion_proporcional():
    segs = [{"start": 0.0, "end": 10.0, "text": "Hola mundo"}]
    words = ws._extract_words(segs)
    assert words == [
        {"word": "Hola", "start": 0.0, "end": 4.0},
        {"word": "mundo", "start": 5.0, "end": 10.0},
    ]


def test_words_varios_segmentos():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "uno dos"},
        {"start": 2.0, "end": 6.0, "text": "tres"},
    ]
    words = ws._extract_words(segs)
    # Interpolación proporcional por caracteres: "uno dos" son 7 caracteres
    assert words == [
        {"word": "uno", "start": 0.0, "end": 0.86},
        {"word": "dos", "start": 1.14, "end": 2.0},
        {"word": "tres", "start": 2.0, "end": 6.0},
    ]


def test_words_segmento_vacio_ignorado():
    assert ws._extract_words([{"start": 0.0, "end": 2.0, "text": ""}]) == []


# -- Validación del ajuste de configuración ---------------------------------


def test_validate_whisper_timestamps_valido():
    validate_config_update({"whisper_timestamps": "word"}, None)
    validate_config_update({"whisper_timestamps": "segment"}, None)
    validate_config_update({"whisper_timestamps": "off"}, None)


def test_validate_whisper_timestamps_invalido():
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        validate_config_update({"whisper_timestamps": "frases"}, None)
