#!/usr/bin/env python3
"""Tests unitarios del divisor de frases (utils.text.split_sentences)."""

from utils.text import split_sentences


def test_split_basic():
    assert split_sentences("Hola mundo. Esto es una prueba. Y otra.") == [
        "Hola mundo.",
        "Esto es una prueba.",
        "Y otra.",
    ]


def test_split_questions_exclamations():
    assert split_sentences("¿Qué tal? Bien. ¡Genial!") == [
        "¿Qué tal?",
        "Bien.",
        "¡Genial!",
    ]


def test_abbreviations_not_split():
    assert split_sentences("El Sr. Pérez vino. Hola") == [
        "El Sr. Pérez vino.",
        "Hola",
    ]


def test_decimal_not_split():
    assert split_sentences("El valor es 3.14 y sigue. Fin.") == [
        "El valor es 3.14 y sigue.",
        "Fin.",
    ]


def test_year_boundary():
    assert split_sentences("Año 2024. Sigue el texto.") == [
        "Año 2024.",
        "Sigue el texto.",
    ]


def test_newline_is_boundary():
    assert split_sentences("Línea uno.\nLínea dos.") == [
        "Línea uno.",
        "Línea dos.",
    ]


def test_quotes_keep_sentence():
    assert split_sentences('"¿Quién viene?", dijo. Luego se fue.') == [
        '"¿Quién viene?", dijo.',
        "Luego se fue.",
    ]


def test_max_chars_hard_split():
    parts = split_sentences("palabra " * 100, max_chars=50)
    assert len(parts) > 2
    assert all(len(p) <= 50 for p in parts)


def test_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
