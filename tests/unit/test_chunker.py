#!/usr/bin/env python3
"""Tests unitarios del componente de chunking (utils.chunker.TextChunker)."""

import pytest

from utils.chunker import TextChunker, TextChunkerError


def test_empty_text():
    assert TextChunker().chunk("") == []
    assert TextChunker().chunk("   ") == []


def test_short_text_single_chunk():
    text = "Hola mundo. Esto es una prueba corta."
    assert TextChunker(max_characters=1000, chunk_size=500).chunk(text) == [text]


def test_sentence_mode_respects_sentences():
    c = TextChunker(max_characters=10000, chunking="sentence", chunk_size=60)
    chunks = c.chunk("Primera frase de prueba. Segunda frase de prueba. Tercera frase.")
    assert all(len(x) <= 60 for x in chunks)
    assert all(x.endswith((".", "!", "?")) or " " not in x for x in chunks)


def test_sentence_mode_no_mixed_paragraphs():
    c = TextChunker(max_characters=10000, chunking="sentence", chunk_size=40)
    text = "Primera frase del primer párrafo. Segunda frase.\n\nSegundo párrafo entero."
    chunks = c.chunk(text)
    assert len(chunks) >= 2
    # ningún fragmento contiene texto de dos párrafos distintos
    for ch in chunks:
        assert "\n\n" not in ch
        assert ("Primera" in ch and "Segundo" in ch) is False


def test_paragraph_mode_keeps_whole_paragraphs():
    c = TextChunker(max_characters=10000, chunking="paragraph", chunk_size=40)
    para1 = "A" * 30 + "."
    para2 = "B" * 30 + "."
    text = f"{para1}\n\n{para2}"
    chunks = c.chunk(text)
    # los dos párrafos no caben juntos (60+1 > 40): cada uno en su fragmento
    assert chunks == [para1, para2]


def test_paragraph_mode_packs_multiple_paragraphs():
    c = TextChunker(max_characters=10000, chunking="paragraph", chunk_size=100)
    paras = [("P" + str(i)) * 10 for i in range(3)]
    text = "\n\n".join(paras)
    chunks = c.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == "\n\n".join(paras)


def test_long_paragraph_subdivided_by_sentences():
    c = TextChunker(max_characters=10000, chunking="paragraph", chunk_size=50)
    para = "Primera frase del párrafo largo. Segunda frase del párrafo largo."
    chunks = c.chunk(para + "\n\nCorto.")
    assert all(len(x) <= 50 for x in chunks)
    assert chunks[-1].endswith("Corto.")


def test_word_boundary_split_within_sentence():
    c = TextChunker(max_characters=10000, chunking="sentence", chunk_size=30)
    text = ("Palabras " * 30).strip() + "."
    chunks = c.chunk(text)
    assert all(len(x) <= 30 for x in chunks)
    # ningún fragmento (salvo el último) corta una palabra a mitad
    for ch in chunks[:-1]:
        assert ch.endswith("Palabras") or ch.endswith(".")


def test_max_characters_limit():
    c = TextChunker(max_characters=50, chunk_size=10)
    with pytest.raises(TextChunkerError):
        c.chunk("x" * 51)
    # justo en el límite pasa
    assert c.chunk("y" * 50)


def test_invalid_mode():
    with pytest.raises(TextChunkerError):
        TextChunker(chunking="tokens")


# -- Modo "none": sin división ---------------------------------------------


def test_none_mode_single_chunk_ignores_chunk_size():
    """Modo none: el texto completo es un único fragmento, aunque supere
    chunk_size (ni frases ni párrafos se dividen)."""
    text = ("Primera frase. Segunda frase." + "\n\n" + "Párrafo dos.")
    c = TextChunker(max_characters=10000, chunking="none", chunk_size=20)
    assert c.chunk(text) == [text]


def test_none_mode_preserves_paragraphs_and_formatting():
    text = "Línea uno.\n\nLínea dos.\nLínea tres."
    c = TextChunker(chunking="none", chunk_size=10)
    assert c.chunk(text) == [text]


def test_none_mode_still_enforces_max_characters():
    c = TextChunker(max_characters=50, chunking="none")
    with pytest.raises(TextChunkerError):
        c.chunk("x" * 51)
    assert c.chunk("y" * 50) == ["y" * 50]


def test_none_mode_empty_text():
    assert TextChunker(chunking="none").chunk("   ") == []


def test_invalid_sizes():
    with pytest.raises(TextChunkerError):
        TextChunker(max_characters=0)
    with pytest.raises(TextChunkerError):
        TextChunker(chunk_size=0)


# -- Casos extremos -------------------------------------------------------


def test_edge_empty_and_whitespace_only():
    c = TextChunker()
    assert c.chunk("") == []
    assert c.chunk(" ") == []
    assert c.chunk("\n") == []
    assert c.chunk("\n\n") == []
    assert c.chunk(" \t\n ") == []


def test_edge_single_short_sentences():
    c = TextChunker()
    assert c.chunk("Hola.") == ["Hola."]
    assert c.chunk("Hola! ¿Cómo estás?") == ["Hola! ¿Cómo estás?"]
    assert c.chunk("¿Qué? ¡Esto funciona!") == ["¿Qué? ¡Esto funciona!"]


def test_edge_decimal_number_not_split():
    c = TextChunker()
    assert c.chunk("1.234,56 €") == ["1.234,56 €"]


def test_edge_url_and_email_not_split():
    c = TextChunker()
    assert c.chunk("www.example.com") == ["www.example.com"]
    assert c.chunk("test@example.com") == ["test@example.com"]


def test_edge_ellipsis_not_split():
    c = TextChunker()
    assert c.chunk("Hello... world") == ["Hello... world"]


def test_edge_chunk_exactly_max_chunk_size():
    cs = 100
    chunks = TextChunker(chunk_size=cs).chunk("A" * cs)
    assert chunks == ["A" * cs]


def test_edge_chunk_just_over_max_chunk_size():
    cs = 100
    text = "A" * (cs + 1)
    chunks = TextChunker(chunk_size=cs).chunk(text)
    assert all(len(x) <= cs for x in chunks)
    assert "".join(chunks) == text
    assert len(chunks) == 2
    assert chunks[0] == "A" * cs


@pytest.mark.parametrize("chunking", ["sentence", "paragraph"])
def test_edge_word_longer_than_chunk_size_fallback_by_characters(chunking):
    """Palabra sin espacios > chunk_size: obliga al fallback por caracteres."""
    cs = 1000
    long_word = "x" * 2500
    chunks = TextChunker(chunking=chunking, chunk_size=cs).chunk(long_word)
    assert all(len(x) <= cs for x in chunks)
    assert "".join(chunks) == long_word
    assert len(chunks) == 3
    assert chunks[0] == "x" * cs
