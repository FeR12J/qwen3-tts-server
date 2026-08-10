#!/usr/bin/env python3
"""Tests unitarios de utilidades de texto."""

from utils.text import truncate_text


def test_truncate_text_short():
    assert truncate_text("hola") == "hola"


def test_truncate_text_long():
    result = truncate_text("a" * 200, 100)
    assert len(result) == 103
    assert result.endswith("...")


def test_truncate_text_empty():
    assert truncate_text("") == ""
