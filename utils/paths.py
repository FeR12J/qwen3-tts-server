#!/usr/bin/env python3
"""Rutas y directorios del proyecto."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
AUDIOS_DIR = os.path.join(BASE_DIR, "audios")
CLONE_PROMPTS_DIR = os.path.join(BASE_DIR, "clone_prompts")
DATA_DIR = os.path.join(BASE_DIR, "data")
WEBUI_DIR = os.path.join(BASE_DIR, "webui")
