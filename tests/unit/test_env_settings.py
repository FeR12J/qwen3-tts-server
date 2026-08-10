#!/usr/bin/env python3
"""Tests de las variables de entorno planas QWEN_TTS_*.

El singleton `settings` se construye al importar config.settings, por lo que
los tests lanzan un subproceso con las variables ya en el entorno.
"""

import os
import subprocess
import sys

CHECK_CODE = '''
from config.settings import settings
from services.config_service import load_runtime_config
load_runtime_config()
checks = {
    "server.host": settings.server.host,
    "server.port": settings.server.port,
    "runtime.device": settings.runtime.device,
    "runtime.dtype": settings.runtime.dtype,
    "tts.default_model": settings.tts.default_model,
    "runtime.api_keys_enabled": settings.runtime.api_keys_enabled,
    "paths.voices_dir": settings.paths.voices_dir,
    "paths.audios_dir": settings.paths.audios_dir,
}
import json
print(json.dumps(checks, ensure_ascii=False))
'''


def _run_with_env(env_vars: dict) -> dict:
    import json
    env = dict(os.environ)
    for key in ("QWEN_TTS_HOST", "QWEN_TTS_PORT", "QWEN_TTS_DEVICE", "QWEN_TTS_DTYPE",
                "QWEN_TTS_MODEL", "QWEN_TTS_REQUIRE_API_KEY", "QWEN_TTS_VOICES_DIR",
                "QWEN_TTS_AUDIO_DIR"):
        env.pop(key, None)
    env.update(env_vars)
    result = subprocess.run(
        [sys.executable, "-c", CHECK_CODE],
        env=env,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_all_flat_env_vars():
    values = _run_with_env({
        "QWEN_TTS_HOST": "0.0.0.0",
        "QWEN_TTS_PORT": "8001",
        "QWEN_TTS_DEVICE": "cuda:0",
        "QWEN_TTS_DTYPE": "bfloat16",
        "QWEN_TTS_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "QWEN_TTS_REQUIRE_API_KEY": "true",
        "QWEN_TTS_VOICES_DIR": "./voices",
        "QWEN_TTS_AUDIO_DIR": "./audios",
    })
    assert values["server.host"] == "0.0.0.0"
    assert values["server.port"] == 8001
    assert values["runtime.device"] == "cuda:0"
    assert values["runtime.dtype"] == "bfloat16"
    assert values["tts.default_model"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert values["runtime.api_keys_enabled"] is True
    assert values["paths.voices_dir"] == "./voices"
    assert values["paths.audios_dir"] == "./audios"


def test_env_types_parsed():
    values = _run_with_env({
        "QWEN_TTS_PORT": "9999",
        "QWEN_TTS_REQUIRE_API_KEY": "false",
    })
    assert values["server.port"] == 9999
    assert values["runtime.api_keys_enabled"] is False


def test_defaults_without_env():
    values = _run_with_env({})
    assert values["server.port"] == 8001
    assert values["runtime.device"] == "auto"
    assert values["runtime.dtype"] == "auto"
    assert values["tts.default_model"] == "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert values["runtime.api_keys_enabled"] is False
