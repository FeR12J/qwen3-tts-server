# Servidor TTS + Transcripción - Qwen3-TTS / Whisper

Servidor FastAPI modular para generación de texto a voz con modelos Qwen3-TTS locales y transcripción de audio con Whisper. Incluye voice cloning, panel de administración web y gestión de claves API.

## Estructura del Proyecto

```
qwen3-tts-server/
├── app.py               # Composición de la aplicación y arranque
├── config/              # Configuración centralizada (Pydantic Settings)
│   └── settings.py      # settings: grupos server/paths/model/whisper/cors/limits/queue/auth/logging/runtime
├── schemas/             # Esquemas Pydantic por dominio
│   ├── tts.py           # TTSRequest, TTSRequestOpenWebUI
│   ├── whisper.py       # WhisperStatusResponse
│   ├── voices.py        # LoadVoiceRequest
│   ├── models.py        # LoadModelRequest
│   ├── system.py        # ConfigUpdate, ApiKeyCreate, SystemStatus
│   └── errors.py        # ErrorResponse
├── routes/              # Endpoints HTTP (sin lógica de inferencia)
│   ├── tts.py           # /tts, /tts/stream, /tts/play, /tts/audio/speech
│   ├── models.py        # /model/load, /model/unload, /models
│   ├── voices.py        # /voice/load, /voice/create, /voice/unload, /voices
│   ├── system.py        # / (estado del servidor)
│   ├── whisper.py       # /transcribe, /transcribe/status, /transcribe/unload
│   ├── auth.py          # Gestión de claves API (/webui/api/apikeys*)
│   └── webui.py         # Panel web, docs y configuración
├── services/            # Lógica de negocio e inferencia
│   ├── model_manager.py # Registro de modelos, carga/descarga, voice clone prompt
│   ├── voice_manager.py # Voces locales y prompt de clonación activo
│   ├── tts_service.py   # Generación TTS (despacho por tipo de modelo)
│   ├── whisper_service.py  # Transcripción con Whisper (transformers)
│   ├── audio_service.py # Codificación WAV, guardado, limpieza y reproducción local
│   ├── queue_service.py # Semáforo de inferencia y serialización de reproducciones
│   ├── metrics_service.py  # Contadores de actividad, logs y VRAM
│   ├── config_service.py   # Configuración en tiempo de ejecución
│   └── apikey_service.py   # Claves API (hash)
├── security/            # Autenticación y validación
│   ├── auth.py          # Dependencia require_api_key (X-API-Key / Bearer)
│   ├── permissions.py   # Modelo cargado, soporte de voice cloning
│   └── validation.py    # validate_text, voice_name, tamaños, config
├── storage/             # Persistencia en disco
│   ├── config_storage.py   # data/runtime.json
│   ├── api_key_storage.py  # data/apikeys.json
│   └── voice_storage.py    # voces locales (voice.wav + text.txt)
├── utils/               # Utilidades
│   ├── paths.py         # Directorios del proyecto
│   ├── gpu.py           # VRAM, dtype, limpieza CUDA, listado de GPUs
│   ├── logging.py       # setup_logging, log_request, rotación
│   └── text.py          # truncate_text
├── tests/               # Tests (unit/ e integration/)
│   ├── unit/
│   └── integration/
├── models/              # Modelos locales (Qwen3-TTS + whisper-large-v3)
├── voices/              # Voces clonadas locales (cada una con voice.wav + text.txt)
├── data/                # Datos persistentes (runtime.json, apikeys.json)
├── webui/               # Panel web (panel.html, docs.html)
├── audios/              # Audios generados
├── requirements.txt
└── requirements-dev.txt
```

Flujo de una petición: HTTP  Autenticación (`security/auth.py`)  Validación de esquema (`schemas/`)  Validación de entrada (`security/validation.py`)  Servicio (`services/`)  `ModelManager`/Whisper  `AudioService`  Respuesta. Las rutas HTTP no contienen lógica de inferencia; todo el acceso al modelo pasa por los servicios.

## Modelos soportados

Cada modelo Qwen3-TTS usa un método de generación distinto (se selecciona automáticamente según el tipo):

| Tipo de modelo | Método de generación | Notas |
|---|---|---|
| `custom_voice` | `generate_custom_voice(text, speaker, ...)` | Voz por locutor (`speaker`, ej. `serena`, `alberto`...) |
| `voice_design` | `generate_voice_design(text, instruct, ...)` | La voz se describe en texto con `instruct` (ej. "voz femenina joven y cálida") |
| `base` | `generate_voice_clone(...)` | Requiere voz de referencia: clonada (`/voice/load`) o automáticamente la primera voz local de `voices/` |

Descargar los modelos en `models/`:

- `Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign` (por defecto)
- `whisper-large-v3` (transcripción)

El tipo de cada modelo se resuelve desde `model.model.tts_model_type` y se muestra en la respuesta de `/model/load`.

## Endpoints

| Nivel | Método | Endpoint | Descripción |
|-------|--------|----------|-------------|
| PUBLIC | GET | `/health` | Estado del servidor (rápido, no carga modelos) |
| PUBLIC | GET | `/ready` | Listo para servir (`ready`, `tts_model_loaded`) |
| PUBLIC | GET | `/system/status` | Estado detallado (GPU, TTS, Whisper) |
| PUBLIC | GET | `/version` | Versión del servidor y de las librerías (diagnóstico) |
| PUBLIC | GET | `/` | Estado del servidor (modelo activo, VRAM) |
| PUBLIC | GET | `/models`, `/tts/audio/models`, `/model/status` | Listar modelos disponibles |
| PUBLIC | GET | `/voices`, `/tts/audio/voices` | Listar voces disponibles |
| PUBLIC | GET | `/transcribe/status` | Estado del modelo Whisper |
| PUBLIC | GET | `/webui`, `/webui/docs` | Panel y documentación |
| PUBLIC | GET | `/webui/api/devices` | GPUs disponibles y dispositivo en uso |
| PROTECTED | POST | `/tts` | Generar TTS (estándar) |
| PROTECTED | POST | `/tts/stream` | TTS en streaming real: audio por frases, WAV o PCM |
| PROTECTED | POST | `/tts/play` | Generar TTS y reproducirlo en este equipo |
| PROTECTED | POST | `/tts/audio/speech` | Generar TTS (compatible OpenWebUI) |
| PROTECTED | POST | `/transcribe` | Transcribir audio a texto con Whisper |
| PROTECTED | POST | `/transcribe/unload` | Descargar Whisper y liberar VRAM |
| ADMIN | POST | `/model/load` | Cargar modelo local |
| ADMIN | POST | `/model/unload` | Descargar modelo y liberar VRAM |
| ADMIN | POST | `/voice/load` | Cargar voz para clonación (solo modelos `base`) |
| ADMIN | POST | `/voice/create` | Subir WAV + transcripción (multipart), la guarda y la clona |
| ADMIN | POST | `/voice/unload` | Desactivar voice cloning |
| ADMIN | GET/POST | `/webui/api/config` | Leer/actualizar configuración en tiempo de ejecución |
| ADMIN | GET/POST | `/webui/api/apikeys` | Listar/crear claves API |
| ADMIN | POST | `/webui/api/apikeys/{id}/toggle` | Activar/desactivar clave API |
| ADMIN | DELETE | `/webui/api/apikeys/{id}` | Eliminar clave API |

## Protección de endpoints

- **PUBLIC** (`/health`, `/version`, estado y listados): sin autenticación.
- **PROTECTED** (`/tts/*`, `/transcribe`): dependencia `require_api_key()`; exige `X-API-Key: qt-...` (o `Authorization: Bearer qt-...`) solo cuando la exigencia global de claves está activada (panel o `QWEN_TTS_REQUIRE_API_KEY=true`).
- **ADMIN** (`/model/*`, `/voice/*`, `/apikeys/*`, `/config/*`): dependencia `require_admin()`; exige **siempre** una clave API válida, esté o no activada la exigencia global, de modo que las operaciones administrativas no quedan accesibles sin autenticación.
- **Bootstrap**: si aún no existe ninguna clave, las operaciones ADMIN quedan abiertas para permitir crear la primera clave desde el panel. A partir de la primera clave, todas las operaciones ADMIN exigen autenticación.

## Claves API

- Las claves se almacenan **únicamente como hash** (SHA-256) en `data/apikeys.json`: `{"id": "key_...", "name": "...", "key_hash": "...", "created_at": "...", "last_used_at": "..."}`. Nunca se guarda la clave en claro.
- La clave completa (`qt-...`) se muestra **una sola vez**, en la respuesta de creación; el panel la guarda localmente en el navegador y permite copiarla en ese momento.
- El listado devuelve solo `id`, `name`, prefijo enmascarado, `created_at` y `last_used_at`.

## Transcripción (Whisper)

El servidor incluye transcripción de audio local con `whisper-large-v3` (transformers). El modelo se carga de forma perezosa en la primera petición y se descarga con `/transcribe/unload` para liberar VRAM.

```bash
# Transcribir (idioma automático)
curl -X POST http://localhost:8001/transcribe \
  -H "X-API-Key: qt-tu-clave" \
  -F "audio=@/ruta/al/audio.mp3"

# Forzar idioma español
curl -X POST http://localhost:8001/transcribe \
  -F "audio=@/ruta/al/audio.wav" -F "language=es"
```

Respuesta: `{"status":"ok","text":"...","language":"es","duration_seconds":4.96,"model":"whisper-large-v3","device":"cuda:0"}`

- Formatos soportados: wav, mp3, flac, ogg, m4a... (decodificación vía ffmpeg, con fallback a soundfile)
- `language`: código ISO de 2-3 letras o vacío para detección automática
- Usa el dispositivo y dtype configurados en runtime (`device`, `dtype`)

## Panel web

`http://localhost:8001/webui` permite:

- Seleccionar, cargar y descargar el modelo (más ver su tipo)
- Seleccionar y clonar voces, y desactivar el voice cloning
- Probar la síntesis (texto, idioma, locutor, instrucción) y reproducir el audio en el navegador
- Transcribir archivos de audio con Whisper y descargar el modelo
- Configurar parámetros de forma persistente (límite de caracteres, defaults, logging, dispositivo)
- Gestionar claves API
- Documentación completa en `/webui/docs`

## Configuración

### config/settings.py
- `settings`: instancia única de `Settings` (Pydantic). Grupos: `server` (host `0.0.0.0`, puerto `8001`), `paths` (directorios), `tts` (`default_model`, `default_voice`), `text` (`chunking` = modo de división `sentence`/`paragraph`), `whisper` (`whisper_model`), `limits` (límites de entrada, comprobados antes de usar GPU: `max_text_characters` = 10000, `max_reference_audio_mb` = 25, `max_audio_duration_seconds` = 30), `queue`, `auth`, `logging` y `runtime` (editable desde el panel y persistida en `data/runtime.json`: `max_text_chars` = tamaño máximo de cada fragmento generado, `device`, `dtype`, flags de VRAM, etc.).
- Variables de entorno con prefijo `QWEN_TTS_`. Soporta al menos:
  - `QWEN_TTS_HOST=0.0.0.0`, `QWEN_TTS_PORT=8001`
  - `QWEN_TTS_DEVICE=cuda:0` (o `auto`/`cpu`), `QWEN_TTS_DTYPE=bfloat16` (o `float16`/`float32`/`auto`)
  - `QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `QWEN_TTS_REQUIRE_API_KEY=true`
  - `QWEN_TTS_TEXT__CHUNKING=sentence`
  - `QWEN_TTS_LIMITS__MAX_TEXT_CHARACTERS=10000`, `QWEN_TTS_LIMITS__MAX_REFERENCE_AUDIO_MB=25`, `QWEN_TTS_LIMITS__MAX_AUDIO_DURATION_SECONDS=30`
  - `QWEN_TTS_VOICES_DIR=./voices`, `QWEN_TTS_AUDIO_DIR=./audios`
- El resto de campos usa el nombre por subgrupo (`QWEN_TTS_<GRUPO>__<CAMPO>`), p.ej. `QWEN_TTS_CORS__ALLOW_ORIGINS=["*"]`.
- Precedencia: variables de entorno > `data/runtime.json` > defaults. Las variables `QWEN_TTS_DEVICE`, `QWEN_TTS_DTYPE` y `QWEN_TTS_REQUIRE_API_KEY` (configuración en tiempo de ejecución) tienen prioridad sobre el archivo persistido.

### Configuración en tiempo de ejecución (`data/runtime.json`)
Gestionada desde el panel: límite de caracteres, voz/idioma/instrucción por defecto, `device` (auto/cuda/cpu), dtype, logging de peticiones, timeout de reproducción, claves API, streaming (`streaming_enabled`) y guardado de audios (`save_audios`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Ejecución

```bash
# Opción A: con entorno virtual (recomendado)
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python app.py

# Opción B: con el Python del sistema (requiere las dependencias instaladas)
python3 app.py
```

El servidor inicia en `http://0.0.0.0:8001` (host y puerto configurables en `config/settings.py`), carga el modelo por defecto y queda listo. La concurrencia está controlada por `QueueService` con locks separados: `model_lock` serializa load/unload/switch y `inference_lock` controla la inferencia GPU (TTS, voice cloning, Whisper). Las operaciones de modelo esperan a la inferencia en curso y bloquean nuevas, evitando carreras del tipo unload + inference + load.
