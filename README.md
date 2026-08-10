# Servidor TTS + Transcripción - Qwen3-TTS / Whisper

Servidor FastAPI modular para generación de texto a voz con modelos Qwen3-TTS locales y transcripción de audio con Whisper. Incluye voice cloning, panel de administración web y gestión de claves API.

## Estructura del Proyecto

```
qwen3-tts-server/
├── app.py               # Composición de la aplicación y arranque
├── config/              # Configuración estática y defaults
│   ├── settings.py      # CONFIG (host, puerto, directorios, modelo/voz por defecto)
│   └── defaults.py      # Defaults de configuración en tiempo de ejecución
├── schemas/             # Esquemas Pydantic por dominio
│   ├── tts.py           # TTSRequest, TTSRequestOpenWebUI
│   ├── whisper.py       # WhisperStatusResponse
│   ├── voices.py        # LoadVoiceRequest
│   ├── models.py        # LoadModelRequest
│   ├── system.py        # ConfigUpdate, ApiKeyCreate, SystemStatus
│   └── errors.py        # ErrorResponse
├── routes/              # Endpoints HTTP (sin lógica de inferencia)
│   ├── tts.py           # /tts, /tts/play, /tts/audio/speech
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

Flujo de una petición: HTTP → Autenticación (`security/auth.py`) → Validación de esquema (`schemas/`) → Validación de entrada (`security/validation.py`) → Servicio (`services/`) → `ModelManager`/Whisper → `AudioService` → Respuesta. Las rutas HTTP no contienen lógica de inferencia; todo el acceso al modelo pasa por los servicios.

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

| Método | Endpoint | Descripción | Clave API |
|--------|----------|-------------|-----------|
| GET | `/` | Estado del servidor | No |
| POST | `/tts` | Generar TTS (estándar) | Sí |
| POST | `/tts/play` | Generar TTS y reproducirlo en este equipo | Sí |
| POST | `/tts/audio/speech` | Generar TTS (compatible OpenWebUI) | Sí |
| GET | `/tts/audio/models` | Modelos disponibles (formato OpenWebUI) | No |
| GET | `/tts/audio/voices` | Voces disponibles (formato OpenWebUI) | No |
| POST | `/model/load` | Cargar modelo local | Sí |
| POST | `/model/unload` | Descargar modelo y liberar VRAM | Sí |
| GET | `/models` | Listar modelos disponibles | No |
| POST | `/voice/load` | Cargar voz para clonación (solo modelos `base`) | Sí |
| POST | `/voice/create` | Subir WAV + transcripción (multipart), la guarda y la clona | Sí |
| POST | `/voice/unload` | Desactivar voice cloning | Sí |
| GET | `/voices` | Listar voces disponibles | No |
| POST | `/transcribe` | Transcribir audio a texto con Whisper | Sí |
| GET | `/transcribe/status` | Estado del modelo Whisper | No |
| POST | `/transcribe/unload` | Descargar Whisper y liberar VRAM | Sí |
| GET | `/webui` | Panel de administración web | No |
| GET | `/webui/docs` | Documentación de la API y ejemplos | No |
| GET/POST | `/webui/api/config` | Leer/actualizar configuración en tiempo de ejecución | No |
| GET | `/webui/api/devices` | GPUs disponibles y dispositivo en uso | No |
| GET/POST | `/webui/api/apikeys` | Listar/crear claves API | No |
| POST | `/webui/api/apikeys/{id}/toggle` | Activar/desactivar clave API | No |
| DELETE | `/webui/api/apikeys/{id}` | Eliminar clave API | No |

Cuando la exigencia de clave API está activada (panel → Claves API), los endpoints de servicio requieren `X-API-Key: qt-...` o `Authorization: Bearer qt-...`.

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
- `CONFIG`: host (`0.0.0.0`), puerto (`8001`), `default_model`, `whisper_model`, `max_text_chars`, directorios de modelos/voces/audios

### Configuración en tiempo de ejecución (`data/runtime.json`)
Gestionada desde el panel: límite de caracteres, voz/idioma/instrucción por defecto, `device` (auto/cuda/cpu), dtype, logging de peticiones, timeout de reproducción, claves API.

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

El servidor inicia en `http://0.0.0.0:8001` (host y puerto configurables en `config/settings.py`), carga el modelo por defecto y queda listo. La concurrencia está controlada por `QueueService` (semáforo global definido en `app.py`).
