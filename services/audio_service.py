#!/usr/bin/env python3
"""Servicio central de audio del servidor.

Toda operación con audio (decodificación, validación, normalización,
conversión, duración y guardado) pasa por AudioService: TTS, Whisper y las
rutas delegan aquí y no repiten esta lógica.

Formatos soportados (lectura y escritura): wav, mp3, flac, ogg, m4a.
Decodificación: soundfile (wav/flac/ogg) con fallback a ffmpeg (mp3/m4a...).
"""

import io
import os
import re
import shutil
import subprocess
import tempfile
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf

from fastapi import HTTPException

logger = logging.getLogger("tts")

# Reproductores de audio disponibles en el sistema para reproducir en este equipo
PLAYERS = [p for p in ["mpv", "ffplay", "paplay", "aplay", "play"] if shutil.which(p)]

# Frecuencia de muestreo de salida de los modelos Qwen3-TTS (24 kHz)
TTS_SAMPLE_RATE = 24000

# Frecuencia de muestreo canónica para voz de referencia y Whisper (16 kHz)
SPEECH_SAMPLE_RATE = 16000

# Caracteres admitidos en identificadores usados como nombre de archivo
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_id(value) -> str:
    """Sanitizar un identificador para su uso en nombres de archivo."""
    if not value:
        return ""
    return _SAFE_ID_RE.sub("", str(value))[:32]


class AudioValidationError(ValueError):
    """Audio inválido, no soportado o que no cumple los límites.

    Hereda de ValueError: las capas HTTP que ya traducen ValueError a 400
    lo manejan sin cambios.
    """


@dataclass
class AudioData:
    """Resultado de AudioService.validate().

    Distingue validación de decodificación: validate() comprueba tamaño,
    formato, duración, sample rate, canales y decodificabilidad; si el
    llamador necesita el contenido decodificado, se devuelve aquí (una sola
    decodificación, reutilizable), evitando que un load() posterior
    decodifique otra vez.

    - ``samples``: numpy array float32 con el contenido decodificado, o None
      si no se decodificó (decode=False y formato reconocible por cabecera).
    - ``sample_rate`` / ``channels`` / ``duration`` / ``format``: propiedades
      del audio (formato nativo, canales originales).
    - ``size_bytes``: tamaño del archivo original.
    """
    format: str
    sample_rate: int
    channels: int
    duration: float
    size_bytes: int
    samples: Optional[np.ndarray] = None


class AudioService:
    """Operaciones de audio del servidor (decodificar, validar, normalizar,
    convertir, medir duración, guardar, reproducir)."""

    # Formatos soportados (sin punto, en minúsculas)
    FORMATS = ("wav", "mp3", "flac", "ogg", "m4a")
    EXTENSIONS = {f".{f}" for f in FORMATS}
    # Formatos que soundfile puede leer/escribir directamente
    SF_FORMATS = {"wav", "flac", "ogg"}
    # Tipos MIME aceptados por formato (audio/*)
    MIME_TYPES = {
        "wav": {"audio/wav", "audio/x-wav", "audio/wave", "audio/x-pn-wav"},
        "mp3": {"audio/mpeg", "audio/mp3", "audio/x-mpeg", "audio/x-mp3"},
        "flac": {"audio/flac", "audio/x-flac"},
        "ogg": {"audio/ogg", "application/ogg", "audio/vorbis", "audio/x-ogg"},
        "m4a": {"audio/mp4", "audio/x-m4a", "audio/m4a"},
    }
    # application/octet-stream: se ignora (curl/upload genéricos no declaran tipo)
    _IGNORED_MIME = {"application/octet-stream", ""}

    def __init__(self, config, queue):
        """config: config.settings.Settings (objeto Settings único)."""
        self._config = config
        self._queue = queue

    # -- Utilidades --------------------------------------------------------

    @staticmethod
    def _as_bytes(source) -> bytes:
        """Normalizar una fuente de audio (bytes, ruta o file-like) a bytes."""
        if isinstance(source, bytes):
            return source
        if hasattr(source, "read"):
            return source.read()
        if isinstance(source, (str, os.PathLike)):
            with open(source, "rb") as f:
                return f.read()
        raise AudioValidationError(
            f"Fuente de audio no reconocida: {type(source).__name__}"
        )

    @staticmethod
    def _resample(audio, sr: int, target_sr: int) -> np.ndarray:
        """Remuestrear audio float32 a target_sr."""
        from scipy.signal import resample_poly
        return resample_poly(audio, target_sr, sr).astype("float32")

    # -- Decodificación ----------------------------------------------------

    def _decode_native(self, source) -> tuple:
        """Decodificar a float32 sin remuestrear ni mezclar canales.

        Devuelve (audio, sample_rate) con el formato nativo de la fuente.
        """
        data = self._as_bytes(source)
        try:
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            return audio, int(sr)
        except Exception as e:
            logger.debug(f"soundfile no pudo decodificar el audio ({e}); probando ffmpeg...")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise AudioValidationError(
                f"No se pudo decodificar el audio. Formatos soportados: {', '.join(self.FORMATS)}. "
                "Se requiere ffmpeg para mp3/m4a."
            )
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(data)
            tmp.flush()
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-i", tmp.name, "-f", "wav", "-"],
                capture_output=True,
            )
        if proc.returncode != 0 or not proc.stdout:
            raise AudioValidationError(
                proc.stderr.decode(errors="ignore").strip()
                or f"No se pudo decodificar el audio. Formatos soportados: {', '.join(self.FORMATS)}"
            )
        audio, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
        return audio, int(sr)

    def load(self, source, target_sr=None, mono: bool = True) -> tuple:
        """Decodificar audio (bytes, ruta o file-like) a float32.

        Devuelve (audio, sample_rate). El sample rate es el nativo salvo que
        se indique target_sr (se remuestrea). Si mono=True, se mezcla a un
        solo canal.
        """
        audio, sr = self._decode_native(source)
        return self.prepare(audio, sr, target_sr, mono)

    def prepare(self, audio, sr, target_sr=None, mono: bool = True) -> tuple:
        """Post-procesar audio ya decodificado: mezclar a mono y remuestrear.

        Produce el mismo resultado que load() pero sin re-decodificar;
        pensado para reutilizar el numpy array devuelto por
        validate(decode=True).
        """
        audio = np.asarray(audio, dtype="float32")
        if mono and audio.ndim > 1:
            audio = audio.mean(axis=1)
        if target_sr is not None and sr != target_sr:
            audio = self._resample(audio, sr, target_sr)
            sr = target_sr
        return audio.astype("float32"), sr

    # -- Información -------------------------------------------------------

    def _probe(self, source, decoded=None) -> dict:
        """Información del audio sin decodificarlo por completo.

        Devuelve {format, sample_rate, channels, duration} (duración en
        segundos). Usa la cabecera (soundfile) o ffprobe; como último
        recurso, decodifica (reutilizando ``decoded`` si se proporciona,
        para no decodificar dos veces).
        """
        data = self._as_bytes(source)
        try:
            info = sf.info(io.BytesIO(data))
            return {
                "format": str(info.format).lower(),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "duration": float(info.frames / info.samplerate),
            }
        except Exception as e:
            logger.debug(f"soundfile no pudo inspeccionar el audio ({e}); probando ffprobe...")

        ffprobe = shutil.which("ffprobe")
        if ffprobe is not None:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(data)
                tmp.flush()
                proc = subprocess.run(
                    [
                        ffprobe, "-v", "error",
                        "-select_streams", "a:0",
                        "-show_entries", "stream=sample_rate,channels:format=format_name,duration",
                        "-of", "default=noprint_wrappers=1",
                        tmp.name,
                    ],
                    capture_output=True,
                )
            if proc.returncode == 0:
                fields = {}
                for line in proc.stdout.decode(errors="ignore").splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        fields[key.strip()] = value.strip()
                duration = float(fields.get("duration") or 0.0)
                sample_rate = int(float(fields.get("sample_rate") or 0))
                channels = int(fields.get("channels") or 1)
                if duration > 0 or sample_rate > 0:
                    return {
                        "format": fields.get("format_name", "audio"),
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "duration": duration,
                    }

        # Último recurso: decodificar y medir (sin re-decodificar si ya hay
        # contenido decodificado).
        if decoded is None:
            audio, sr = self._decode_native(data)
        else:
            audio, sr = decoded
        return {
            "format": "audio",
            "sample_rate": int(sr),
            "channels": 2 if audio.ndim > 1 else 1,
            "duration": float(audio.shape[0] / sr),
        }

    def get_duration(self, source) -> float:
        """Duración en segundos de una fuente de audio (bytes, ruta o file-like)."""
        return self._probe(source)["duration"]

    # -- Normalización -----------------------------------------------------

    def normalize(self, wav, dbfs: float = None) -> np.ndarray:
        """Normalizar el pico de amplitud a ``dbfs`` dBFS.

        Si ``dbfs`` es None, usa el valor editable del panel
        (normalization_dbfs, por defecto -1 dBFS). Devuelve float32 en
        [-1, 1]. El silencio se devuelve sin cambios.
        """
        if dbfs is None:
            from services.config_service import get_runtime
            dbfs = get_runtime().normalization_dbfs
        wav = np.asarray(wav, dtype="float32")
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak < 1e-6:
            return wav
        target = 10 ** (dbfs / 20.0)
        return np.clip(wav * (target / peak), -1.0, 1.0)

    # -- Validación --------------------------------------------------------

    @staticmethod
    def _detect_format(data: bytes):
        """Detectar el formato real por los bytes mágicos (contenido).

        Devuelve el formato (wav, mp3, flac, ogg, m4a) o None si no hay
        una firma reconocible.
        """
        if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
            return "wav"
        if data.startswith(b"fLaC"):
            return "flac"
        if data.startswith(b"OggS"):
            return "ogg"
        if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and 0xE0 <= data[1] <= 0xFF):
            return "mp3"
        if len(data) >= 8 and data[4:8] == b"ftyp":
            return "m4a"
        return None

    @staticmethod
    def _format_from_mime(content_type: str):
        """Formato declarado por un tipo MIME.

        Devuelve el formato, o None si no se declara tipo (vacío o
        application/octet-stream, típico de curl -F). Lanza
        AudioValidationError si se declara un tipo no soportado.
        """
        mime = (content_type or "").split(";")[0].strip().lower()
        if mime in AudioService._IGNORED_MIME:
            return None
        for fmt, types in AudioService.MIME_TYPES.items():
            if mime in types:
                return fmt
        raise AudioValidationError(
            f"Tipo MIME no soportado ('{content_type}'). "
            f"Soportados: {', '.join(sorted(f for types in AudioService.MIME_TYPES.values() for f in types))}"
        )

    def validate(self, source, max_bytes=None, max_duration=None,
                 formats=None, filename=None, content_type=None,
                 min_sample_rate=None, max_sample_rate=None,
                 max_channels=None, decode=False) -> AudioData:
        """Validar una fuente de audio y devolver el resultado (AudioData).

        Comprueba (sin confiar en la extensión): que no esté vacía, el
        tamaño (max_bytes), el tipo MIME declarado (content_type), que el
        contenido real (bytes mágicos) coincida con el formato declarado,
        la extensión (formats + filename), la duración (max_duration), el
        sample rate y el número de canales, y que el audio sea decodificable
        (decode=True valida el contenido completo, antes de consumir GPU).

        Validación y decodificación van separadas: el contenido se decodifica
        UNA sola vez (cuando hace falta por decode=True o por falta de firma
        reconocible) y se devuelve en AudioData.samples, que los llamadores
        pueden consumir sin necesidad de un load() posterior (que volvería a
        decodificar).

        Lanza AudioValidationError si no cumple.
        """
        data = self._as_bytes(source)
        size = len(data)
        if size == 0:
            raise AudioValidationError("El archivo de audio está vacío")
        if max_bytes is not None and size > max_bytes:
            raise AudioValidationError(
                f"El archivo de audio excede {max_bytes // (1024 * 1024)} MB"
            )

        # Formato esperado desde la extensión y el tipo MIME (si se declaran).
        expected = None
        if filename is not None:
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.EXTENSIONS:
                expected = ext.lstrip(".")
            elif ext:
                raise AudioValidationError(
                    f"Extensión no soportada ('{ext}'). "
                    f"Soportadas: {', '.join('.' + f for f in self.FORMATS)}"
                )
        mime_format = self._format_from_mime(content_type)
        if mime_format is not None and expected is not None and mime_format != expected:
            raise AudioValidationError(
                f"El tipo MIME ('{content_type}') no corresponde con la "
                f"extensión del archivo ('{expected}')"
            )
        expected = mime_format or expected

        if formats is not None:
            allowed = {f if f.startswith(".") else f".{f}" for f in formats}
            if filename is not None:
                ext = os.path.splitext(filename)[1].lower()
                if ext and ext not in allowed:
                    raise AudioValidationError(
                        f"Formato no soportado ('{ext or 'desconocido'}'). "
                        f"Soportados: {', '.join(sorted(f.lstrip('.') for f in allowed))}"
                    )

        # Contenido real: los bytes mágicos no deben contradecir lo declarado.
        detected = self._detect_format(data)
        if detected is not None and expected is not None and detected != expected:
            raise AudioValidationError(
                f"El contenido real del archivo es {detected}, no '{expected}' "
                "(la extensión/MIME no coinciden con el contenido)"
            )

        # Decodificación única: la necesitan los archivos sin firma
        # reconocible (para validar contenido) y decode=True (para validar
        # el contenido completo antes de la GPU).
        decoded = None
        if detected is None or decode:
            audio, sr = self._decode_native(data)
            if audio.shape[0] == 0:
                raise AudioValidationError(
                    "El archivo de audio está corrupto o solo contiene la cabecera "
                    "(no hay datos de audio)"
                )
            decoded = (audio, int(sr))

        # Información: cabecera (si hay formato reconocible) o, sin firma,
        # derivada del audio ya decodificado (sin decodificar de nuevo).
        if decoded is not None and detected is None:
            audio, sr = decoded
            info = {
                "format": "audio",
                "sample_rate": int(sr),
                "channels": 2 if audio.ndim > 1 else 1,
                "duration": float(audio.shape[0] / sr),
            }
        else:
            info = self._probe(data, decoded=decoded)

        if max_duration is not None and info["duration"] > max_duration:
            raise AudioValidationError(
                f"El audio excede la duración máxima de {max_duration:.0f}s "
                f"(duración: {info['duration']:.1f}s)"
            )
        if min_sample_rate is not None and info["sample_rate"] < min_sample_rate:
            raise AudioValidationError(
                f"Sample rate fuera del rango permitido: {info['sample_rate']} Hz "
                f"(mínimo: {min_sample_rate} Hz)"
            )
        if max_sample_rate is not None and info["sample_rate"] > max_sample_rate:
            raise AudioValidationError(
                f"Sample rate fuera del rango permitido: {info['sample_rate']} Hz "
                f"(máximo: {max_sample_rate} Hz)"
            )
        if max_channels is not None and info["channels"] > max_channels:
            raise AudioValidationError(
                f"Demasiados canales: {info['channels']} "
                f"(máximo admitido: {max_channels})"
            )
        return AudioData(
            format=info["format"],
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            duration=info["duration"],
            size_bytes=size,
            samples=decoded[0] if decoded is not None else None,
        )

    # -- Conversión --------------------------------------------------------

    def convert(self, wav, sr, format: str = "wav",
                sample_rate=None, channels: int = 1) -> bytes:
        """Codificar audio float32 a bytes en el formato indicado.

        format: wav | mp3 | flac | ogg | m4a. Si sample_rate difiere del de
        la entrada, se remuestrea. channels: 1 (mono) o 2 (stéreo).
        """
        fmt = str(format).lower().lstrip(".")
        if fmt not in self.FORMATS:
            raise AudioValidationError(
                f"Formato de salida no soportado: '{format}'. "
                f"Soportados: {', '.join(self.FORMATS)}"
            )

        audio = np.asarray(wav, dtype="float32")
        if sample_rate is not None and int(sample_rate) != sr:
            audio = self._resample(audio, sr, int(sample_rate))
            sr = int(sample_rate)
        if channels == 2 and audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        elif channels == 1 and audio.ndim > 1:
            audio = audio.mean(axis=1)

        if fmt in self.SF_FORMATS:
            buffer = io.BytesIO()
            sf.write(buffer, audio, sr, format=fmt)
            return buffer.getvalue()
        return self._encode_with_ffmpeg(audio, sr, fmt)

    def _encode_with_ffmpeg(self, audio: np.ndarray, sr: int, fmt: str) -> bytes:
        """Codificar a mp3/m4a vía ffmpeg (lame para mp3, aac para m4a)."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise AudioValidationError(
                f"Convertir a {fmt} requiere ffmpeg en el sistema"
            )
        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_out = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        in_name, out_name = tmp_in.name, tmp_out.name
        tmp_in.close()
        tmp_out.close()
        try:
            sf.write(in_name, audio, sr)
            codec = ["-c:a", "libmp3lame", "-b:a", "192k"] if fmt == "mp3" \
                else ["-c:a", "aac", "-b:a", "192k"]
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-i", in_name, *codec, out_name],
                capture_output=True,
            )
            if proc.returncode != 0:
                raise AudioValidationError(
                    proc.stderr.decode(errors="ignore").strip()
                    or f"ffmpeg no pudo convertir el audio a {fmt}"
                )
            with open(out_name, "rb") as f:
                return f.read()
        finally:
            for path in (in_name, out_name):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # -- Codificación (compatibilidad con el flujo TTS) --------------------

    def encode_wav(self, wav, sr) -> bytes:
        """Codificar audio a bytes WAV."""
        return self.convert(wav, sr, "wav")

    def encode_pcm(self, wav, sr) -> bytes:
        """Codificar audio a PCM 16-bit little-endian (raw)."""
        pcm = np.clip(wav, -1.0, 1.0)
        return (pcm * 32767).astype(np.int16).tobytes()

    def wav_stream_header(self, sr: int, channels: int = 1, bits: int = 16) -> bytes:
        """Cabecera WAV para streaming: tamaños desconocidos (0xFFFFFFFF).

        Válida para flujos chunked: los reproductores (ffmpeg, mpv, aplay...)
        aceptan esta marca de "tamaño indeterminado" y leen hasta el cierre
        de la conexión.
        """
        import struct
        byte_rate = sr * channels * bits // 8
        block_align = channels * bits // 8
        header = b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        header += b"fmt " + struct.pack(
            "<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, bits
        )
        header += b"data" + struct.pack("<I", 0xFFFFFFFF)
        return header

    # -- Guardado ----------------------------------------------------------

    def save(self, wav, sr, prefix: str, format: str = "wav",
             request_id: str = None, chunk_index: int = None) -> str:
        """Guardar audio en disco y devolver la ruta.

        No guarda nada si save_audios está desactivado en la configuración
        en tiempo de ejecución (devuelve cadena vacía).

        El nombre incluye un identificador único: no se confía solo en el
        timestamp (precisión de 1 segundo: dos guardados en el mismo segundo
        colisionarían y el segundo machacaría al primero, algo plausible con
        streaming multi-fragmento o peticiones concurrentes). Con
        request_id (+ chunk_index) el nombre es trazable; sin ellos, un
        sufijo aleatorio garantiza unicidad.

        Formato: ``<prefix>_<YYYYMMDD_HHMMSS>_<id>.wav``, con ``id`` =
        ``<request_id>_<chunk_index>``, ``<request_id>`` o un sufijo aleatorio.
        """
        fmt = str(format).lower().lstrip(".")
        from services.config_service import get_runtime_config
        if not get_runtime_config().get("save_audios", False):
            logger.info(f"Guardado de audios desactivado (se omite {prefix}_*.{fmt})")
            return ""
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        rid = _safe_id(request_id)
        if rid and chunk_index is not None:
            unique = f"{rid}_{int(chunk_index)}"
        elif rid:
            unique = rid
        else:
            unique = uuid.uuid4().hex[:6]
        path = f"{self._config.paths.audios_dir}/{prefix}_{dt}_{unique}.{fmt}"
        with open(path, "wb") as f:
            f.write(self.convert(wav, sr, fmt))
        logger.info(f"Audio guardado: {path}")
        return path

    def cleanup_old(self, max_age_seconds: float) -> int:
        """Eliminar audios generados hace más de max_age_seconds segundos.

        Es la base de la limpieza automática (storage.generated_audio_ttl_hours):
        impide que el directorio de audios crezca indefinidamente.
        """
        if max_age_seconds <= 0:
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        try:
            for fname in os.listdir(self._config.paths.audios_dir):
                path = os.path.join(self._config.paths.audios_dir, fname)
                try:
                    if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        removed += 1
                except OSError as e:
                    logger.warning(f"No se pudo eliminar {path}: {e}")
        except OSError as e:
            logger.warning(f"Error leyendo directorio de audios {self._config.paths.audios_dir}: {e}")
        if removed:
            logger.info(f"Limpieza: {removed} audio(s) antiguo(s) eliminados de {self._config.paths.audios_dir}")
        return removed

    # -- Reproducción local -------------------------------------------------

    def pick_player(self):
        """Primer reproductor de audio disponible en el sistema."""
        return PLAYERS[0] if PLAYERS else None

    async def play(self, audio_bytes: bytes, sr: int, timeout: int) -> dict:
        """Reproducir audio en este equipo, serializando con las reproducciones previas.

        Debe ejecutarse dentro de queue.inference_lock(). Devuelve info de la reproducción.
        """
        player = self.pick_player()
        if not player:
            raise HTTPException(
                500,
                "No hay reproductor de audio disponible (mpv, ffplay, paplay, aplay o play)",
            )

        async with self._queue.playback(timeout):
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"tts_play_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav",
            )
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)

            if player == "ffplay":
                cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
            else:
                cmd = [player, tmp_path]

            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._queue.register_playback(proc)
            logger.info(f"Audio en reproducción en este equipo (player={player}): {tmp_path}")

        return {
            "player": player,
            "temp_file": tmp_path,
            "sample_rate": sr,
        }
