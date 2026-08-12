#!/usr/bin/env python3
"""Benchmarks del servidor TTS usando exclusivamente su API HTTP.

No modifica el servidor: solo consume /tts, /tts/stream, /metrics,
/system/status, /model/* y /webui/api/config.

Benchmarks disponibles (--benchmark):
  A   Modelos        -> compara ids de modelo (--models 0.6B,1.7B...)
  B   Hardware       -> cpu vs gpu (--devices cpu,gpu)
  C   Streaming      -> /tts normal vs /tts/stream (--streaming)
  D   Chunking       -> sentence vs paragraph (--chunking)

Uso:
  python benchmarks/run.py bench a --models Qwen3-TTS-12Hz-0.6B-CustomVoice,Qwen3-TTS-12Hz-1.7B-Base --api-key KEY
  python benchmarks/run.py list-texts
  python benchmarks/run.py mods ['<bad_float32','']"

Resultados JSON en benchmarks/results/.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time

import psutil
import requests

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BENCHMARKS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(BENCHMARKS_DIR, "results")
DEFAULT_URL = os.environ.get("QTT_BASE_URL", "http://localhost:8001")

SAMPLE_RATE = 24000  # TTS_SAMPLE_RATE del servidor (wav)


# --------------------------------------------------------------------------
# Cliente ligero sobre la API
# --------------------------------------------------------------------------
class TTSClient:
    def __init__(self, base_url, api_key=None, timeout=300):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.timeout = timeout
        self.s = requests.Session()
        self._memory_peak_thread = None

    def _h(self):
        h = {}
        if self.key:
            h["X-API-Key"] = self.key
        return h

    # ---- llamadas
    @staticmethod
    def _raise(resp):
        """Levantar un error incluyendo el detalle JSON del servidor."""
        try:
            body = resp.json()
        except Exception:
            body = None
        detail = None
        if isinstance(body, dict):
            # Errores unificados del servidor: {"error": {"code", "message"}}
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message") or err.get("code")
            else:
                detail = body.get("detail")
        if isinstance(detail, dict):
            detail = detail.get("message") or detail
        msg = f"HTTP {resp.status_code} {resp.reason} en {resp.url}"
        if detail:
            msg += f" :: {detail}"
        raise requests.HTTPError(msg, response=resp)

    def get(self, path, **kw):
        r = self.s.get(self.base + path, headers=self._h(), timeout=self.timeout, **kw)
        if not r.ok:
            self._raise(r)
        return r

    def post(self, path, json=None, **kw):
        r = self.s.post(self.base + path, headers=self._h(), json=json, timeout=self.timeout, **kw)
        if not r.ok:
            self._raise(r)
        return r

    # ---- estado
    def metrics(self):
        return self.get("/metrics").json()

    def system_status(self):
        return self.get("/system/status").json()

    def models(self):
        return self.get("/models").json()

    def config(self):
        return self.get("/webui/api/config").json()

    def set_config(self, changes):
        return self.post("/webui/api/config", json=changes).json()

    def load_model(self, model_id):
        return self.post("/model/load", json={"model_id": model_id}).json()

    def unload_model(self, model_id=None):
        body = {} if model_id is None else {"model_id": model_id}
        return self.post("/model/unload", json=body).json()

    # ---- TTS
    def synth(self, text, **extra):
        payload = {"text": text, "output_format": "wav", **extra}
        r = self.post("/tts", json=payload)
        return r.content

    def stream(self, text, **extra):
        payload = {"text": text, "output_format": "wav", **extra}
        r = self.post("/tts/stream", json=payload)
        return r.content


# --------------------------------------------------------------------------
# Medición de recursos (sin tocar el servidor)
# --------------------------------------------------------------------------
def _wav_duration(data: bytes) -> float:
    """Duración de un WAV de 16-bit mono (o 0.0 si no se puede decodificar)."""
    if len(data) < 44 or data[:4] != b"RIFF":
        return 0.0
    fmt = None
    i = 12
    while i + 8 <= len(data):
        chunk_id = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "little")
        if chunk_id == b"fmt " and size >= 16:
            channels = int.from_bytes(data[i + 8 + 2:i + 8 + 4], "little")
            rate = int.from_bytes(data[i + 8 + 4:i + 8 + 8], "little")
            bits = int.from_bytes(data[i + 8 + 14:i + 8 + 16], "little")
            fmt = (channels, rate, bits)
            break
        i += 8 + size
    if fmt is None:
        return 0.0
    channels, rate, bits = fmt
    bytes_per_sample = max(1, channels * bits // 8)
    data_start = 12
    i = 12
    while i + 8 <= len(data):
        if data[i:i + 4] == b"data":
            data_start = i + 8
            break
        i += 8 + int.from_bytes(data[i + 4:i + 8], "little")
    audio_bytes = max(0, len(data) - data_start)
    return (audio_bytes / bytes_per_sample) / rate if rate else 0.0


class _RamSampler:
    """Sondea en un hilo la RAM del proceso servidor (RSS, GB) mientras dura
    una generación. Requiere que se fije la variable de entorno BENCH_PID al
    PID del proceso del servidor. Si no se conoce, devuelve 0 y se puede
    ignorar (no forma parte del núcleo del benchmark)."""
    def __init__(self, interval=0.05):
        self._interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        pid = os.environ.get("BENCH_PID")
        self._proc = psutil.Process(int(pid)) if pid else None

    def start(self):
        if not self._proc:
            return
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                self._samples.append(self._proc.memory_info().rss / (1024 ** 3))
                time.sleep(self._interval)
        except Exception:
            pass

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=1)

    def peak(self):
        return round(max(self._samples), 3) if self._samples else 0.0


def _vram_gb(client):
    """VRAM usada que reporta el propio servidor (GB, /system/status)."""
    try:
        used_mb = client.system_status().get("gpu", {}).get("used_vram_mb")
        if used_mb is not None:
            return used_mb / 1024
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Métricas por petición
# --------------------------------------------------------------------------
_sampler = None


def _measure(client, text, kind, tags=None):
    """Ejecuta una generación y devuelve métricas de latencia/audio/recursos.

    kind: 'tts' (no-streaming, TTFB=fin de generación) o 'stream' (TTFB =
    primer byte del stream).
    """
    global _sampler
    if _sampler is None:
        _sampler = _RamSampler()
    start = time.perf_counter()
    vram_before = _vram_gb(client)
    _sampler.start()
    try:
        if kind == "stream":
            ttf_start = time.perf_counter()
            data = client.stream(text, model=tags.get("model") if tags else None,
                                 language=tags.get("language") if tags else None,
                                 voice=tags.get("voice") if tags else None)
            ttft = (time.perf_counter() - ttf_start) * 1000
        else:
            ttf_start = time.perf_counter()
            data = client.synth(text, model=tags.get("model") if tags else None,
                                language=tags.get("language") if tags else None,
                                voice=tags.get("voice") if tags else None)
            ttft = (time.perf_counter() - ttf_start) * 1000
        elapsed_ms = (time.perf_counter() - start) * 1000
        peak_ram = _sampler.peak()
    finally:
        _sampler.stop()

    vram_after = _vram_gb(client)
    audio_dur = _wav_duration(data)
    rtf = elapsed_ms / (audio_dur * 1000) if audio_dur else 0.0
    return {
        "ok": True,
        "bytes": len(data),
        "audio_duration_s": round(audio_dur, 3),
        "total_ms": round(elapsed_ms, 1),
        "ttfb_ms": round(ttft, 1),
        "rtf": round(rtf, 3),
        "vram_used_gb": round((vram_after or 0), 3),
        "vram_before_gb": round((vram_before or 0), 3),
        "peak_ram_gb": peak_ram,
    }


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
# Idiomas que espera la API del servidor (nombres completos, no códigos ISO).
LANG_MAP = {"es": "spanish", "en": "english", "zh": "chinese", "ja": "japanese",
            "ko": "korean", "de": "german", "fr": "french", "ru": "russian",
            "pt": "portuguese", "it": "italian"}


def _run_all(client, texts, tags, kind="tts"):
    rows = []
    for t in texts:
        lang = LANG_MAP.get(t["lang"], t["lang"])
        try:
            row = _measure(client, t["text"], kind, tags={**tags, "language": lang})
            row["error"] = None
        except Exception as e:
            row = {"ok": False, "error": str(e), "total_ms": 0, "ttfb_ms": 0,
                   "rtf": 0.0, "audio_duration_s": 0.0, "bytes": 0,
                   "vram_used_gb": 0.0, "peak_ram_gb": 0.0}
        row["text"] = t["id"]
        row["category"] = t["category"]
        rows.append(row)
    return rows


def _server_limits(client):
    """Límites del servidor (runtime) para filtrar textos antes de enviarlos."""
    try:
        c = client.config()
        return {
            "max_text_chars": c.get("max_text_characters", 10000),
            "max_audio_s": c.get("max_estimated_audio_duration_seconds", 30),
        }
    except Exception:
        return {"max_text_chars": 10000, "max_audio_s": 30}


def _valid_texts(client, texts):
    """Textos que no exceden los límites del servidor (heurística len/16 s).

    El servidor rechaza con 400 los textos cuya duración estimada
    (len/CHARS_PER_SECOND) supera max_estimated_audio_duration_seconds.
    """
    limits = _server_limits(client)
    valid, skipped = [], []
    for t in texts:
        est = len(t["text"]) / 16.0
        if len(t["text"]) > limits["max_text_chars"] or est > limits["max_audio_s"]:
            skipped.append((t["id"], round(est, 1)))
        else:
            valid.append(t)
    if skipped:
        print(f"   aviso: {len(skipped)} texto(s) superan el límite de audio "
              f"estimado ({limits['max_audio_s']}s) y se omiten: "
              + ", ".join(f"{i}~{e}s" for i, e in skipped))
    return valid


def _model_type(client, model_id):
    """Tipo de modelo ('base', 'custom_voice', 'voice_design', ...) o None."""
    try:
        for e in client.get("/models/status").json().get("models", []):
            if e.get("model") == model_id:
                return e.get("type") or e.get("model_type")
        st = client.get("/model/status").json()
        if st.get("model_id") == model_id or (st.get("model") == model_id):
            return st.get("model_type") or st.get("type")
    except Exception:
        pass
    return None


def _first_voice(client):
    """Primera voz local válida (id) para referencia de clonación, o None."""
    try:
        for v in client.get("/voices").json().get("voices_detail", []):
            if v.get("valid"):
                return v.get("id")
    except Exception:
        pass
    return None


def _voice_for_model(client, model_id):
    """Voz de referencia para modelos Base (si hay voces locales)."""
    if _model_type(client, model_id) == "base":
        return _first_voice(client)
    return None


def bench_models(client, models, texts, tag_base):
    """Benchmark A: comparar varios ids de modelo."""
    results = []
    for m in models:
        print(f">> modelo: {m}  (cargando...)")
        load_start = time.perf_counter()
        client.load_model(m)
        load_s = round(time.perf_counter() - load_start, 2)
        voice = _voice_for_model(client, m)
        print(f"   tipo: {_model_type(client, m)} · voz ref: {voice or 'ninguna'}")
        sys.stdout.flush()
        rows = _run_all(client, _valid_texts(client, texts), {"model": m, "voice": voice}, kind="tts")
        results.append({
            "benchmark": "A-models",
            "model": m,
            "model_type": _model_type(client, m),
            "voice_ref": voice,
            "load_time_s": load_s,
            "samples": rows,
        })
        print(f"   modelo listo: {m}")
    return results


def _normalize_device(dev):
    """'gpu' -> 'cuda' (el servidor acepta auto|cpu|cuda|cuda:N)."""
    return "cuda" if dev.lower() == "gpu" else dev


def bench_hardware(client, devices, texts):
    """Benchmark B: cpu vs gpu (cambia device en runtime)."""
    results = []
    # solo modelos no auditados: usar el activo o el primero disponible
    available = client.models().get("available_models", [])
    m = available[0] if available else None
    voice = _voice_for_model(client, m) if m else None
    prev_device = None
    try:
        prev_device = client.config().get("device")
    except Exception:
        pass
    for dev in devices:
        dev = _normalize_device(dev)
        print(f">> device: {dev}")
        client.set_config({"device": dev})
        time.sleep(0.5)
        if m:
            client.load_model(m)
        rows = _run_all(client, _valid_texts(client, texts), {"model": m, "voice": voice}, kind="tts")
        results.append({
            "benchmark": "B-hardware",
            "device": dev,
            "model": m,
            "samples": rows,
        })
        print(f"   device {dev} listo")
    # restaurar el device previo (no forzar "auto": no pisar la config del usuario)
    if prev_device:
        try:
            client.set_config({"device": prev_device})
            print(f"   device restaurado: {prev_device}")
        except Exception:
            pass
    return results


def bench_streaming(client, texts):
    """Benchmark C: no-streaming vs streaming."""
    m = _current_model(client)
    voice = _voice_for_model(client, m) if m else None
    texts = _valid_texts(client, texts)
    normal = _run_all(client, texts, {"model": m, "voice": voice}, kind="tts")
    stream = _run_all(client, texts, {"model": m, "voice": voice}, kind="stream")
    return [{
        "benchmark": "C-streaming",
        "model": m,
        "mode": "normal",
        "samples": normal,
    }, {
        "benchmark": "C-streaming",
        "model": m,
        "mode": "stream",
        "samples": stream,
    }]


def bench_chunking(client, texts):
    """Benchmark D: sentence vs paragraph."""
    m = _current_model(client)
    voice = _voice_for_model(client, m) if m else None
    prev = client.config().get("chunking", "sentence")
    results = []
    texts = _valid_texts(client, texts)
    for mode in ("sentence", "paragraph"):
        print(f">> chunking: {mode}")
        client.set_config({"chunking": mode})
        time.sleep(0.5)
        rows = _run_all(client, texts, {"model": m, "voice": voice}, kind="tts")
        results.append({"benchmark": "D-chunking", "model": m, "mode": mode, "samples": rows})
    try:
        client.set_config({"chunking": prev})
    except Exception:
        pass
    return results


def _current_model(client):
    try:
        m = client.models().get("current_model")
        if m:
            return m
    except Exception:
        pass
    av = client.models().get("available_models", [])
    return av[0] if av else None


BENCH_ALIAS = {
    "models": "a", "b": "hardware", "c": "streaming", "d": "chunking",
}


# --------------------------------------------------------------------------
# Agregación & guardado & salida
# --------------------------------------------------------------------------
def _agg(rows):
    """Media de las peticiones correctas habilitadas."""
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {
            "error_rate_pct": 100.0, "count": len(rows),
            "avg_total_ms": 0, "avg_ttfb_ms": 0, "avg_rtf": 0.0,
            "avg_audio_s": 0, "avg_vram_gb": 0.0, "avg_peak_ram_gb": 0.0,
            "total_audio_s": 0, "errors": len(rows) - len(ok),
        }
    mean = lambda k: round(sum(r[k] for r in ok) / len(ok), 3) if ok else 0
    return {
        "error_rate_pct": round((len(rows) - len(ok)) / len(rows) * 100, 1) if rows else 0,
        "count": len(ok),
        "errors": len(rows) - len(ok),
        "avg_total_ms": round(mean("total_ms"), 1),
        "avg_ttfb_ms": round(mean("ttfb_ms"), 1),
        "avg_rtf": round(mean("rtf"), 3),
        "avg_audio_s": round(mean("audio_duration_s"), 1),
        "total_audio_s": round(sum(r["audio_duration_s"] for r in ok), 1),
        "avg_vram_gb": round(mean("vram_used_gb"), 3),
        "avg_peak_ram_gb": round(mean("peak_ram_gb"), 3),
    }


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(RESULTS_DIR, f"bench-{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {path}")
    return path


def _print_table(results):
    print()
    print("-" * 58)
    for group in results:
        a = _agg(group.get("samples", []))
        label = " · ".join(str(group.get(k, "")) for k in ("benchmark", "model", "device", "mode") if group.get(k))
        print(f"{label:<40} count={a['count']:>2} err={a['error_rate_pct']:>5.1f}% "
              f"total={a['avg_total_ms']:>7.1f}ms ttfb={a['avg_ttfb_ms']:>7.1f}ms "
              f"rtf={a['avg_rtf']:>5.3f} audio={a['avg_audio_s']:>6.1f}s "
              f"vram={a['avg_vram_gb']:>5.2f}G ram={a['avg_peak_ram_gb']:>5.2f}G")
    print("-" * 58)


def _print_block(title, agg, load_time=None):
    """Imprime el bloque resumen de un conjunto de peticiones."""
    print()
    print("-" * 41)
    print(title)
    print("-" * 41)
    if load_time is not None:
        print(f"Load time  {load_time:>8.2f} s")
    print(f"TTFT       {agg['avg_ttfb_ms'] / 1000:>8.2f} s")
    print(f"Generation {agg['avg_total_ms'] / 1000:>8.2f} s")
    print(f"Audio dur  {agg['avg_audio_s']:>8.2f} s")
    print(f"RTF        {agg['avg_rtf']:>8.3f}")
    print(f"Peak VRAM  {agg['avg_vram_gb']:>8.2f} GB")
    print(f"Peak RAM   {agg['avg_peak_ram_gb']:>8.2f} GB")
    print(f"Error rate {agg['error_rate_pct']:>7.1f}%")
    print("-" * 41)


# --------------------------------------------------------------------------
# Trampolín
# --------------------------------------------------------------------------
async def _run(client, args, texts):
    if args.bench_unit == "a":
        ids = (args.models or ",".join(
            m for m in ["Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen3-TTS-12Hz-1.7B-Base"]
            if m in (client.models().get("available_models") or [])
        )).split(",")
        bench = bench_models(client, [i for i in ids if i], texts, {})
        for group in bench:
            agg = _agg(group["samples"])
            _print_block(f"Modelo: {group['model']}  (A-models)",
                         agg, load_time=group.get("load_time_s"))
    elif args.bench_unit == "hardware":
        devs = (args.devices or "cpu,gpu").split(",")
        bench = bench_hardware(client, [d for d in devs if d], texts)
        for group in bench:
            _print_block(f"Device: {group['device']}  (B-hardware)", _agg(group["samples"]))
    elif args.bench_unit == "streaming":
        bench = bench_streaming(client, texts)
        for group in bench:
            _print_block(f"Modo: {group['mode']}  (C-streaming)", _agg(group["samples"]))
    elif args.bench_unit == "chunking":
        bench = bench_chunking(client, texts)
        for group in bench:
            _print_block(f"Chunking: {group['mode']}  (D-chunking)", _agg(group["samples"]))
    else:
        raise SystemExit(f"Benchmark desconocido: {args.bench_unit}")

    _save(bench)
    _print_table(bench)


async def _test(client):
    """Echo para verificar conectividad."""
    st = client.system_status()
    print("servidor:", client.base)
    print("gpu:", st.get("gpu"))
    print("tts:", st.get("tts"))


def main():
    p = argparse.ArgumentParser(description="Benchmarks TTS vía API")
    p.add_argument("url", nargs="?", default=DEFAULT_URL, help="URL base del servidor")
    p.add_argument("--api-key", default=os.environ.get("QTT_API_KEY"), help="X-API-Key")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--output", help="ruta del resumen JSON (para report)")

    sub = p.add_subparsers(dest="sub")
    sub.add_parser("list-texts").set_defaults(fn="list_texts")
    sub.add_parser("test").set_defaults(fn="test")

    b = sub.add_parser("bench")
    b.add_argument("--bench-unit", required=True,
                   help="a|b|c|d|models|hardware|streaming|chunking")
    b.add_argument("--models", help="ids de modelo separados por coma")
    b.add_argument("--devices", help="devices separados por coma (cpu,gpu)")
    b.add_argument("--texts", help="ids de texto o categoría (short,medium,long,hard)")

    args = p.parse_args()
    client = TTSClient(args.url, args.api_key, args.timeout)

    fn = getattr(args, "fn", None)
    if fn:
        if fn == "list_texts":
            import benchmarks.texts as T
            for t in T.all_texts():
                print(f"{t['id']:<14} {t['category']:<8} {t['lang']:<4} {t['text'][:50]}...")
            return
        if fn == "test":
            asyncio.run(_test(client))
            return


    # filtrado de textos por categoría/s
    import benchmarks.texts as T
    texts = T.all_texts()
    if getattr(args, "texts", None):
        sel = set(args.texts.split(","))
        texts = [t for t in texts if t["id"] in sel or t["category"] in sel]

    unit = args.bench_unit
    unit = BENCH_ALIAS.get(unit, unit)
    args.bench_unit = unit
    asyncio.run(_run(client, args, texts))


if __name__ == "__main__":
    main()
