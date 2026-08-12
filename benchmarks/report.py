#!/usr/bin/env python3
"""Genera informes a partir de los JSON de resultados de benchmarks.

Produce, para un archivo JSON (o todos los de benchmarks/results/):
  - resumen por bloque (Markdown, tabla)
  - CSV por muestra
  - gráficas opcionales (requiere matplotlib)

Uso:
  python benchmarks/report.py results/bench-20260101-120000.json
  python benchmarks/report.py --all --dir results
  python benchmarks/report.py results/bench-*.json --csv out.csv --charts
"""

import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------
def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _row(agg, group, sample):
    base = {
        "benchmark": group.get("benchmark", "?"),
        "model": group.get("model", ""),
        "device": group.get("device", ""),
        "mode": group.get("mode", ""),
        "chunking": group.get("mode", ""),
    }
    base.update(sample)
    return base


def _flatten(results):
    """Aplana un JSON de run.py (lista de grupos con 'samples') en filas."""
    rows = []
    for group in results:
        if isinstance(group, dict) and "samples" in group:
            for s in group["samples"]:
                r = dict(group)
                r.pop("samples", None)
                r.pop("load_time_s", None)
                r.update(s)
                rows.append(r)
        else:
            rows.append(group)
    return rows


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------
def markdown_table(results):
    """Tabla agregada por bloque."""
    lines = []
    lines.append("| Bloque | N | Error% | TTFB (s) | Gen (s) | Audio (s) | RTF | VRAM (GB) | RAM (GB) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group in results:
        if not (isinstance(group, dict) and "samples" in group):
            continue
        s = group["samples"]
        agg = _aggregate(s)
        label = _group_label(group)
        lines.append(
            f"| {label} | {agg['count']} | {agg['error_rate_pct']:.1f}% | "
            f"{agg['avg_ttfb_ms']/1000:.2f} | {agg['avg_total_ms']/1000:.2f} | "
            f"{agg['avg_audio_s']:.1f} | {agg['avg_rtf']:.3f} | "
            f"{agg['avg_vram_gb']:.2f} | {agg['avg_peak_ram_gb']:.2f} |"
        )
    return "\n".join(lines)


def _group_label(group):
    parts = [group.get("benchmark", "?")]
    for k in ("model", "device", "mode"):
        v = group.get(k)
        if v:
            parts.append(str(v))
    return " · ".join(parts)


# --------------------------------------------------------------------------
# Agregación
# --------------------------------------------------------------------------
def _aggregate(rows):
    ok = [r for r in rows if r.get("ok")]
    n = len(rows) or 1
    mean = lambda k: (sum(r.get(k, 0) for r in ok) / len(ok)) if ok else 0.0
    return {
        "count": len(ok),
        "error_rate_pct": round((len(rows) - len(ok)) / n * 100, 1),
        "avg_ttfb_ms": round(mean("ttfb_ms"), 1),
        "avg_total_ms": round(mean("total_ms"), 1),
        "avg_audio_s": round(mean("audio_duration_s"), 1),
        "avg_rtf": round(mean("rtf"), 3),
        "avg_vram_gb": round(mean("vram_used_gb"), 3),
        "avg_peak_ram_gb": round(mean("peak_ram_gb"), 3),
    }


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def to_csv(results, path, headers=None):
    rows = _flatten(results)
    if not rows:
        print("No hay filas que exportar.")
        return False
    keys = headers or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV: {path} ({len(rows)} filas)")
    return True


# --------------------------------------------------------------------------
# Gráficas (opcional)
# --------------------------------------------------------------------------
def charts(results, out_dir, name="benchmarks.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado; omito gráficas.")
        return

    os.makedirs(out_dir, exist_ok=True)
    labels = []
    ttfb = []
    gen = []
    rtf = []
    vram = []
    for group in [g for g in results if isinstance(g, dict) and "samples" in g]:
        a = _aggregate(group["samples"])
        labels.append(_group_label(group))
        ttfb.append(a["avg_ttfb_ms"] / 1000)
        gen.append(a["avg_total_ms"] / 1000)
        rtf.append(a["avg_rtf"])
        vram.append(a["avg_vram_gb"])

    if not labels:
        print("Sin grupos para graficar.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = range(len(labels))

    axes[0, 0].bar(x, ttfb, color="#4c72b0")
    axes[0, 0].set_title("TTFB (s)"); axes[0, 0].set_xticks(list(x)); axes[0, 0].set_xticklabels(labels, rotation=15, ha="right")

    axes[0, 1].bar(x, gen, color="#dd8452")
    axes[0, 1].set_title("Generación (s)"); axes[0, 1].set_xticks(list(x)); axes[0, 1].set_xticklabels(labels, rotation=15, ha="right")

    axes[1, 0].bar(x, rtf, color="#55a868")
    axes[1, 0].set_title("RTF"); axes[1, 0].set_xticks(list(x)); axes[1, 0].set_xticklabels(labels, rotation=15, ha="right")

    axes[1, 1].bar(x, vram, color="#c44e52")
    axes[1, 1].set_title("VRAM pico (GB)"); axes[1, 1].set_xticks(list(x)); axes[1, 1].set_xticklabels(labels, rotation=15, ha="right")

    plt.tight_layout()
    out = os.path.join(out_dir, name)
    plt.savefig(out, dpi=140)
    print(f"Gráfica: {out}")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Genera informe de benchmarks")
    p.add_argument("paths", nargs="*", help="archivos JSON de resultados")
    p.add_argument("--all", action="store_true", help="procesar todos los JSON de results/")
    p.add_argument("--dir", default=RESULTS_DIR, help="directorio de resultados")
    p.add_argument("--csv", help="ruta del CSV de salida (opcional)")
    p.add_argument("--charts", action="store_true", help="generar gráficas (matplotlib)")
    args = p.parse_args()

    if args.all:
        paths = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    else:
        paths = args.paths
    if not paths:
        print("No se indicó ningún archivo. Usa rutas o --all.")
        return

    all_rows = []
    for path in paths:
        results = load(path)
        print(f"\n=== {os.path.basename(path)} ===")
        print(markdown_table(results))
        all_rows.extend(_flatten(results))

    if args.csv:
        # Volcar solo el primer archivo si hay varios; mejor: mézclalos
        to_csv_path = args.csv
        with open(to_csv_path, "w", newline="", encoding="utf-8") as f:
            keys = list(all_rows[0].keys()) if all_rows else []
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"CSV: {to_csv_path} ({len(all_rows)} filas)")

    if args.charts:
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0] + ".png"
            charts(load(path), os.path.join(args.dir, "charts"), name=name)


if __name__ == "__main__":
    main()
