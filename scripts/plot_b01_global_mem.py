#!/usr/bin/env python3
"""Plot B-01 global memory measured results.

Reads:
  docs/results/B-01_modes.csv

Writes:
  article/02_memory_optim/assets/B-01-mode-gbps-bars.png
  article/02_memory_optim/assets/B-01-speedup-vs-aligned.png

Usage (from repo root):
  python scripts/plot_b01_global_mem.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES_CSV = ROOT / "docs" / "results" / "B-01_modes.csv"
OUT_DIR = ROOT / "article" / "02_memory_optim" / "assets"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"
GREEN = "#3fb950"
MUTED = "#8b949e"

# Preferred order for bars
MODE_ORDER = ["misaligned", "aligned", "float4", "ldg_nt"]
MODE_COLOR = {
    "misaligned": AMBER,
    "aligned": CYAN,
    "float4": GREEN,
    "ldg_nt": MUTED,
}


def _style_axes(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, which="both", ls="--", color=GRID, alpha=0.7)


def load_modes(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mode = (row.get("mode") or "").strip()
            if not mode or mode.startswith("#"):
                continue
            try:
                rows.append(
                    {
                        "mode": mode,
                        "median_ms": float(row["median_ms"]),
                        "gbps": float(row["gbps"]),
                        "speedup_vs_aligned": float(row["speedup_vs_aligned"]),
                    }
                )
            except (KeyError, ValueError) as e:
                print(f"skip row {row!r}: {e}", file=sys.stderr)
    return rows


def _sorted(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rank = {m: i for i, m in enumerate(MODE_ORDER)}
    return sorted(rows, key=lambda r: rank.get(str(r["mode"]), 100))


def plot_gbps_bars(rows: list[dict[str, float | str]], out: Path) -> None:
    labels = [str(r["mode"]) for r in rows]
    vals = [float(r["gbps"]) for r in rows]
    colors = [MODE_COLOR.get(m, CYAN) for m in labels]

    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=BG)
    _style_axes(ax)
    ax.bar(labels, vals, color=colors, edgecolor=GRID)
    ax.set_ylabel("useful GB/s (R+W)")
    ax.set_title("B-01: Global Memory modes (median)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", color=FG, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def plot_speedup_bars(rows: list[dict[str, float | str]], out: Path) -> None:
    labels = [str(r["mode"]) for r in rows]
    vals = [float(r["speedup_vs_aligned"]) for r in rows]
    colors = [MODE_COLOR.get(m, CYAN) for m in labels]

    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=BG)
    _style_axes(ax)
    ax.bar(labels, vals, color=colors, edgecolor=GRID)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.2)
    ax.set_ylabel("speedup (aligned_ms / mode_ms)")
    ax.set_title("B-01: speedup vs aligned")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}x", ha="center", va="bottom", color=FG, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    if not MODES_CSV.is_file():
        print(f"missing {MODES_CSV}", file=sys.stderr)
        return 1
    rows = _sorted(load_modes(MODES_CSV))
    if not rows:
        print(
            f"no data rows in {MODES_CSV}; run binary --mode modes and paste CSV first",
            file=sys.stderr,
        )
        return 2
    plot_gbps_bars(rows, OUT_DIR / "B-01-mode-gbps-bars.png")
    plot_speedup_bars(rows, OUT_DIR / "B-01-speedup-vs-aligned.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
