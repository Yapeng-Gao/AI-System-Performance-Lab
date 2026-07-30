#!/usr/bin/env python3
"""Plot B-09 layout / transpose measured results.

Reads:
  docs/results/B-09_sweep.csv
  docs/results/B-09_modes.csv

Writes:
  article/02_memory_optim/assets/B-09-speedup-vs-touch.png
  article/02_memory_optim/assets/B-09-transpose-gbps-bars.png

Usage (from repo root):
  python scripts/plot_b09_layout.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "B-09_sweep.csv"
MODES_CSV = ROOT / "docs" / "results" / "B-09_modes.csv"
OUT_DIR = ROOT / "article" / "02_memory_optim" / "assets"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"
GREEN = "#3fb950"
MUTED = "#8b949e"


def _style_axes(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, which="both", ls="--", color=GRID, alpha=0.7)


def load_sweep(path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {
        "touch_fields": [],
        "aos_gbps": [],
        "soa_gbps": [],
        "speedup_soa": [],
    }
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows["touch_fields"].append(float(row["touch_fields"]))
            rows["aos_gbps"].append(float(row["aos_gbps"]))
            rows["soa_gbps"].append(float(row["soa_gbps"]))
            rows["speedup_soa"].append(float(row["speedup_soa"]))
    return rows


def load_modes(path: Path) -> tuple[list[str], list[float]]:
    modes: list[str] = []
    gbps: list[float] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            modes.append(row["mode"])
            gbps.append(float(row["useful_gbps"]))
    return modes, gbps


def plot_speedup_vs_touch(sweep: dict[str, list[float]], out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5.2), facecolor=BG)
    _style_axes(ax1)

    x = np.asarray(sweep["touch_fields"])
    ax1.plot(x, sweep["speedup_soa"], "o-", color=CYAN, lw=2.2, label="SoA / AoS speedup")
    ax1.axhline(1.0, color=MUTED, ls=":", lw=1.2)
    ax1.set_xlabel("touch_fields")
    ax1.set_ylabel("speedup (aos_ms / soa_ms)")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    _style_axes(ax2)
    ax2.plot(x, sweep["aos_gbps"], "s--", color=AMBER, lw=1.6, label="AoS useful GB/s")
    ax2.plot(x, sweep["soa_gbps"], "^--", color=GREEN, lw=1.6, label="SoA useful GB/s")
    ax2.set_ylabel("useful GB/s (R+W touched fields)")
    ax2.spines["right"].set_color(GRID)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, facecolor=BG, edgecolor=GRID, labelcolor=FG)

    ax1.set_title("B-09: SoA vs AoS vs touch_fields")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def plot_transpose_bars(modes: list[str], gbps: list[float], out: Path) -> None:
    # Keep transpose family (+ copy) only
    keep = {"copy", "transpose_naive", "transpose_tiled", "transpose_pad"}
    pairs = [(m, g) for m, g in zip(modes, gbps) if m in keep]
    if not pairs:
        print(f"skip transpose plot: no matching modes in {MODES_CSV}", file=sys.stderr)
        return
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [CYAN if m == "copy" else AMBER if "naive" in m else GREEN for m in labels]

    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=BG)
    _style_axes(ax)
    ax.bar(labels, vals, color=colors, edgecolor=GRID)
    ax.set_ylabel("useful GB/s (R+W)")
    ax.set_title("B-09: copy vs transpose family")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", color=FG, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    if not SWEEP_CSV.is_file():
        print(f"missing {SWEEP_CSV}", file=sys.stderr)
        return 1
    sweep = load_sweep(SWEEP_CSV)
    plot_speedup_vs_touch(sweep, OUT_DIR / "B-09-speedup-vs-touch.png")

    if MODES_CSV.is_file():
        modes, gbps = load_modes(MODES_CSV)
        plot_transpose_bars(modes, gbps, OUT_DIR / "B-09-transpose-gbps-bars.png")
    else:
        print(f"warn: missing {MODES_CSV}, skip transpose bars", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
