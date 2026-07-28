#!/usr/bin/env python3
"""Plot B-08 TMA measured results (reproducible charts).

Reads:
  docs/results/B-08_sweep.csv
  docs/results/B-08_modes.csv

Writes:
  article/02_memory_optim/assets/B-08-speedup-vs-fma.png
  article/02_memory_optim/assets/B-08-mode-speedup-bars.png

Usage (from repo root):
  python scripts/plot_b08_tma.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "B-08_sweep.csv"
MODES_CSV = ROOT / "docs" / "results" / "B-08_modes.csv"
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
        "fma_iters": [],
        "speedup_bulk1d": [],
        "speedup_tensor2d": [],
        "speedup_pipe2": [],
    }
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows["fma_iters"].append(float(row["fma_iters"]))
            rows["speedup_bulk1d"].append(float(row["speedup_bulk1d"]))
            rows["speedup_tensor2d"].append(float(row["speedup_tensor2d"]))
            rows["speedup_pipe2"].append(float(row["speedup_pipe2"]))
    return rows


def load_modes(path: Path) -> tuple[list[str], list[float]]:
    modes: list[str] = []
    speedups: list[float] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            modes.append(row["mode"])
            speedups.append(float(row["speedup_vs_sync"]))
    return modes, speedups


def plot_speedup_vs_fma(sweep: dict[str, list[float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=BG)
    _style_axes(ax)

    x = np.asarray(sweep["fma_iters"])
    ax.plot(x, sweep["speedup_bulk1d"], "o-", color=CYAN, label="bulk1d", lw=2)
    ax.plot(x, sweep["speedup_tensor2d"], "s-", color=AMBER, label="tensor2d", lw=2)
    ax.plot(x, sweep["speedup_pipe2"], "^-", color=GREEN, label="pipe2", lw=2)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.5, label="parity")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("fma_iters (proxy arithmetic intensity)")
    ax.set_ylabel("speedup (sync / mode)")
    ax.set_title("B-08 TMA: speedup vs intensity")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_mode_bars(modes: list[str], speedups: list[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=BG)
    _style_axes(ax)
    colors = [CYAN if m != "sync" else MUTED for m in modes]
    ax.bar(modes, speedups, color=colors, edgecolor=GRID)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.5)
    ax.set_ylabel("speedup vs sync")
    ax.set_title("B-08 TMA: fixed-mode speedup")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    if not SWEEP_CSV.is_file() or not MODES_CSV.is_file():
        print(
            f"Missing CSV. Run on sm_90+ then write:\n"
            f"  {SWEEP_CSV}\n  {MODES_CSV}\n"
            f"See docs/results/B-08_tma.md",
            file=sys.stderr,
        )
        return 1
    sweep = load_sweep(SWEEP_CSV)
    modes, speedups = load_modes(MODES_CSV)
    plot_speedup_vs_fma(sweep, OUT_DIR / "B-08-speedup-vs-fma.png")
    plot_mode_bars(modes, speedups, OUT_DIR / "B-08-mode-speedup-bars.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
