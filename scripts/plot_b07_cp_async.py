#!/usr/bin/env python3
"""Plot B-07 cp.async pipeline measured results (reproducible charts).

Reads:
  docs/results/B-07_sweep.csv
  docs/results/B-07_modes.csv

Writes:
  article/02_memory_optim/assets/B-07-speedup-vs-fma.png
  article/02_memory_optim/assets/B-07-mode-speedup-bars.png

Usage (from repo root):
  python scripts/plot_b07_cp_async.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "docs" / "results" / "B-07_sweep.csv"
MODES_CSV = ROOT / "docs" / "results" / "B-07_modes.csv"
OUT_DIR = ROOT / "article" / "02_memory_optim" / "assets"

# Dark theme to match column covers (cyan / amber accents)
BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"
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
        "speedup_pipe2": [],
        "speedup_pipe4": [],
    }
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows["fma_iters"].append(float(row["fma_iters"]))
            rows["speedup_pipe2"].append(float(row["speedup_pipe2"]))
            rows["speedup_pipe4"].append(float(row["speedup_pipe4"]))
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
    y2 = np.asarray(sweep["speedup_pipe2"])
    y4 = np.asarray(sweep["speedup_pipe4"])

    ax.plot(x, y2, "o-", color=CYAN, lw=2, ms=7, label="pipe2 / sync")
    ax.plot(x, y4, "s--", color=AMBER, lw=2, ms=7, label="pipe4 / sync")
    ax.axhline(1.0, color=MUTED, ls="-", lw=1.5, label="break-even (1.0×)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("fma_iters (≈ arithmetic intensity)")
    ax.set_ylabel("speedup (sync_ms / pipe_ms)")
    ax.set_title("B-07 RTX 5090 — pipeline speedup vs intensity (CUDA event median)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax.set_ylim(0.90, 1.38)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def plot_mode_bars(modes: list[str], speedups: list[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=BG)
    _style_axes(ax)

    colors = []
    for m, s in zip(modes, speedups):
        if m == "pipe2":
            colors.append(CYAN)
        elif s < 1.0:
            colors.append("#f85149")
        else:
            colors.append(AMBER)

    xs = np.arange(len(modes))
    bars = ax.bar(xs, speedups, color=colors, edgecolor=GRID, width=0.65)
    ax.axhline(1.0, color=MUTED, ls="-", lw=1.5, label="sync baseline (1.0×)")

    ax.set_xticks(xs)
    ax.set_xticklabels(modes)
    ax.set_ylabel("speedup vs sync")
    ax.set_title("B-07 RTX 5090 — fixed modes at fma_iters=8 (CUDA event median)")
    ax.set_ylim(0.85, 1.20)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)

    for bar, s in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            s + 0.012,
            f"{s:.2f}×",
            ha="center",
            va="bottom",
            color=FG,
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = load_sweep(SWEEP_CSV)
    modes, speedups = load_modes(MODES_CSV)

    d1 = OUT_DIR / "B-07-speedup-vs-fma.png"
    d2 = OUT_DIR / "B-07-mode-speedup-bars.png"
    plot_speedup_vs_fma(sweep, d1)
    plot_mode_bars(modes, speedups, d2)
    print(f"wrote {d1.relative_to(ROOT)}")
    print(f"wrote {d2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
