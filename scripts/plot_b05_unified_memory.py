#!/usr/bin/env python3
"""Plot B-05 Unified Memory measured results (reproducible charts).

Reads:
  docs/results/B-05_modes_warm.csv
  docs/results/B-05_cold_fault.csv

Writes:
  article/02_memory_optim/assets/B-05-mode-latency-bars.png
  article/02_memory_optim/assets/B-05-cold-vs-warm.png

Usage (from repo root):
  python scripts/plot_b05_unified_memory.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WARM_CSV = ROOT / "docs" / "results" / "B-05_modes_warm.csv"
COLD_CSV = ROOT / "docs" / "results" / "B-05_cold_fault.csv"
OUT_DIR = ROOT / "article" / "02_memory_optim" / "assets"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"
MUTED = "#8b949e"
RED = "#f85149"


def _style_axes(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, which="both", ls="--", color=GRID, alpha=0.7)


def load_warm(path: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "mode": row["mode"],
                    "first_ms": float(row["first_ms"]),
                    "median_ms": float(row["median_ms"]),
                    "p95_ms": float(row["p95_ms"]),
                }
            )
    return rows


def load_cold(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = float(row["ms"])
    return out


def plot_mode_latency(rows: list[dict[str, str | float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=BG)
    _style_axes(ax)

    modes = [str(r["mode"]) for r in rows]
    x = np.arange(len(modes))
    w = 0.25
    first = np.asarray([float(r["first_ms"]) for r in rows])
    median = np.asarray([float(r["median_ms"]) for r in rows])
    p95 = np.asarray([float(r["p95_ms"]) for r in rows])

    ax.bar(x - w, first, width=w, color=MUTED, edgecolor=GRID, label="first")
    ax.bar(x, median, width=w, color=CYAN, edgecolor=GRID, label="median")
    ax.bar(x + w, p95, width=w, color=AMBER, edgecolor=GRID, label="p95")

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("kernel time (ms)")
    ax.set_title("B-05 RTX 5090 — warm UM modes (n=16M, iters=32, warmup=1)")
    ax.set_ylim(0, 0.32)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def plot_cold_vs_warm(cold: dict[str, float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.0), facecolor=BG)
    _style_axes(ax)

    labels = ["first (cold)", "median (warm)", "mean (misleading)"]
    vals = [cold["first"], cold["median"], cold["mean"]]
    colors = [RED, CYAN, MUTED]

    xs = np.arange(len(labels))
    bars = ax.bar(xs, vals, color=colors, edgecolor=GRID, width=0.55)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("kernel time (ms, log scale)")
    ax.set_title("B-05 RTX 5090 — cold fault start (warmup=0, runs=3)")

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v * 1.15,
            f"{v:g} ms",
            ha="center",
            va="bottom",
            color=FG,
            fontsize=10,
        )

    ratio = cold["first"] / cold["median"]
    ax.text(
        0.98,
        0.92,
        f"first / median ≈ {ratio:.0f}×\n(stop threshold in text: 1.5×)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=FG,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=GRID),
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warm = load_warm(WARM_CSV)
    cold = load_cold(COLD_CSV)

    d1 = OUT_DIR / "B-05-mode-latency-bars.png"
    d2 = OUT_DIR / "B-05-cold-vs-warm.png"
    plot_mode_latency(warm, d1)
    plot_cold_vs_warm(cold, d2)
    print(f"wrote {d1.relative_to(ROOT)}")
    print(f"wrote {d2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
