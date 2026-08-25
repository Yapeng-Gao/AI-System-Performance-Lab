#!/usr/bin/env python3
"""Plot A-08 async pipeline median bars (RTX 5090).

Reads:  docs/results/A-08_async_pipeline_rtx5090.csv
Writes: article/01_cuda_basic/assets/A-08-mode-median-bars.png

Usage (repo root):
  python scripts/plot_a08_async_pipeline.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "results" / "A-08_async_pipeline_rtx5090.csv"
OUT = ROOT / "article" / "01_cuda_basic" / "assets" / "A-08-mode-median-bars.png"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"
MUTED = "#8b949e"


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, axis="y", ls="--", color=GRID, alpha=0.7)


def main() -> None:
    rows: list[dict[str, str | float]] = []
    with CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "label": row["label"],
                    "median_ms": float(row["median_ms"]),
                }
            )

    labels = [str(r["label"]) for r in rows]
    vals = [float(r["median_ms"]) for r in rows]
    colors = [AMBER, CYAN, MUTED]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
    _style(ax)
    bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
    ax.set_ylabel("CUDA event median (ms)")
    ax.set_title("A-08 RTX 5090 — serial vs depth/breadth pipeline")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            color=FG,
            fontsize=10,
        )
    # Annotate ratios vs B (index 1)
    b = vals[1]
    ax.text(
        0.98,
        0.95,
        f"A/B={vals[0]/b:.2f}×   C/B={vals[2]/b:.2f}×",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
