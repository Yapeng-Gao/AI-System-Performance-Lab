#!/usr/bin/env python3
"""Plot B-06 pinned/DMA measured results (reproducible charts).

Reads:
  docs/results/B-06_modes.csv
  docs/results/B-06_overlap.csv

Writes:
  article/02_memory_optim/assets/B-06-mode-gbs-bars.png
  article/02_memory_optim/assets/B-06-overlap-median-bars.png

Usage (from repo root):
  python scripts/plot_b06_pinned_dma.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES_CSV = ROOT / "docs" / "results" / "B-06_modes.csv"
OVERLAP_CSV = ROOT / "docs" / "results" / "B-06_overlap.csv"
OUT_DIR = ROOT / "article" / "02_memory_optim" / "assets"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39c5cf"
AMBER = "#f0a202"
MUTED = "#8b949e"
GREEN = "#3fb950"


def _style_axes(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, which="both", ls="--", color=GRID, alpha=0.7)


def load_modes(path: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "mode": row["mode"],
                    "label": row["label"],
                    "median_ms": float(row["median_ms"]),
                    "gbs": float(row["gbs"]),
                }
            )
    return rows


def load_overlap(path: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "case": row["case"],
                    "role": row["role"],
                    "median_ms": float(row["median_ms"]),
                }
            )
    return rows


def plot_mode_gbs(rows: list[dict[str, str | float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=BG)
    _style_axes(ax)

    labels = [str(r["label"]) for r in rows]
    gbs = np.asarray([float(r["gbs"]) for r in rows])
    modes = [str(r["mode"]) for r in rows]

    colors = []
    for m in modes:
        if m == "pinned":
            colors.append(CYAN)
        elif m == "bidir":
            colors.append(GREEN)
        elif m.startswith("overlap"):
            colors.append(AMBER)
        else:
            colors.append(MUTED)

    xs = np.arange(len(labels))
    bars = ax.bar(xs, gbs, color=colors, edgecolor=GRID, width=0.7)
    ax.axhline(52.42, color=CYAN, ls=":", lw=1.2, alpha=0.8, label="pinned H2D ceiling")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("GB/s")
    ax.set_title("B-06 RTX 5090 — mode effective bandwidth (CUDA event median)")
    ax.set_ylim(0, 110)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)

    for bar, v in zip(bars, gbs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.5,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            color=FG,
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_median(rows: list[dict[str, str | float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=BG)
    _style_axes(ax)

    order = ["ceiling", "overlap", "serial"]
    by_role = {str(r["role"]): r for r in rows}
    roles = [r for r in order if r in by_role]
    labels = {
        "ceiling": "pinned (H2D ceiling)",
        "overlap": "overlap (iters=256)",
        "serial": "serial (iters=256)",
    }
    colors = {"ceiling": CYAN, "overlap": AMBER, "serial": MUTED}

    xs = np.arange(len(roles))
    vals = [float(by_role[r]["median_ms"]) for r in roles]
    bars = ax.bar(
        xs,
        vals,
        color=[colors[r] for r in roles],
        edgecolor=GRID,
        width=0.55,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([labels[r] for r in roles])
    ax.set_ylabel("median end-to-end (ms)")
    ax.set_title("B-06 RTX 5090 — overlap hides kernel (iters=256)")
    ax.set_ylim(0, 6.2)

    ceil = float(by_role["ceiling"]["median_ms"])
    ax.axhline(ceil, color=CYAN, ls=":", lw=1.2, alpha=0.85)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.08,
            f"{v:.3f} ms",
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
    modes = load_modes(MODES_CSV)
    overlap = load_overlap(OVERLAP_CSV)

    d1 = OUT_DIR / "B-06-mode-gbs-bars.png"
    d2 = OUT_DIR / "B-06-overlap-median-bars.png"
    plot_mode_gbs(modes, d1)
    plot_overlap_median(overlap, d2)
    print(f"wrote {d1.relative_to(ROOT)}")
    print(f"wrote {d2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
