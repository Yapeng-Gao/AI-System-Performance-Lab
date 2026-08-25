#!/usr/bin/env python3
"""Plot A-10 measured Roofline (RTX 5090 probes).

Reads:  docs/results/A-10_roofline_rtx5090.csv
Writes: article/01_cuda_basic/assets/A-10-measured-roofline.png

Roofline roofs use *measured* BW and TFLOPS from the same CSV meta row.
Copy AI≈0 is plotted at 0.01 FLOP/byte for log-x.

Usage (repo root):
  python scripts/plot_a10_roofline.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "results" / "A-10_roofline_rtx5090.csv"
OUT = ROOT / "article" / "01_cuda_basic" / "assets" / "A-10-measured-roofline.png"

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
    ax.grid(True, which="both", ls="--", color=GRID, alpha=0.55)


def main() -> None:
    probes: dict[str, dict[str, float | str]] = {}
    with CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            probes[row["probe"]] = {
                "label": row["label"],
                "ai": float(row["ai_flop_per_byte"]),
                "tflops": float(row["perf_tflops"]),
                "bw": float(row["bw_gbs"]),
            }

    meta = probes["meta"]
    bw = float(meta["bw"])  # GB/s
    peak_tflops = float(meta["tflops"])
    ridge = float(meta["ai"])

    oi = np.logspace(-2, 3.2, 400)
    # Attainable TFLOPS = min(BW_GBs * AI / 1000, peak_TFLOPS)
    mem_roof = bw * oi / 1000.0
    comp_roof = np.full_like(oi, peak_tflops)
    attainable = np.minimum(mem_roof, comp_roof)

    fig, ax = plt.subplots(figsize=(8.8, 5.4), facecolor=BG)
    _style(ax)
    ax.loglog(oi, attainable, color=CYAN, lw=2.2, label="measured roof", zorder=2)
    ax.axvline(ridge, color=MUTED, ls=":", lw=1.4, label=f"ridge≈{ridge:.1f}")

    # Copy: show as performance in "effective" terms on BW axis —
    # for scatter we plot TFLOPS = BW*AI/1000 at the display AI.
    copy = probes["copy"]
    copy_ai = float(copy["ai"])
    copy_tflops = bw * copy_ai / 1000.0
    ax.scatter([copy_ai], [copy_tflops], s=90, color=AMBER, zorder=4)
    ax.annotate(
        "A copy\n(~1954 GB/s)",
        (copy_ai, copy_tflops),
        textcoords="offset points",
        xytext=(8, 10),
        color=AMBER,
        fontsize=9,
    )

    fma = probes["fma"]
    ax.scatter([float(fma["ai"])], [float(fma["tflops"])], s=90, color=CYAN, zorder=4)
    ax.annotate(
        f"B FMA\n({float(fma['tflops']):.1f} TFLOPS)",
        (float(fma["ai"]), float(fma["tflops"])),
        textcoords="offset points",
        xytext=(-70, -28),
        color=CYAN,
        fontsize=9,
    )

    ax.set_xlabel("Arithmetic intensity (FLOP/byte)  [copy AI≈0 → plotted at 0.01]")
    ax.set_ylabel("Attainable / achieved (TFLOPS)")
    ax.set_title("A-10 RTX 5090 — measured Roofline (event median probes)")
    leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    for text in leg.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
