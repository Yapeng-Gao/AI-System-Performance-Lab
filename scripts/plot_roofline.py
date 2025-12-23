import numpy as np
import matplotlib.pyplot as plt
import sys
import json

def roofline_plot(data, hw, out):
    oi = np.logspace(-2, 3, 500)

    mem_roof = hw["bw"] * oi
    comp_roof = np.ones_like(oi) * hw["tflops"]

    plt.figure(figsize=(8,6))
    plt.loglog(oi, mem_roof, '--', label="Memory Roof")
    plt.loglog(oi, comp_roof, '-', label="Compute Roof")

    for name, p in data.items():
        plt.scatter(p["oi"], p["perf"], s=80)
        plt.text(p["oi"]*1.1, p["perf"], name)

    plt.xlabel("Operational Intensity (FLOP / Byte)")
    plt.ylabel("Performance (TFLOPs)")
    plt.title(hw["name"] + " Roofline")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.savefig(out)
    plt.close()

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

    hw = {
        "name": sys.argv[2],
        "bw": float(sys.argv[3]),     # GB/s
        "tflops": float(sys.argv[4])  # TFLOPs
    }

    roofline_plot(data, hw, sys.argv[5])
