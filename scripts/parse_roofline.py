import csv
import sys

def parse_csv(path):
    data = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            metric = row[0]
            try:
                value = float(row[-1])
                data[metric] = value
            except:
                pass
    return data

def main(csv_file):
    d = parse_csv(csv_file)

    bytes_read = d.get("dram__bytes_read.sum", 0)
    bytes_write = d.get("dram__bytes_write.sum", 0)
    total_bytes = bytes_read + bytes_write

    fadd = d.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0)
    fmul = d.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0)
    ffma = d.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)

    flops = fadd + fmul + 2 * ffma

    time_ns = d.get("gpu__time_duration.sum", 1)
    time_s = time_ns * 1e-9

    bw = total_bytes / time_s / 1e9
    tflops = flops / time_s / 1e12
    oi = flops / total_bytes if total_bytes > 0 else 0

    print(f"Bandwidth (GB/s): {bw:.2f}")
    print(f"TFLOPs:           {tflops:.2f}")
    print(f"Operational Intensity (FLOP/Byte): {oi:.4f}")

if __name__ == "__main__":
    main(sys.argv[1])
