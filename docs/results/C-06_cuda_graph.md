# C-06 CUDA Graph — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 正文：`article/03_compute_primitives/C-06*.md`  
> 示例：`examples/03_compute_primitives/06_cuda_graph.cu`

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CC | sm_120 |
| SMs | 170 |
| n | 4096（小 footprint，强调 launch） |
| block / grid | 256 / 16 |
| chain_reps | 200 |
| runs / warmup | 7 / 2 |
| 主证据 | CUDA event **median**（热路径；instantiate **不计**） |

## 复现命令

```bash
./bin/03_compute_primitives_06_cuda_graph --mode sweep
./bin/03_compute_primitives_06_cuda_graph --mode sweep_work
./bin/03_compute_primitives_06_cuda_graph --mode modes
```

## Sweep（主曲线：stream/graph vs n_nodes，work=1）

| n_nodes | stream_ms | graph_ms | stream/graph | instantiate_ms |
|---:|---:|---:|---:|---:|
| 1 | 1.180 | 0.871 | **1.36×** | 0.424 |
| 2 | 2.217 | 0.681 | **3.26×** | 0.038 |
| 4 | 3.224 | 0.837 | **3.85×** | 0.030 |
| 8 | 5.080 | 1.244 | **4.08×** | 0.029 |
| 16 | 9.065 | 2.472 | **3.67×** | 0.045 |
| 32 | 18.085 | 4.522 | **4.00×** | 0.086 |
| 64 | 36.039 | 9.022 | **4.00×** | 0.271 |

![C-06：stream/graph vs n_nodes](../../article/03_compute_primitives/assets/C-06-speedup-vs-nnodes.png)

### 怎么读

1. **节点≥2 后加速比稳定在 ~3.3～4.1×**（峰值 @8≈4.08×）；比值平台，**不是随节点无限抬升**——两边几乎都线性涨。
2. **stream 近似随节点线性**（1.18→36 ms）：短核链上 launch 税主导。
3. **nodes=1 仅 1.36×**：单节点图几乎没有「多次提交→一次 Launch」可摊，收益薄——对齐「节点太少别上图」。
4. instantiate 0.03～0.4 ms 量级，远小于 `chain_reps=200` 的热路径墙钟；**勿摊进 median**。首次/冷 instantiate 偏高属正常。

## Sweep work（副曲线：固定 n_nodes=16）

| work | stream_ms | graph_ms | stream/graph |
|---:|---:|---:|---:|
| 0 | 9.062 | 1.653 | **5.48×** |
| 1 | 9.074 | 2.472 | **3.67×** |
| 8 | 9.049 | 2.473 | **3.66×** |
| 64 | 9.109 | 2.881 | **3.16×** |
| 512 | 9.071 | 5.335 | **1.70×** |
| 4096 | 23.702 | 23.535 | **1.01×** |

![C-06：stream/graph vs work](../../article/03_compute_primitives/assets/C-06-speedup-vs-work.png)

### 怎么读

1. **work=0 最大（5.48×）**；加重到 512→1.70×，**4096→1.01×**——body 主导端到端时 Graph 几乎无感。假设②成立。
2. stream 在 work≤512 几乎钉在 ~9 ms（仍 launch 墙）；到 4096 才跳到 ~24 ms（算力/访存开始主导）。
3. graph 更早随 work 变厚（1.65→5.34→23.5）——replay 把提交税削薄后，单核 body 暴露出来。

## Modes（定点 n_nodes=16, work=1）

| tag | median_ms | instantiate_ms |
|---|---:|---:|
| stream | 9.110 | — |
| graph | 2.473 | **0.241**（单独一行） |

speedup **stream/graph=3.68×**；verify 全过。

## CSV

- `docs/results/C-06_sweep.csv`
- `docs/results/C-06_sweep_work.csv`

```bash
python scripts/plot_c06_cuda_graph.py
```
