# C-09 Named Barrier — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 正文：`article/03_compute_primitives/C-09*.md`  
> 示例：`examples/03_compute_primitives/09_named_barrier.cu`

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CC | sm_120 |
| SMs / grid | **170** / **170**（每 SM 一个 block） |
| block / count | 64 / 64（warp0 生产，warp1 消费） |
| barrier | full=1，empty=2 |
| phases / work / tail | 1024 / 512 / 512 |
| runs / warmup | 7 / 2 |
| 校验和 | 每 block `524800` = `1024*1025/2`，cta / named / named_pre 均通过 |
| 主证据 | CUDA event **median**；**同 stream 记 event** |

## 复现命令

```bash
./bin/03_compute_primitives_09_named_barrier --mode sweep
./bin/03_compute_primitives_09_named_barrier --mode sweep_tail
./bin/03_compute_primitives_09_named_barrier --mode modes
```

Windows：`.\bin\03_compute_primitives_09_named_barrier.exe --mode sweep`

`CMAKE_CUDA_ARCHITECTURES=120`。增删 `.cu` 后先重跑 cmake。

## Sweep（主曲线：固定 tail=512，扫 work）

| work | cta_ms | named_ms | cta/named |
|---:|---:|---:|---:|
| 0 | 0.9190 | 0.8735 | **1.05×** |
| 32 | 1.0097 | 0.8743 | **1.15×** |
| 64 | 1.0580 | 0.8742 | **1.21×** |
| 128 | 1.1552 | 0.8724 | **1.32×** |
| 256 | 1.3486 | 0.8745 | **1.54×** |
| 512 | 1.7347 | 0.8820 | **1.97×** |
| 1024 | 2.5119 | 1.6568 | **1.52×** |
| 2048 | 4.0591 | 3.2072 | **1.27×** |
| 4096 | 7.1569 | 6.3048 | **1.14×** |

## Sweep tail（固定 work=512，扫 tail）

| tail | cta_ms | named_ms | cta/named |
|---:|---:|---:|---:|
| 0 | 0.9155 | 0.8806 | **1.04×** |
| 32 | 1.0076 | 0.8817 | **1.14×** |
| 64 | 1.0568 | 0.8816 | **1.20×** |
| 128 | 1.1530 | 0.8816 | **1.31×** |
| 256 | 1.3486 | 0.8806 | **1.53×** |
| 512 | 1.7358 | 0.8806 | **1.97×** |
| 1024 | 2.5088 | 1.6483 | **1.52×** |
| 2048 | 4.0581 | 3.1977 | **1.27×** |
| 4096 | 7.1547 | 6.2945 | **1.14×** |

## Modes

| tag | median_ms | 对照 |
|---|---:|---|
| cta | 1.7347 | 基线 |
| named | 0.8827 | cta/named = **1.965×** |
| named_pre | 1.7224 | cta/named_pre = **1.007×** |
| indep_cta | 0.4034 | work_a=2048，work_b=32，phases=128，block=256 |
| indep_named | 0.4170 | cta/named = **0.967×** |

## 怎么读

`named` 在 work≤512（tail 固定 512）时停在 **0.87～0.88 ms**。消费者的 work 藏在生产者的 tail 下面。`cta` 从 0.92 ms 爬到 1.73 ms，两边相等时 **1.97×**。work 或 tail 再加大，长的那一侧变成杆，比值收到 **1.14×**。某一侧为 0 时是 **1.04×～1.05×**。

`named_pre` 把 FMA 放回 arrive 之前，墙钟回到 cta（1.007×）。`indep` 的 kernel 墙钟是 **0.97×**：短组先结束不缩短 kernel，named 还略慢一点。
