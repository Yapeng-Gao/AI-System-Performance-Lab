# C-07 Persistent Kernel — 实测摘要

> 状态：✅ 已测（RTX 5090 / `sm_120`）  
> 正文：`article/03_compute_primitives/C-07*.md`  
> 示例：`examples/03_compute_primitives/07_persistent.cu`

## 平台

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CC | sm_120 |
| SMs | 170 |
| blocks_per_sm / persist_grid | **6** / **1020**（=170×6） |
| oversub grid | 8160（×8） |
| n_tasks 默认 | 4096 |
| work 默认 | 1（FMA iters） |
| block / launch grid | 256 / **1**（一任务一发） |
| runs / warmup | 7 / 2 |
| 主证据 | CUDA event **median**；**同 stream 记 event** |

## 复现命令

```bash
./bin/03_compute_primitives_07_persistent --mode sweep
./bin/03_compute_primitives_07_persistent --mode sweep_work
./bin/03_compute_primitives_07_persistent --mode modes
```

## Sweep（主曲线：launch/persistent vs n_tasks，work=1）

| n_tasks | launch_ms | persist_ms | launch/persistent |
|---:|---:|---:|---:|
| 64 | 0.2100 | 0.0065 | **32.48×** |
| 256 | 0.7987 | 0.0062 | **129.3×** |
| 1024 | 3.1160 | 0.0074 | **421.5×** |
| 4096 | 12.1844 | 0.0085 | **1431×** |
| 16384 | 48.7068 | 0.0130 | **3740×** |

![C-07：launch/persistent vs n_tasks](../../article/03_compute_primitives/assets/C-07-speedup-vs-ntasks.png)

### 怎么读

1. **launch 近似按任务数线性**：0.21→48.7 ms，约 **3.0 µs/次**（12.18/4096、48.71/16384 都贴 3.0µs）。短任务上端到端就是 launch 税。
2. **persist 几乎贴地**：64～4096 task 都在 6～9µs；16384 才 13µs。一次 occupancy 网格 + 拉活，墙钟几乎不跟任务数走。
3. **千倍级加速比是两件事叠在一起**：① 少付 N 次 launch；② launch 侧永远是 **1 block 串行**，persist 侧是 **1020 block 常驻**。不要把 1431× 贴到 Graph（C-06 ~4×）或 GEMM 幻灯片上。
4. 主看形状：任务越多，一任务一发越亏。禁止为「好看」改成 batch。

## Sweep work（副曲线：固定 n_tasks=4096）

| work | launch_ms | persist_ms | launch/persistent |
|---:|---:|---:|---:|
| 0 | 12.3437 | 0.0080 | **1543×** |
| 1 | 12.2741 | 0.0079 | **1553×** |
| 8 | 12.3182 | 0.0069 | **1774×** |
| 64 | 12.3428 | 0.0088 | **1403×** |
| 512 | 12.2818 | 0.0192 | **640×** |
| 4096 | 29.4231 | 0.0938 | **314×** |

![C-07：launch/persistent vs work](../../article/03_compute_primitives/assets/C-07-speedup-vs-work.png)

### 怎么读

1. **work≤64**：launch 锁在 ~12.3 ms（税主导）；persist 仍 ~8µs。
2. **work=4096**：launch 12.3→**29.4 ms**（body 开始露头）；persist 0.008→**0.094 ms**。加速比 **1550×→314×**，在收，**收不到 C-06 那种 1.01×**。
3. 收不到 1× 不是测废了。C-06 两边占用相近、只摊提交；本章 launch 仍是 1-block 串行，persist 仍是整卡。body 变重只会削弱 launch 税占比，削不掉并行度差。
4. **不要**为了逼近 1× 去加长 `work` 或改成 batch。314× 已说明「任务变重，杠杆变弱」。

## Modes / oversub（定点 n_tasks=4096，work=1）

| tag | grid | median_ms | 备注 |
|---|---:|---:|---|
| launch | 1 | 12.4248 | 一任务一发 |
| persistent | 1020 | 0.0079 | Occupancy API；`launch/persistent`=**1572×** |
| oversub | 8160 | 0.0117 | 比 persist **慢**；`oversub/persistent`=**1.48×**（与 `launch/persistent` 同向：A/B = tA/tB） |

过大不是常驻：多出来的 block 要等调度，队列多半已被住满的那一波抽干。

## 读数纪律

- 禁止把 ncu 附着墙钟当结论。
- 禁止为了让 persist 赢或为了收到 1× 而 batch。
- 千倍数字只描述「一任务一发」反模式，不是 Persistent 的通用加速比。
