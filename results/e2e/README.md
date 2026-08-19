# End-to-end serving results

Full-model A/B on TPU v6e-4 (TP=4) via `sgl_jax.bench_serving`:
sharegpt, 160 prompts, max-concurrency 32, warmup 8, request-rate inf.
Both sides are the **same build**; the baseline side sets
`KDA_FORCE_BASELINE=1` so every ablation switch is off (== upstream 4-stage
path). Reproduce with `bench/e2e_serving.sh <model-dir> <tag>`.

> **Provenance.** These runs predate submodule commit `043ff29a`. Between them
> and the pin that shipped afterwards (`d88ed081`), the backend no longer read
> `KDA_FORCE_BASELINE` and no longer threaded `gate_lower_bound` into the
> kernel, so an image built from that pin would have compared a config against
> itself. `043ff29a` restores both. Reproduce only with a pin at `043ff29a` or
> later; on an image built from `d88ed081` these numbers do not reproduce.

## Kimi-Linear-48B (real weights) — unbounded gate

`safe_gate` / `head_block` do not apply (finite-Neumann solve is only
numerically safe under a bounded gate), so this isolates the **structural**
switches: `fuse` + `unified_layout` + `flat_grid`.

| Metric | upstream KDA | this kernel | speedup |
|---|---|---|---|
| Total token throughput | 1959 tok/s | **2412 tok/s** | **1.23x** |
| Input / output throughput | 1194 / 765 tok/s | 1471 / 941 tok/s | 1.23x |
| TTFT p50 | 244 ms | **164 ms** | **1.49x** |
| TTFT p99 | 2112 ms | 1171 ms | 1.80x |
| TPOT p50 | 36.7 ms | **28.9 ms** | **1.27x** |
| TPOT p99 | 180 ms | 107 ms | 1.69x |
| Wall time, 160 requests | 45.2 s | 36.7 s | 1.23x |

Accuracy on the same server: **gsm8k 0.94** (200 examples, nightly gate 0.89) —
see `kl48_gsm8k.log`. Raw: `kl48_base_bench.log`, `kl48_bench.log`.

## mini-K3 (K3 geometry, dummy weights) — bounded gate, all switches apply

12 layers (9 KDA + 3 MLA) with K3's real per-layer geometry: hidden 7168,
KDA 96 heads x 128 (H=24 per chip at TP=4), `gate_lower_bound=-5`,
MoE 24 routed + 2 shared experts, topk 16.

| Metric | upstream KDA | this kernel | speedup |
|---|---|---|---|
| Total token throughput | 1418 tok/s | **2638 tok/s** | **1.86x** |
| TTFT p50 | 2816 ms | **989 ms** | **2.85x** |
| TPOT p50 | 36.4 ms | **15.4 ms** | **2.36x** |
| Wall time, 160 requests | 62.5 s | 33.6 s | 1.86x |

Raw: `clean_base_bench.log`, `clean_ours_bench.log`.

### Server-side trace attribution (same workload, `--profile`)

| | upstream | ours |
|---|---|---|
| KDA target kernel, % of device time | **59.3%** | **4.6%** |
| KDA decode recurrence | 0.6% | 1.6% |
| MoE total | 16.9% | 44.9% |
| Achieved bf16 rate | 20.9 TFLOP/s/chip (MXU 2.28%) | **52.2 TFLOP/s/chip (MXU 5.69%)** |
| HBM bandwidth utilization | 27% | **68%** |
| Device time in 30 s window | 101.9 s | 42.9 s |

Raw: `official_base_extract.json`, `official_ours_extract.json`.

The 48B-vs-mini-K3 gap (1.23x vs 1.86x throughput, 1.49x vs 2.85x TTFT) is
exactly what `safe_gate` (FlashKDA-style MXU mapping) and `head_block`
(native layout + cross-head interleaving) contribute on a bounded-gate model.
