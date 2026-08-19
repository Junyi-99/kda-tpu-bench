# Published results

Raw data and figures backing [sglang-jax PR #4](https://github.com/Junyi-99/sglang-jax/pull/4).
Measured on a single TPU v6e chip, `jax==0.10.2`, BF16, Kimi-K3 per-chip
geometry `H=24, K=V=128, BT=64, lower_bound=-5`; median of 100 runs after
20 warmup per point. Workloads are `(S, B)`: S = tokens per request,
B = sequences packed into one kernel call via `cu_seqlens` (`T = B·S`,
no separate batch dimension).

| File | Contents |
|---|---|
| `kda_grid.json` | 56-point fixed-length grid (8 S × 7 B) × 3 implementations — 56/56 improved, 5.14×–32.69×, GM 13.55× |
| `kda_sharegpt.json` | ShareGPT-derived replay, 7 B configs × 100 seed-0 batches × 3 impls — GM 14.34× |
| `kda_extras.json` | zero/random `initial_state` equivalence (−0.02%) + 6-rung ablation ladders at (S=32,B=1) and (S=1024,B=64) |
| `sharegpt_k3_lengths.json` | 52,229 tokenizer-exact prompt lengths (pinned dataset revision + vLLM filter) |
| `kda-fig*.png` | main latency figure, speedup heatmap, ablation waterfall, achieved-TFLOP/s |

These files are also shipped inside the Docker image under `/root/results/`.
Regenerate any of them with the corresponding image mode (`grid` / `extras` /
`sharegpt`).
