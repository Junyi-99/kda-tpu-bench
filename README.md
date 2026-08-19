# kda-tpu-bench

Reproducible benchmarks for the Kimi K3 KDA prefill kernel on TPU v6e
(SGLang-JAX). Kernel code is pinned via the `sglang-jax` submodule
(branch `kda-fused-unified-neumann`); the Docker image is built
hermetically from this repo's contents — what you review is what runs.

## Quick start

```bash
# run on a TPU v6e VM — FINAL kernel (this repo's pinned submodule, default)
docker run --rm --privileged --net=host -v $(pwd)/out:/out \
  ghcr.io/junyi-99/kda-tpu-bench:latest grid       # 56-point S×B grid × 3 impls

# BASELINE kernel (upstream sglang-jax main, pinned e6e47830) — same modes
docker run --rm --privileged --net=host -e KDA_KERNEL=baseline -v $(pwd)/out:/out \
  ghcr.io/junyi-99/kda-tpu-bench:latest grid
# other modes: correctness / micro / extras / sharegpt / lengths / all / shell
```

`KDA_KERNEL=baseline|final` selects the kernel under test. Note: upstream main
has already merged the safe_gate MXU port, so baseline-vs-final isolates the
**structural** optimizations (fusion / unified addressing / flat grid /
head-block + interleaving) of PR #4.

Results are written to `/root/*.json` inside the container and copied to
`/out` when mounted.

## What each mode measures

| Mode | Measurement |
|---|---|
| `correctness` | ablation-switch combination matrix vs reference (CPU-safe) |
| `micro` | per-layer kernel latency: prefill T-sweep + decode B-sweep |
| `grid` | 56 points (8 S × 7 B) × {original, mxu_port, optimized} |
| `extras` | zero/non-zero initial_state equivalence + 6-rung ablation ladders |
| `lengths` | ShareGPT-derived length distribution (pinned revision, sha-verified) |
| `sharegpt` | 7 B configs × 100 seed-0 replay batches × 3 impls |

Protocol details and published results: see `docs/kda-benchmarks/` in the
submodule and [PR #4](https://github.com/Junyi-99/sglang-jax/pull/4).

## Layout

```
bench/        benchmark & correctness scripts (review target)
docker/       Dockerfile + reproduce.sh entrypoint
sglang-jax/   pinned kernel submodule
.github/      CI: build & push image to ghcr on every push to main
```

## Environment pins

Python 3.12 · `jax[tpu]==0.10.2` · BF16 (fp32 accumulation in strip-GEMMs) ·
Kimi-K3 per-chip geometry `H=24, K=V=128, chunk=64, lower_bound=-5`.

> Validated on TPU v6e: in-container benchmark results match native (non-Docker)
> runs within measurement noise — the container adds no measurable overhead.
