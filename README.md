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


## Official sglang-jax benchmarks (full model)

The image also wraps sglang-jax's own tools, so you can drive the official
`bench_serving` / `run_eval` against either model with one command. `MODEL=`
takes a preset (`kimi-linear-48b`, `mini-k3`, `mini-k3-half`) or any path;
presets prepare the model dir themselves (GCS weights for 48B, geometry-faithful
dummy config for mini-K3).

```bash
# official bench_serving, sharegpt defaults (160 prompts / concurrency 32)
docker run --rm --privileged --net=host -e MODEL=kimi-linear-48b \
  -v $(pwd)/out:/out ghcr.io/junyi-99/kda-tpu-bench:latest bench-serving

# same, upstream KDA path — the A/B baseline
docker run --rm --privileged --net=host -e MODEL=kimi-linear-48b \
  -e KDA_FORCE_BASELINE=1 -v $(pwd)/out:/out \
  ghcr.io/junyi-99/kda-tpu-bench:latest bench-serving

# any bench_serving flags pass straight through
docker run --rm --privileged --net=host -e MODEL=mini-k3 \
  -v $(pwd)/out:/out ghcr.io/junyi-99/kda-tpu-bench:latest \
  bench-serving --dataset-name random --random-input-len 4096 \
    --random-output-len 128 --num-prompts 64 --max-concurrency 16

# correctness (official run_eval.py)
docker run --rm --privileged --net=host -e MODEL=kimi-linear-48b \
  -v $(pwd)/out:/out ghcr.io/junyi-99/kda-tpu-bench:latest run-eval

# just the server, drive it yourself from the host
docker run --rm --privileged --net=host -e MODEL=mini-k3 \
  ghcr.io/junyi-99/kda-tpu-bench:latest serve
```

Extra `launch_server` flags go in `SERVER_ARGS`, e.g.
`-e SERVER_ARGS="--max-prefill-tokens 2048 --chunked-prefill-size 2048 --max-running-requests 32"`.

## What each mode measures

| Mode | Measurement |
|---|---|
| `correctness` | ablation-switch combination matrix vs reference (CPU-safe) |
| `micro` | per-layer kernel latency: prefill T-sweep + decode B-sweep |
| `grid` | 56 points (8 S × 7 B) × {original, mxu_port, optimized} |
| `extras` | zero/non-zero initial_state equivalence + 6-rung ablation ladders |
| `lengths` | ShareGPT-derived length distribution (pinned revision, sha-verified) |
| `sharegpt` | 7 B configs × 100 seed-0 replay batches × 3 impls |

| `e2e` | end-to-end serving A/B on a full model (`E2E_MODEL=<dir>`): ours vs `KDA_FORCE_BASELINE=1` |
| `bench-serving` | official `sgl_jax.bench_serving` against a `MODEL=` preset or path |
| `run-eval` | official `run_eval.py` correctness (gsm8k by default) |
| `serve` | launch the official server only, and leave it up |

Published results: kernel-level in `results/`, full-model serving (Kimi-Linear-48B
and mini-K3, with trace attribution) in [`results/e2e/`](results/e2e/README.md),
and the write-up in [PR #4](https://github.com/Junyi-99/sglang-jax/pull/4).

## Layout

```
bench/        benchmark & correctness scripts (review target)
docker/       Dockerfile + reproduce.sh entrypoint
sglang-jax/   pinned kernel submodule
.github/      CI: build & push image to ghcr on every push to main
```

## Environment pins

Python 3.12 · `jax[tpu]==0.11.1` · `libtpu==0.0.46` · `xprof==2.23.1` · BF16
(fp32 accumulation in strip-GEMMs) · Kimi-K3 per-chip geometry
`H=24, K=V=128, chunk=64, lower_bound=-5`.

Previously pinned to `jax[tpu]==0.10.2` / `libtpu==0.0.42.1`; kernel latency is
unchanged across the bump (per-shape medians within run-to-run noise, e.g.
prefill T=8192: 3.899 → 3.874 ms; decode B=32: 0.2464 → 0.2489 ms).

> Validated on TPU v6e: in-container benchmark results match native (non-Docker)
> runs within measurement noise — the container adds no measurable overhead.
