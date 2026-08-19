"""A 层：KDA kernel 微基准（K3 每芯几何：H=24, D=128, chunk 64, bf16）。

prefill: 目标 Pallas kernel（chunk_kda_fwd, safe_gate+hb 全开），BS=1, T 扫描。
decode:  生产路径 naive_recurrent_kda（T=1 递归步），BS 扫描。
         递归更新按构造与 context 长度无关（state 固定 [B,H,K,V]），此处不设 ctx 轴；
         ctx 相关性在 C 层用 prompt 长度实证（KDA 平、MLA 增）。

VM: PALLAS_INTERPRET=0 ~/venv-sgl/bin/python bench_kda_micro.py
"""

import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import sys

_SGL = next(p for p in (os.path.expanduser("~/sglang-jax/python"),
                        os.path.expanduser("~/claude/kimi-k3-kda/sglang-jax/python"))
            if os.path.isdir(p))
sys.path.insert(0, _SGL)
import sgl_jax.srt.kernels.kda.kda as kda

import sgl_jax.srt.kernels.kda.naive as naive

H, K, V, BT, LB = 24, 128, 128, 64, -5.0
SCALE = K**-0.5
WARMUP, RUNS = 20, 100
rng = np.random.RandomState(0)


def stats(fn, args):
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return {"median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a, 10)),
            "p90_ms": float(np.percentile(a, 90))}


results = {"geometry": {"H_per_chip": H, "K": K, "V": V, "chunk": BT, "dtype": "bf16",
                        "lower_bound": LB, "warmup": WARMUP, "runs": RUNS},
           "prefill": {}, "decode": {}}

f32 = lambda *s: jnp.asarray(rng.randn(*s), dtype=jnp.float32)
l2 = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)

# ---- prefill：目标 Pallas kernel ----
A_log = jnp.asarray(rng.randn(H) * 0.3, dtype=jnp.float32)
dt_bias = jnp.asarray(rng.randn(H, K) * 0.3, dtype=jnp.float32)
for T in (128, 256, 512, 1024, 2048, 4096, 8192):
    q = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
    k = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
    v = f32(1, T, H, V).astype(jnp.bfloat16)
    g = f32(1, T, H, K)
    beta = jax.nn.sigmoid(f32(1, T, H))
    init = f32(1, H, K, V) * 0.1
    cu = jnp.asarray([0, T], dtype=jnp.int32)

    @jax.jit
    def fn(q, k, v, g, beta, init, cu):
        return kda.chunk_kda_fwd(
            q, k, v, g, beta, scale=SCALE, initial_state=init, output_final_state=True,
            cu_seqlens=cu, chunk_size=BT, safe_gate=True, lower_bound=LB,
            use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
        )[0]

    r = stats(fn, (q, k, v, g, beta, init, cu))
    results["prefill"][T] = r
    print(f"prefill T={T:>5}: {r['median_ms']:.3f} ms (p10 {r['p10_ms']:.3f} / p90 {r['p90_ms']:.3f})", flush=True)

# ---- decode：生产 naive 递归（T=1）----
for B in (1, 4, 8, 16, 32):
    q = l2(f32(B, 1, H, K)).astype(jnp.bfloat16)
    k = l2(f32(B, 1, H, K)).astype(jnp.bfloat16)
    v = f32(B, 1, H, V).astype(jnp.bfloat16)
    g = -jax.nn.softplus(f32(B, 1, H, K))  # decode 门在 kernel 外算好
    beta = jax.nn.sigmoid(f32(B, 1, H))
    h0 = f32(B, H, K, V) * 0.1

    @jax.jit
    def fn(q, k, v, g, beta, h0):
        o, s = naive.naive_recurrent_kda(q, k, v, g, beta, scale=SCALE,
                                         initial_state=h0, output_final_state=True)
        return o

    r = stats(fn, (q, k, v, g, beta, h0))
    results["decode"][B] = r
    print(f"decode B={B:>3}: {r['median_ms']:.4f} ms (p10 {r['p10_ms']:.4f} / p90 {r['p90_ms']:.4f})", flush=True)

with open(os.path.expanduser("~/kda_micro.json"), "w") as f:
    json.dump(results, f, indent=1)
print("MICRO_DONE")
