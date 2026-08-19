"""fixed-length 56 点网格 × 3 实现（报告 §8.1.1/§9 口径）。

S ∈ {8,16,32,64,128,256,512,1024} × B ∈ {1,2,4,8,16,32,64}，
K3 每芯几何 H=24, K=V=128, BT=64, bf16, lower_bound=-5, use_gate_in_kernel。
B>1 = cu_seqlens 打包 B 条等长序列（serving 真实形态）。
实现：original(全关) / mxu_port(仅 safe_gate) / optimized(全开)。
warmup 20 / run 100，median + p10/p90 + actual-token throughput。

VM: PALLAS_INTERPRET=0 ~/venv-sgl/bin/python bench_kda_grid.py
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

H, K, V, BT, LB = 24, 128, 128, 64, -5.0
SCALE = K**-0.5
WARMUP, RUNS = 20, 100
S_SET = (8, 16, 32, 64, 128, 256, 512, 1024)
B_SET = (1, 2, 4, 8, 16, 32, 64)
IMPLS = {
    "original":  dict(sg=False, fuse=False, uni=False, flat=False, hb=False),
    "mxu_port":  dict(sg=True,  fuse=False, uni=False, flat=False, hb=False),
    "optimized": dict(sg=True,  fuse=True,  uni=True,  flat=True,  hb=True),
}
rng = np.random.RandomState(0)
f32 = lambda *s: jnp.asarray(rng.randn(*s), dtype=jnp.float32)
l2 = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
A_log = jnp.asarray(rng.randn(H) * 0.3, dtype=jnp.float32)
dt_bias = jnp.asarray(rng.randn(H, K) * 0.3, dtype=jnp.float32)

results = {"geometry": dict(H=H, K=K, V=V, BT=BT, lower_bound=LB, dtype="bf16",
                            warmup=WARMUP, runs=RUNS,
                            note="B>1 = cu_seqlens packed equal-length seqs"),
           "grid": {}}

for S in S_SET:
    for B in B_SET:
        T = S * B
        q = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
        k = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
        v = f32(1, T, H, V).astype(jnp.bfloat16)
        g = f32(1, T, H, K)
        beta = jax.nn.sigmoid(f32(1, T, H))
        init = f32(B, H, K, V) * 0.1
        cu = jnp.asarray(np.arange(B + 1) * S, dtype=jnp.int32)
        row = {}
        for name, sw in IMPLS.items():
            print(f"  [{name}] S={S} B={B} compiling...", flush=True)
            @jax.jit
            def fn(q, k, v, g, beta, init, cu):
                return kda.chunk_kda_fwd(
                    q, k, v, g, beta, scale=SCALE, initial_state=init,
                    output_final_state=True, cu_seqlens=cu, chunk_size=BT,
                    safe_gate=sw["sg"], lower_bound=LB,
                    use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
                    fuse=sw["fuse"], unified_layout=sw["uni"],
                    flat_grid=sw["flat"], head_block=sw["hb"],
                )[0]
            fargs = (q, k, v, g, beta, init, cu)
            try:
                for _ in range(WARMUP):
                    jax.block_until_ready(fn(*fargs))
                ts = []
                for _ in range(RUNS):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(*fargs))
                    ts.append((time.perf_counter() - t0) * 1e3)
                a = np.array(ts)
                row[name] = {"median_ms": float(np.median(a)),
                             "p10_ms": float(np.percentile(a, 10)),
                             "p90_ms": float(np.percentile(a, 90)),
                             "tok_per_s": float(T / (np.median(a) / 1e3))}
            except Exception as e:  # noqa: BLE001
                row[name] = {"error": str(e)[:150]}
        results["grid"][f"S{S}_B{B}"] = row
        o, m, p = (row.get(n, {}).get("median_ms") for n in ("original", "mxu_port", "optimized"))
        if o and p:
            print(f"S={S:>5} B={B:>2}: orig {o:8.3f}  mxu {m:8.3f}  opt {p:8.3f} ms  "
                  f"| mxu {o/m:4.2f}x  full {o/p:5.2f}x", flush=True)
        else:
            print(f"S={S:>5} B={B:>2}: FAILED {row}", flush=True)

with open(os.path.expanduser("~/kda_grid.json"), "w") as f:
    json.dump(results, f, indent=1)
print("GRID_DONE")
