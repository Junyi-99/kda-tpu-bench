"""#3 零/非零 initial_state 等价性验证 + #4 双代表点完整 6 档阶梯。

VM: JAX_COMPILATION_CACHE_DIR=~/jax_kernel_cache PALLAS_INTERPRET=0 \
    ~/venv-sgl/bin/python bench_kda_extras.py
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
rng = np.random.RandomState(0)
f32 = lambda *s: jnp.asarray(rng.randn(*s), dtype=jnp.float32)
l2 = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
A_log = jnp.asarray(rng.randn(H) * 0.3, dtype=jnp.float32)
dt_bias = jnp.asarray(rng.randn(H, K) * 0.3, dtype=jnp.float32)


def bench(S, B, sw, init, warmup=20, runs=100):
    T = S * B
    q = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
    k = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
    v = f32(1, T, H, V).astype(jnp.bfloat16)
    g = f32(1, T, H, K)
    beta = jax.nn.sigmoid(f32(1, T, H))
    cu = jnp.asarray(np.arange(B + 1) * S, dtype=jnp.int32)

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
    for _ in range(warmup):
        jax.block_until_ready(fn(*fargs))
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*fargs))
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return {"median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a, 10)),
            "p90_ms": float(np.percentile(a, 90))}


OPT = dict(sg=True, fuse=True, uni=True, flat=True, hb=True)
ORIG = dict(sg=False, fuse=False, uni=False, flat=False, hb=False)
out = {"state_equivalence": {}, "ladder": {}}

# ---- #3 等价性：零 state（P=0 冷启动）vs 随机 state（任意 cached prefix）----
for tag, sw in (("optimized", OPT), ("original", ORIG)):
    S, B = 512, 8
    z = bench(S, B, sw, jnp.zeros((B, H, K, V), jnp.float32))
    r = bench(S, B, sw, f32(B, H, K, V) * 0.1)
    out["state_equivalence"][tag] = {"S": S, "B": B, "zero_state": z, "random_state": r,
                                     "delta_pct": 100 * (r["median_ms"] - z["median_ms"]) / z["median_ms"]}
    print(f"equiv {tag}: zero {z['median_ms']:.3f} vs random {r['median_ms']:.3f} ms "
          f"({out['state_equivalence'][tag]['delta_pct']:+.2f}%)", flush=True)

# ---- #4 双代表点完整 6 档阶梯 ----
LADDER = [
    ("baseline",  dict(sg=False, fuse=False, uni=False, flat=False, hb=False)),
    ("+safe_gate", dict(sg=True, fuse=False, uni=False, flat=False, hb=False)),
    ("+fuse",     dict(sg=True, fuse=True,  uni=False, flat=False, hb=False)),
    ("+unified",  dict(sg=True, fuse=True,  uni=True,  flat=False, hb=False)),
    ("+flat",     dict(sg=True, fuse=True,  uni=True,  flat=True,  hb=False)),
    ("+hb",       dict(sg=True, fuse=True,  uni=True,  flat=True,  hb=True)),
]
for S, B in ((32, 1), (1024, 64)):
    init = f32(B, H, K, V) * 0.1
    rows = {}
    prev = None
    for name, sw in LADDER:
        r = bench(S, B, sw, init)
        rows[name] = r
        inc = (prev / r["median_ms"]) if prev else 1.0
        print(f"ladder S={S} B={B} {name:11s}: {r['median_ms']:8.3f} ms  (+{inc:.2f}x)", flush=True)
        prev = r["median_ms"]
    out["ladder"][f"S{S}_B{B}"] = rows

json.dump(out, open(os.path.expanduser("~/kda_extras.json"), "w"), indent=1)
print("EXTRAS_DONE")
