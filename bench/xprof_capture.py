"""xprof trace 采集：3 实现 × 2 代表形状，各自独立 logdir（供 TensorBoard/XProf 截图）。

VM: PALLAS_INTERPRET=0 ~/venv-sgl/bin/python xprof_capture.py
输出: ~/kda_xprof/{impl}_{S}x{B}/ （每个含 plugins/profile/<run>/*.xplane.pb）
"""

import os

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
SHAPES = [(1024, 64), (128, 8)]
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

for S, B in SHAPES:
    T = S * B
    q = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
    k = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
    v = f32(1, T, H, V).astype(jnp.bfloat16)
    g = f32(1, T, H, K)
    beta = jax.nn.sigmoid(f32(1, T, H))
    init = f32(B, H, K, V) * 0.1
    cu = jnp.asarray(np.arange(B + 1) * S, dtype=jnp.int32)
    for name, sw in IMPLS.items():
        marker = f"gs://medusa-experiments/env-cache/kimi48b-results/k3camp/xprof/xp_{name}_S{S}xB{B}.tar.gz"
        if os.system(f"gsutil -q stat {marker} 2>/dev/null") == 0:
            print(f"skip {name} S={S} B={B} (already in GCS)", flush=True)
            continue
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
        for _ in range(10):
            jax.block_until_ready(fn(*fargs))
        logdir = os.path.expanduser(f"~/kda_xprof/{name}_S{S}xB{B}")
        os.makedirs(logdir, exist_ok=True)
        with jax.profiler.trace(logdir):
            for _ in range(30):
                jax.block_until_ready(fn(*fargs))
        print(f"traced {name} S={S} B={B} -> {logdir}", flush=True)
        os.system(f"cd ~ && tar czf /tmp/xp_{name}_S{S}xB{B}.tar.gz kda_xprof/{name}_S{S}xB{B} && "
                  f"gsutil -q cp /tmp/xp_{name}_S{S}xB{B}.tar.gz "
                  f"gs://medusa-experiments/env-cache/kimi48b-results/k3camp/xprof/ && echo uploaded_{name}_S{S}xB{B}")
print("XPROF_CAPTURE_DONE")
