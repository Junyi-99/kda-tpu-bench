"""Capture an instrumented XProf trace of the KDA prefill kernel.

XProf can open up a pallas custom call when libtpu is told to trace it:

    LIBTPU_INIT_ARGS="--xla_xprof_enable_custom_call_tracing=true \
                      --xla_xprof_register_llo_debug_info=true \
                      --xla_enable_mxu_trace=true"

Those flag names need libtpu >= 0.0.46; older runtimes only know the legacy
names (xla_enable_custom_call_region_trace / xla_enable_mxu_trace /
xla_enable_local_dma_trace). Without any of them the trace stops at the
custom-call boundary — one opaque box, no units inside.

Usage: capture_llo.py <logdir> <impl> <iters> <S> <B>
  impl: original | mxu_port | optimized
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_SGL = next(p for p in (os.path.expanduser("~/sglang-jax/python"),
                        "/root/sglang-jax/python",
                        os.path.expanduser("~/claude/kimi-k3-kda/sglang-jax/python"))
            if os.path.isdir(p))
sys.path.insert(0, _SGL)
import sgl_jax.srt.kernels.kda.kda as kda  # noqa: E402

H, K, V, BT, LB = 24, 128, 128, 64, -5.0  # Kimi-K3 per-chip geometry
SCALE = K**-0.5
IMPLS = {
    "original":  dict(sg=False, fuse=False, uni=False, flat=False, hb=False),
    "mxu_port":  dict(sg=True,  fuse=False, uni=False, flat=False, hb=False),
    "optimized": dict(sg=True,  fuse=True,  uni=True,  flat=True,  hb=True),
}


def main():
    logdir, impl, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
    S, B = int(sys.argv[4]), int(sys.argv[5])
    sw = IMPLS[impl]
    T = S * B

    rng = np.random.RandomState(0)
    f32 = lambda *s: jnp.asarray(rng.randn(*s), dtype=jnp.float32)
    l2 = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    A_log = jnp.asarray(rng.randn(H) * 0.3, dtype=jnp.float32)
    dt_bias = jnp.asarray(rng.randn(H, K) * 0.3, dtype=jnp.float32)
    q = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
    k = l2(f32(1, T, H, K)).astype(jnp.bfloat16)
    v = f32(1, T, H, V).astype(jnp.bfloat16)
    g = f32(1, T, H, K)
    beta = jax.nn.sigmoid(f32(1, T, H))
    init = f32(B, H, K, V) * 0.1
    cu = jnp.asarray(np.arange(B + 1) * S, dtype=jnp.int32)

    supported = set(__import__("inspect").signature(kda.chunk_kda_fwd).parameters)

    @jax.jit
    def fn(q, k, v, g, beta, init, cu):
        kw = dict(scale=SCALE, initial_state=init, output_final_state=True,
                  cu_seqlens=cu, chunk_size=BT, safe_gate=sw["sg"], lower_bound=LB,
                  use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
                  fuse=sw["fuse"], unified_layout=sw["uni"],
                  flat_grid=sw["flat"], head_block=sw["hb"])
        return kda.chunk_kda_fwd(q, k, v, g, beta,
                                 **{a: b for a, b in kw.items() if a in supported})[0]

    args = (q, k, v, g, beta, init, cu)
    for _ in range(5):
        jax.block_until_ready(fn(*args))
    os.makedirs(logdir, exist_ok=True)
    with jax.profiler.trace(logdir):
        for _ in range(iters):
            jax.block_until_ready(fn(*args))
    print(f"TRACE_OK {impl} S={S} B={B} iters={iters} -> {logdir}", flush=True)


if __name__ == "__main__":
    main()
