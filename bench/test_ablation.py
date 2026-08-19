"""All valid ablation-switch combos of the tree kda.py vs the naive recurrence.

Combos: (fuse, unified_layout, flat_grid) x safe_gate, constraint unified=>fuse.
Local: PALLAS_INTERPRET=1 python test_ablation.py
TPU:   PALLAS_INTERPRET=0 python test_ablation.py
"""

import importlib.util
import os

os.environ.setdefault("PALLAS_INTERPRET", "1")

import numpy as np
import jax
import jax.numpy as jnp

from naive import naive_recurrent_kda

TREE = os.path.expanduser(
    "~/claude/kimi-k3-kda/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py"
)
if not os.path.exists(TREE):
    TREE = os.path.expanduser("~/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py")
_spec = importlib.util.spec_from_file_location("kda_tree", TREE)
kda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kda)

LENS = [30, 130, 64, 210]
H, K, V = 8, 32, 32  # H%8==0 使 head_block 路径可用
BT = 64
SCALE = K**-0.5
LB = -5.0
TOL = 2e-3

rng = np.random.RandomState(0)
T = sum(LENS)
f32 = lambda *s: jnp.asarray(rng.randn(*s), dtype=jnp.float32)
l2 = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
q, k = l2(f32(1, T, H, K)), l2(f32(1, T, H, K))
v = f32(1, T, H, V)
g_raw = f32(1, T, H, K)
beta = jax.nn.sigmoid(f32(1, T, H))
init = f32(len(LENS), H, K, V) * 0.1
cu = jnp.asarray(np.concatenate([[0], np.cumsum(LENS)]), dtype=jnp.int32)
A_log = jnp.asarray(rng.randn(H) * 0.3, dtype=jnp.float32)
dt_bias = jnp.asarray(rng.randn(H, K) * 0.3, dtype=jnp.float32)

g_act = LB * jax.nn.sigmoid(
    jnp.exp(A_log)[None, None, :, None] * (g_raw + dt_bias.reshape(H, K))
)
o_ref, s_ref = [], []
for i in range(len(LENS)):
    s, e = int(cu[i]), int(cu[i + 1])
    oi, si = naive_recurrent_kda(
        q[:, s:e], k[:, s:e], v[:, s:e], g_act[:, s:e], beta[:, s:e],
        scale=SCALE, initial_state=init[i : i + 1], output_final_state=True,
    )
    o_ref.append(oi)
    s_ref.append(si)
o_ref = jnp.concatenate(o_ref, axis=1)
s_ref = jnp.concatenate(s_ref, axis=0)

COMBOS = [  # (fuse, unified_layout, flat_grid, head_block)
    (True, True, True, True),    # hb 原生布局路径（unified/flat 被 hb 取代）
    (True, True, True, False),
    (True, True, False, False),
    (True, False, True, False),
    (True, False, False, False),
    (False, False, True, False),
    (False, False, False, False),
]

ok = True
print(f"LENS={LENS} T={T} H={H} K={K} BT={BT}")
print(f"{'fuse':>5} {'unified':>8} {'flat':>5} {'hb':>5} {'safe_gate':>9}   max|Δo|      max|ΔS|")
for fuse, uni, flat, hb in COMBOS:
    for sg in (True, False):
        out = kda.chunk_kda_fwd(
            q, k, v, g_raw, beta, scale=SCALE, initial_state=init,
            output_final_state=True, cu_seqlens=cu, chunk_size=BT,
            safe_gate=sg, lower_bound=LB, use_gate_in_kernel=True,
            A_log=A_log, dt_bias=dt_bias,
            fuse=fuse, unified_layout=uni, flat_grid=flat, head_block=hb,
        )
        do = float(jnp.max(jnp.abs(out[0] - o_ref)))
        ds = float(jnp.max(jnp.abs(out[1] - s_ref)))
        good = do < TOL and ds < TOL
        ok &= good
        print(f"{str(fuse):>5} {str(uni):>8} {str(flat):>5} {str(hb):>5} {str(sg):>9}   "
              f"{do:.2e} {'OK ' if good else 'FAIL'} {ds:.2e}")

print("ALL PASS" if ok else "SOME FAILED")
raise SystemExit(0 if ok else 1)
