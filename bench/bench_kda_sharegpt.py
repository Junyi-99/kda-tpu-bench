"""ShareGPT-derived replay bench（报告 §8.1.2 协议）。

- lengths.json 的经验分布；np.random.default_rng(0)，按 B∈{1..64} 顺序各抽 100 个 S
- 每 batch = B 条等长 S 的序列（cu_seqlens 打包），三实现 replay 完全相同的序列
- padded 口径：T_total 打到 bucket（尾部补一条 dummy 序列，模拟 serving 的 token bucket），
  同时报告 actual-token 与 padded-token throughput
- 每 batch：warmup 3 / run 15，取 median；每 (B, impl) 汇总 100 个 batch 的分布

VM: JAX_COMPILATION_CACHE_DIR=~/jax_kernel_cache PALLAS_INTERPRET=0 \
    ~/venv-sgl/bin/python bench_kda_sharegpt.py <lengths.json> <out.json>
"""

import json
import os
import sys
import time
import importlib.util

import numpy as np
import jax
import jax.numpy as jnp

TREE = os.path.expanduser("~/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py")
_spec = importlib.util.spec_from_file_location("kda_tree", TREE)
kda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kda)

H, K, V, BT, LB = 24, 128, 128, 64, -5.0
SCALE = K**-0.5
WARMUP, RUNS = 3, 15
B_SET = (1, 2, 4, 8, 16, 32, 64)
N_BATCH = 100
BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
IMPLS = {
    "original":  dict(sg=False, fuse=False, uni=False, flat=False, hb=False),
    "mxu_port":  dict(sg=True,  fuse=False, uni=False, flat=False, hb=False),
    "optimized": dict(sg=True,  fuse=True,  uni=True,  flat=True,  hb=True),
}

lengths = np.array(json.load(open(sys.argv[1]))["lengths"])
out_path = sys.argv[2]
rng_pick = np.random.default_rng(0)
sampled = {B: rng_pick.choice(lengths, size=N_BATCH, replace=True).tolist() for B in B_SET}

rng = np.random.RandomState(0)
f32 = lambda *s: jnp.asarray(rng.randn(*s), dtype=jnp.float32)
l2 = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
A_log = jnp.asarray(rng.randn(H) * 0.3, dtype=jnp.float32)
dt_bias = jnp.asarray(rng.randn(H, K) * 0.3, dtype=jnp.float32)

_fns = {}


def get_fn(name, sw, t_pad, n_cu):
    key = (name, t_pad, n_cu)
    if key not in _fns:
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
        _fns[key] = fn
    return _fns[key]


PARTIAL = os.path.expanduser("~/kda_sharegpt_partial.json")
_done = {}
if os.path.exists(PARTIAL):
    try:
        _done = json.load(open(PARTIAL)).get("configs", {})
        print(f"resume: 已完成 B={list(_done)}", flush=True)
    except Exception:
        _done = {}

results = {"protocol": {"seed": 0, "n_batch": N_BATCH, "warmup": WARMUP, "runs": RUNS,
                        "geometry": dict(H=H, K=K, V=V, BT=BT, lb=LB, dtype="bf16"),
                        "padding": "T padded to bucket; trailing pad = dummy seq (serving-style)",
                        "sampled_lengths": sampled},
           "configs": {}}

for B in B_SET:
    if str(B) in _done:
        results["configs"][B] = _done[str(B)]
        print(f"skip B={B} (resumed)", flush=True)
        continue
    per_impl = {n: {"batch_median_ms": [], "actual_tok": [], "padded_tok": []} for n in IMPLS}
    for bi, S in enumerate(sampled[B]):
        S = int(S)
        T_actual = S * B
        t_pad = next(x for x in BUCKETS if x >= T_actual)
        cu_list = [i * S for i in range(B + 1)]
        n_seq = B
        if t_pad > T_actual:
            cu_list.append(t_pad)
            n_seq += 1
        cu = jnp.asarray(cu_list, dtype=jnp.int32)
        q = l2(f32(1, t_pad, H, K)).astype(jnp.bfloat16)
        k = l2(f32(1, t_pad, H, K)).astype(jnp.bfloat16)
        v = f32(1, t_pad, H, V).astype(jnp.bfloat16)
        g = f32(1, t_pad, H, K)
        beta = jax.nn.sigmoid(f32(1, t_pad, H))
        init = f32(n_seq, H, K, V) * 0.1
        fargs = (q, k, v, g, beta, init, cu)
        for name, sw in IMPLS.items():
            fn = get_fn(name, sw, t_pad, len(cu_list))
            for _ in range(WARMUP):
                jax.block_until_ready(fn(*fargs))
            ts = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*fargs))
                ts.append((time.perf_counter() - t0) * 1e3)
            med = float(np.median(ts))
            per_impl[name]["batch_median_ms"].append(med)
            per_impl[name]["actual_tok"].append(T_actual)
            per_impl[name]["padded_tok"].append(t_pad)
        if (bi + 1) % 20 == 0:
            print(f"B={B}: {bi+1}/{N_BATCH} batches", flush=True)
    cfg = {}
    for name, d in per_impl.items():
        m = np.array(d["batch_median_ms"])
        at, pt = np.sum(d["actual_tok"]), np.sum(d["padded_tok"])
        cfg[name] = {
            "median_of_medians_ms": float(np.median(m)),
            "p10_ms": float(np.percentile(m, 10)), "p90_ms": float(np.percentile(m, 90)),
            "tok_weighted_actual_tok_per_s": float(at / (np.sum(m) / 1e3)),
            "tok_weighted_padded_tok_per_s": float(pt / (np.sum(m) / 1e3)),
        }
    o = np.array(per_impl["original"]["batch_median_ms"])
    p = np.array(per_impl["optimized"]["batch_median_ms"])
    mx = np.array(per_impl["mxu_port"]["batch_median_ms"])
    cfg["speedup_full_median"] = float(np.median(o / p))
    cfg["speedup_full_gm"] = float(np.exp(np.mean(np.log(o / p))))
    cfg["speedup_mxu_median"] = float(np.median(o / mx))
    results["configs"][B] = cfg
    json.dump(results, open(PARTIAL, "w"))
    os.system("gsutil -q cp %s gs://medusa-experiments/env-cache/kimi48b-results/k3camp/ 2>/dev/null" % PARTIAL)
    print(f"== B={B}: opt median {cfg['optimized']['median_of_medians_ms']:.3f} ms, "
          f"full speedup median {cfg['speedup_full_median']:.2f}x gm {cfg['speedup_full_gm']:.2f}x", flush=True)

json.dump(results, open(out_path, "w"), indent=1)
print("SHAREGPT_BENCH_DONE")
