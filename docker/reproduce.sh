#!/bin/bash
# KDA kernel 复现入口。用法: reproduce.sh <mode>
set -e
PY=python
MODE=${1:-help}
# 内核选择：KDA_KERNEL=baseline（上游 main 原始 kernel）| final（默认，本仓库最新）
KDIR=/root/sglang-jax/python/sgl_jax/srt/kernels/kda
case "${KDA_KERNEL:-final}" in
  baseline)
    rm -rf $KDIR/v2
    cp /root/kernels/baseline/kda.py $KDIR/kda.py
    echo ">>> kernel = BASELINE (upstream main, pinned e6e47830)"
    ;;
  final|*)
    rm -rf $KDIR
    cp -r /root/kernels/final $KDIR
    echo ">>> kernel = FINAL (this repo's pinned submodule)"
    ;;
esac

SG_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"

collect() { [ -d /out ] && cp -f /root/kda_*.json /root/sharegpt_k3_lengths.json /out/ 2>/dev/null || true; }

case "$MODE" in
  correctness)
    # 无 TPU 也可跑：PALLAS_INTERPRET=1（默认）；TPU 上传 PALLAS_INTERPRET=0
    PALLAS_INTERPRET=${PALLAS_INTERPRET:-1} $PY /root/bench/test_ablation.py
    ;;
  micro)
    # A 层微基准：prefill T 扫描 + decode BS 扫描（K3 每芯几何 H=24）
    PALLAS_INTERPRET=0 $PY /root/bench/bench_kda_micro.py; collect
    ;;
  grid)
    # 56 点 fixed-length 网格 × 3 实现（8 S × 7 B）
    PALLAS_INTERPRET=0 $PY /root/bench/bench_kda_grid.py; collect
    ;;
  extras)
    # 零/非零 initial_state 等价性 + 双代表点 6 档消融阶梯
    PALLAS_INTERPRET=0 $PY /root/bench/bench_kda_extras.py; collect
    ;;
  lengths)
    # ShareGPT-derived 长度分布（钉死 dataset revision + K3 tokenizer，校验 sha256 与统计量）
    [ -f /root/sharegpt.json ] || curl -L -o /root/sharegpt.json "$SG_URL"
    $PY /root/bench/k3_sharegpt_lengths.py /root/sharegpt.json "moonshotai/Kimi-K3" /root/sharegpt_k3_lengths.json
    collect
    ;;
  sharegpt)
    # ShareGPT-derived replay bench（7 B × 100 batches × 3 实现，seed=0）
    [ -f /root/sharegpt_k3_lengths.json ] || /root/reproduce.sh lengths
    PALLAS_INTERPRET=0 $PY /root/bench/bench_kda_sharegpt.py /root/sharegpt_k3_lengths.json /root/kda_sharegpt.json
    collect
    ;;
  all)
    /root/reproduce.sh correctness
    /root/reproduce.sh micro
    /root/reproduce.sh grid
    /root/reproduce.sh extras
    /root/reproduce.sh sharegpt
    ;;
  selftest)
    # CPU 冒烟：包导入 + 消融开关签名存在（无 TPU 依赖）
    python - <<'PY'
import sys, inspect
sys.path.insert(0, "/root/sglang-jax/python")
import sgl_jax.srt.kernels.kda.kda as kda
sig = inspect.signature(kda.chunk_kda_fwd)
import os
need = ("safe_gate", "fuse", "unified_layout", "flat_grid", "head_block") if os.environ.get("KDA_KERNEL", "final") != "baseline" else ("safe_gate",)
for k in need:
    assert k in sig.parameters, f"missing switch: {k}"
print(f"SELFTEST_OK: kernel={os.environ.get('KDA_KERNEL','final')}, switches verified: {need}")
PY
    ;;
  shell) exec /bin/bash ;;
  *)
    cat <<'USAGE'
Kimi K3 KDA TPU kernel benchmarks (SGLang-JAX, pinned jax==0.10.2, commit e1cd9ed7)

modes:
  correctness   开关组合正确性矩阵（无 TPU 可跑，PALLAS_INTERPRET=1）
  micro         A 层 kernel 微基准（需 TPU）
  grid          56 点 fixed-length S×B 网格 × 3 实现（需 TPU）
  extras        state 等价性验证 + 双点消融阶梯（需 TPU）
  lengths       ShareGPT-derived 长度分布生成（CPU；下载 672MB 数据集）
  sharegpt      ShareGPT replay bench（需 TPU；自动先跑 lengths）
  selftest      CPU 冒烟：kda 包导入 + 开关签名（CI 用）

环境变量 KDA_KERNEL=baseline|final（默认 final）切换被测内核：
  baseline = 上游 sglang-jax main 的原始 kernel（含已合并的 safe_gate MXU port）
  final    = 本仓库 submodule 钉死的最新 commit（PR#4 全部结构优化）
  all           以上全部
  shell         进容器调试

TPU VM 运行需要: --privileged --net=host
结果 JSON 落 /root/，挂载 -v $(pwd)/out:/out 自动收集。
USAGE
    ;;
esac
