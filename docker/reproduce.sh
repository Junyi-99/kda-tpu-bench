#!/bin/bash
# KDA kernel 复现入口。用法: reproduce.sh <mode>
set -e
PY=python
MODE=${1:-help}
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
  all           以上全部
  shell         进容器调试

TPU VM 运行需要: --privileged --net=host
结果 JSON 落 /root/，挂载 -v $(pwd)/out:/out 自动收集。
USAGE
    ;;
esac
