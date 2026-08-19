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

# 解析后的配置全部打印，默认值也打印——日志里能看到实际跑的是什么，不用回头读源码
echo ">>> config: KDA_KERNEL=${KDA_KERNEL:-final} MODEL=${MODEL:-<mode default>} PORT=${PORT:-30040} TP=${TP:-4}" \
     "PALLAS_INTERPRET=${PALLAS_INTERPRET:-0} KDA_FORCE_BASELINE=${KDA_FORCE_BASELINE:-0}" \
     "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/root/jax_kernel_cache}"

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
  llo)
    # Pallas 内部归因：三实现各采一条插桩 trace，解析出指令构成 + 硬件计数器，出对比图。
    # 新 flag 名需要 libtpu >= 0.0.46；旧运行时只认 legacy 名（见 bench/llo/capture_llo.py）。
    # 插桩只用于归因——计时数字一律取未插桩的 micro/grid。
    LLO_FLAGS="--xla_xprof_enable_custom_call_tracing=true --xla_xprof_register_llo_debug_info=true --xla_enable_mxu_trace=true --xla_enable_local_dma_trace=true"
    MIX_S=${LLO_MIX_S:-1024}; MIX_B=${LLO_MIX_B:-8}     # 指令构成：大形状，三侧同窗口
    CTR_S=${LLO_CTR_S:-64};   CTR_B=${LLO_CTR_B:-1}     # 计数器：小形状，采样才够密
    MIX_JSON=""; CTR_JSON=""
    for impl in original mxu_port optimized; do
      for kind in mix ctr; do
        [ "$kind" = mix ] && { S=$MIX_S; B=$MIX_B; IT=3; } || { S=$CTR_S; B=$CTR_B; IT=1; }
        LIBTPU_INIT_ARGS="$LLO_FLAGS" PALLAS_INTERPRET=0 \
          $PY /root/bench/llo/capture_llo.py /root/llo_${kind}_${impl} $impl $IT $S $B
        $PY /root/bench/llo/analyze_llo.py /root/llo_${kind}_${impl} $impl /root/kda_llo_${kind}_${impl}.json
        rm -rf /root/llo_${kind}_${impl}          # trace 本体动辄 GB，解析完即弃
      done
      MIX_JSON="$MIX_JSON /root/kda_llo_mix_${impl}.json"
      CTR_JSON="$CTR_JSON /root/kda_llo_ctr_${impl}.json"
    done
    $PY /root/bench/llo/plot_units.py /root/kda_llo_units.png $MIX_JSON -- $CTR_JSON
    [ -d /out ] && cp -f /root/kda_llo_units.png /out/ 2>/dev/null
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
  e2e)
    # 端到端 serving A/B（需真/dummy 模型目录，通过 E2E_MODEL 指定）
    : "${E2E_MODEL:?set E2E_MODEL=/path/to/model-dir}"
    PY=$PY SGL=/root/sglang-jax bash /root/bench/e2e_serving.sh "$E2E_MODEL" "${E2E_TAG:-run}" $E2E_SERVER_ARGS
    collect
    ;;
  serve)
    # 只起官方 server（模型 preset 或路径），前台阻塞，供外部自行压测
    shift || true
    bash /root/bench/serve_model.sh "${MODEL:-mini-k3}" "$@"
    echo ">>> server stays up; press Ctrl-C to stop"
    tail -f /root/server.log
    ;;
  bench-serving)
    # 官方 sgl_jax.bench_serving 入口：起 server（MODEL=preset|path）后透传所有参数
    shift || true
    bash /root/bench/serve_model.sh "${MODEL:-mini-k3}" $SERVER_ARGS
    [ -f /root/sharegpt.json ] || curl -sL -o /root/sharegpt.json "$SG_URL"
    ARGS="$*"
    [ -n "$ARGS" ] || ARGS="--dataset-name sharegpt --dataset-path /root/sharegpt.json --num-prompts 160 --max-concurrency 32 --request-rate inf --warmup-requests 8 --output-details"
    case "$ARGS" in *--dataset-path*) ;; *--dataset-name\ sharegpt*) ARGS="$ARGS --dataset-path /root/sharegpt.json";; esac
    ( cd /root/sglang-jax && $PY -m sgl_jax.bench_serving --backend sgl-jax \
        --host 127.0.0.1 --port ${PORT:-30040} $ARGS \
        --output-file /root/bench_serving_${MODEL:-mini-k3}${KDA_FORCE_BASELINE:+_baseline}.jsonl \
        2>&1 | tail -50 | tee /root/bench_serving_${MODEL:-mini-k3}${KDA_FORCE_BASELINE:+_baseline}.log )
    pkill -f "sgl_jax.launch_server" 2>/dev/null || true
    collect
    ;;
  run-eval)
    # 官方 run_eval.py（正确性，默认 gsm8k）
    shift || true
    bash /root/bench/serve_model.sh "${MODEL:-kimi-linear-48b}" $SERVER_ARGS
    ARGS="$*"
    [ -n "$ARGS" ] || ARGS="--eval-name gsm8k --num-examples 200"
    ( cd /root/sglang-jax/test/srt && $PY run_eval.py --host 127.0.0.1 --port ${PORT:-30040} $ARGS \
        2>&1 | tail -10 | tee /root/run_eval_${MODEL:-kimi-linear-48b}.log )
    pkill -f "sgl_jax.launch_server" 2>/dev/null || true
    collect
    ;;
  shell) exec /bin/bash ;;
  *)
    cat <<'USAGE'
Kimi K3 KDA TPU kernel benchmarks
SGLang-JAX submodule 043ff29a | jax[tpu]==0.11.1 | libtpu==0.0.46 | xprof==2.23.1

modes:
  correctness   开关组合正确性矩阵（无 TPU 可跑，PALLAS_INTERPRET=1）
  micro         A 层 kernel 微基准（需 TPU）
  grid          56 点 fixed-length S×B 网格 × 3 实现（需 TPU）
  extras        state 等价性验证 + 双点消融阶梯（需 TPU）
  lengths       ShareGPT-derived 长度分布生成（CPU；下载 672MB 数据集）
  sharegpt      ShareGPT replay bench（需 TPU；自动先跑 lengths）
  llo           Pallas 内部归因（需 TPU）：指令构成 + MXU/VPU 硬件计数器 + 对比图
                形状可调：LLO_MIX_S/LLO_MIX_B（构成，默认 1024×8）、LLO_CTR_S/LLO_CTR_B（计数器，默认 64×1）
  selftest      CPU 冒烟：kda 包导入 + 开关签名（CI 用）
  e2e           端到端 serving A/B（E2E_MODEL=模型目录；ours vs KDA_FORCE_BASELINE=1）
  serve         只起官方 server（MODEL=kimi-linear-48b|mini-k3|mini-k3-half|<path>）
  bench-serving 官方 sgl_jax.bench_serving（起 server 后透传参数；MODEL= 同上）
  run-eval      官方 run_eval.py 正确性评测（默认 gsm8k 200 题）
  all           correctness + micro + grid + extras + sharegpt
  shell         进容器调试

模型 preset（MODEL=）：
  kimi-linear-48b  真权重，从 GCS model-cache 拉取（~92GB -> /dev/shm）
  mini-k3          K3 几何 12 层（9 KDA + 3 MLA），dummy 权重
  mini-k3-half     K3 几何 24 层（18 KDA + 6 MLA），dummy 权重
  <任意路径>       直接使用

环境变量（建议每条命令都显式写全，包括默认值）：
  KDA_KERNEL=final|baseline     默认 final。被测内核。
                                final    = 本仓库 submodule 钉死的 commit（PR#4 全部优化）
                                baseline = 上游 sglang-jax main 原始 kernel（含已合并的 safe_gate MXU port）
  KDA_FORCE_BASELINE=0|1        默认 0。=1 时模型侧把 safe_gate/fuse/unified_layout/
                                flat_grid/head_block 全部关掉（≡上游 4-stage 路径），
                                gate 语义不变，用于同一构建内的 A/B。
  MODEL=<preset|path>           serve / bench-serving 默认 mini-k3；run-eval 默认 kimi-linear-48b。
  PALLAS_INTERPRET=0|1          默认 0（真硬件）。=1 走 Pallas 解释器，无 TPU 时才用，慢且会掩盖真硬件问题。
  PORT=<int>                    默认 30040。server 端口。
  TP=<int>                      默认 4。张量并行度——默认值要求 v6e-4，不是单芯。
  SERVER_ARGS="<flags>"         默认空。透传给 launch_server，如 "--max-prefill-tokens 2048"。
  MODELS_DIR=<path>             默认 /root/models。dummy preset 的 config 来源。
  GCS_WEIGHTS=<gs://...>        默认 gs://medusa-experiments/model-cache/kimi-linear-48b。
  JAX_COMPILATION_CACHE_DIR=<path>  默认 /root/jax_kernel_cache（镜像 ENV）。
  E2E_MODEL / E2E_TAG / E2E_SERVER_ARGS   仅 e2e 模式。

安全 gate 的实际取值不由环境变量决定：模型 config 声明 gate_lower_bound 时
（Kimi-K3 类）走 safe_gate 快路径，无界 gate（如 Kimi-Linear-48B）保持 generic 路径。

TPU VM 运行需要: --privileged --net=host
结果 JSON 落 /root/，挂载 -v $(pwd)/out:/out 自动收集。
USAGE
    ;;
esac
