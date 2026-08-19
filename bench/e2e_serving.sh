#!/bin/bash
# End-to-end serving A/B: upstream KDA path vs this PR's kernel, same build.
#
# Usage (on a TPU v6e VM, inside or outside the container):
#   bash e2e_serving.sh <model-dir> <tag> [extra-server-args...]
#
# Runs both sides sequentially:
#   ours  = all ablation switches on (default kernel behavior)
#   base  = KDA_FORCE_BASELINE=1 (all switches off == upstream 4-stage path)
#
# Workload matches the numbers reported in the PR:
#   sharegpt, 160 prompts, max-concurrency 32, warmup 8, request-rate inf
# Results: ~/e2e_<tag>_<side>_bench.log (+ .jsonl with --output-details)
set -e
MODEL=${1:?usage: e2e_serving.sh <model-dir> <tag> [extra server args]}
TAG=${2:?usage: e2e_serving.sh <model-dir> <tag> [extra server args]}
shift 2
EXTRA_ARGS="$*"

PY=${PY:-python}
SGL=${SGL:-$HOME/sglang-jax}
PORT=${PORT:-30040}
NUM_PROMPTS=${NUM_PROMPTS:-160}
CONCURRENCY=${CONCURRENCY:-32}
SHAREGPT=${SHAREGPT:-$HOME/sharegpt.json}
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"

[ -f "$SHAREGPT" ] || curl -sL -o "$SHAREGPT" "$SHAREGPT_URL"

run_side () {
  local SIDE=$1 ENV_PREFIX=$2
  echo "=== $TAG / $SIDE ==="
  pkill -f "sgl_jax.launch_server.*--port $PORT" 2>/dev/null || true
  sleep 5
  ( cd "$SGL" && env $ENV_PREFIX JAX_COMPILATION_CACHE_DIR=$HOME/jax_cache \
      $PY -m sgl_jax.launch_server --model-path "$MODEL" --trust-remote-code \
        --tp-size 4 --dp-size 1 --port $PORT --skip-server-warmup --random-seed 3 \
        --mem-fraction-static 0.85 --dtype bfloat16 --attention-backend fa \
        --page-size 256 --disable-radix-cache $EXTRA_ARGS \
        > $HOME/e2e_${TAG}_${SIDE}_server.log 2>&1 & )
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:$PORT/health 2>/dev/null || true)
    [ "$code" = "200" ] && break
    sleep 15
  done
  [ "$code" = "200" ] || { echo "server failed to become healthy; see e2e_${TAG}_${SIDE}_server.log"; exit 1; }

  ( cd "$SGL" && $PY -m sgl_jax.bench_serving --backend sgl-jax \
      --host 127.0.0.1 --port $PORT \
      --dataset-name sharegpt --dataset-path "$SHAREGPT" \
      --num-prompts $NUM_PROMPTS --max-concurrency $CONCURRENCY --request-rate inf \
      --warmup-requests 8 --output-file $HOME/e2e_${TAG}_${SIDE}.jsonl --output-details \
      2>&1 | tail -45 | tee $HOME/e2e_${TAG}_${SIDE}_bench.log )
}

run_side ours ""
run_side base "KDA_FORCE_BASELINE=1"
pkill -f "sgl_jax.launch_server.*--port $PORT" 2>/dev/null || true

echo "=== summary ($TAG) ==="
for SIDE in base ours; do
  echo "-- $SIDE --"
  grep -E "Total token throughput|Median TTFT|Median TPOT|Benchmark duration" \
    $HOME/e2e_${TAG}_${SIDE}_bench.log || true
done
echo E2E_DONE
