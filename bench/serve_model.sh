#!/bin/bash
# Prepare a model directory and launch sglang-jax's official server.
#
#   bash serve_model.sh <preset|path> [extra launch_server args...]
#
# Presets:
#   kimi-linear-48b   real weights, pulled from GCS model-cache (~92 GB -> /dev/shm)
#   mini-k3           K3 geometry, 12 layers (9 KDA + 3 MLA), dummy weights
#   mini-k3-half      K3 geometry, 24 layers (18 KDA + 6 MLA), dummy weights
#   /any/path         use as-is
#
# Env:
#   KDA_FORCE_BASELINE=1  run the upstream KDA path (all ablation switches off)
#   PORT                  default 30040
#   TP                    default 4
# Prints "SERVER_READY" once /health returns 200.
set -e
TARGET=${1:?usage: serve_model.sh <preset|path> [extra args]}
shift || true
EXTRA="$*"

PY=${PY:-python}
SGL=${SGL:-/root/sglang-jax}
PORT=${PORT:-30040}
TP=${TP:-4}
MODELS_DIR=${MODELS_DIR:-/root/models}
GCS_WEIGHTS=${GCS_WEIGHTS:-gs://medusa-experiments/model-cache/kimi-linear-48b}

prep_dummy () {  # prep_dummy <config-file> <dir>
  local CFG=$1 DIR=$2
  mkdir -p "$DIR"
  # tokenizer/aux files come from the released Kimi-Linear repo; weights are dummy
  if [ ! -f "$DIR/tiktoken.model" ]; then
    for f in $(gsutil ls "$GCS_WEIGHTS/" | grep -vE "safetensors|/figures/$|/$"); do
      gsutil -q cp "$f" "$DIR/"
    done
  fi
  cp "$CFG" "$DIR/config.json"
  echo "$DIR"
}

case "$TARGET" in
  kimi-linear-48b)
    MODEL=/dev/shm/kimi-linear-48b
    if [ ! -f "$MODEL/config.json" ]; then
      mkdir -p "$MODEL"
      echo ">>> pulling real weights (~92 GB) from $GCS_WEIGHTS"
      gsutil -m -q cp -r "$GCS_WEIGHTS/*" "$MODEL/"
    fi
    LOAD_FMT=""
    ;;
  mini-k3)
    MODEL=$(prep_dummy "$MODELS_DIR/mini-k3-config.json" /root/mini-k3)
    LOAD_FMT="--load-format dummy"
    ;;
  mini-k3-half)
    MODEL=$(prep_dummy "$MODELS_DIR/mini-k3-half-config.json" /root/mini-k3-half)
    LOAD_FMT="--load-format dummy"
    ;;
  *)
    MODEL=$TARGET
    LOAD_FMT=${LOAD_FORMAT:-}
    ;;
esac

echo ">>> model: $MODEL  (kernel: ${KDA_FORCE_BASELINE:+upstream baseline}${KDA_FORCE_BASELINE:-this branch})"
pkill -f "sgl_jax.launch_server.*--port $PORT" 2>/dev/null || true
sleep 3
( cd "$SGL" && JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/jax_cache} \
    $PY -m sgl_jax.launch_server --model-path "$MODEL" $LOAD_FMT --trust-remote-code \
      --tp-size $TP --dp-size 1 --port $PORT --skip-server-warmup --random-seed 3 \
      --mem-fraction-static 0.85 --dtype bfloat16 --attention-backend fa \
      --page-size 256 --disable-radix-cache $EXTRA \
      > /root/server.log 2>&1 & )

for i in $(seq 1 160); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:$PORT/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "SERVER_READY (after ~$((i*15))s)"; exit 0; }
  sleep 15
done
echo "SERVER_FAILED — tail of /root/server.log:"
tail -30 /root/server.log
exit 1
