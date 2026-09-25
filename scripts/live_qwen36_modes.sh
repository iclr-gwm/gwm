#!/usr/bin/env bash
# Live three-mode GWM test on physical GPU slot 1 with Qwen3.6-27B.
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/live_qwen36_modes.sh
#
# Modes:
#   1) frozen tiny fixture — live advise/select at K=1,2,10
#   2) build a graph from copied toucan-qwen36 10p trajectories, then infer
#   3) self-evolve (force EvolveManager rebuild from seeded rollouts)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_REPO="$(cd "$HERE/../.." && pwd)"
OUTER="$(cd "$VLLM_REPO/.." && pwd)"
if [[ "$(basename "$OUTER")" == "api_vllm" ]]; then
  DEFAULT_DATA="$(cd "$OUTER/.." && pwd)/gwm_data"
else
  DEFAULT_DATA="$OUTER/gwm_data"
fi
cd "$HERE"

PORT="${VLLM_GWM_LIVE_PORT:-8771}"
MODEL="${VLLM_GWM_LIVE_MODEL:-Qwen/Qwen3.6-27B}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
GWM_DATA="${GWM_DATA_ROOT:-$DEFAULT_DATA}"
LIVE_ROOT="${VLLM_GWM_DATA_ROOT:-$GWM_DATA/live}"
SLUG="$(echo "$MODEL" | tr '/A-Z' '_a-z' | tr -cd 'a-z0-9._-')"
PIDFILE="${VLLM_GWM_LIVE_PIDFILE:-$LIVE_ROOT/vllm_${SLUG}.pid}"
LOGFILE="${VLLM_GWM_LIVE_LOGFILE:-$LIVE_ROOT/vllm_${SLUG}.log}"
ATTACH=0
for arg in "$@"; do
  case "$arg" in
    --attach) ATTACH=1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
# FlashInfer sampler JIT needs CUDA 13 nvcc; fall back if the cache is stale.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-$NO_PROXY}"
export GWM_DATA_ROOT="$GWM_DATA"
export RUN_GWM_LIVE=1

echo "== vllm-gwm live_qwen36_modes =="
echo "model=$MODEL port=$PORT gpu=$GPU data=$GWM_DATA attach=$ATTACH"

bash "$HERE/scripts/prep_gwm_data.sh"

resolve_vllm_python() {
  local cand
  for cand in \
    "${VLLM_PYTHON:-}" \
    "$HERE/.venv/bin/python" \
    "$VLLM_REPO/.venv/bin/python" \
    "$VLLM_REPO/extensions/vllm-gwm/.venv/bin/python" \
    "$OUTER/vllm/extensions/vllm-gwm/.venv/bin/python" \
    "$(cd "$OUTER/.." && pwd)/vllm/extensions/vllm-gwm/.venv/bin/python"
  do
    if [[ -n "$cand" && -x "$cand" ]] && "$cand" -c "import vllm" 2>/dev/null; then
      echo "$cand"
      return
    fi
  done
  echo ""
}

VLLM_PY="$(resolve_vllm_python)"
if [[ -z "$VLLM_PY" ]] || ! "$VLLM_PY" -c "import vllm" 2>/dev/null; then
  echo "ERROR: vLLM python not found. export VLLM_PYTHON=/path/to/venv/with/vllm" >&2
  exit 1
fi

echo "installing this tree's vllm-gwm[test,model,server,build] into $VLLM_PY"
"$VLLM_PY" -m pip install -e "$HERE[test,model,server,build]" -q

mkdir -p "$LIVE_ROOT/rollouts" "$LIVE_ROOT/builds"
SEED="$GWM_DATA/rollouts/toucan_qwen36_10p/seed_collection.jsonl"
if [[ -f "$SEED" ]]; then
  cp -a "$SEED" "$LIVE_ROOT/rollouts/rollouts.jsonl"
  echo "seeded $(wc -l < "$SEED") trajectories -> $LIVE_ROOT/rollouts/rollouts.jsonl"
fi

TINY="$GWM_DATA/graphs/tiny"
TOUCAN="$GWM_DATA/graphs/toucan_qwen36_10p"
# evolved starts as the frozen tiny bundle; mode 3 swaps it after rebuild.
CHAT_KWARGS='{"enable_thinking": false}'
export VLLM_PLUGINS=gwm
export VLLM_GWM_MODULES="tiny=${TINY},toucan=${TOUCAN},evolved=${TINY}"
export VLLM_GWM_DATA_ROOT="$LIVE_ROOT"
export VLLM_GWM_PRESET_DIR="$GWM_DATA/fixtures/presets"
export VLLM_GWM_PRESET_DEFAULT=tiny
export VLLM_GWM_MAX_GRAPHS=8
export VLLM_GWM_EMBED_DEVICE="${VLLM_GWM_EMBED_DEVICE:-cpu}"
export VLLM_GWM_BUILD_EMBED_DEVICE="${VLLM_GWM_BUILD_EMBED_DEVICE:-cpu}"
export VLLM_GWM_EVOLVE_EVERY="${VLLM_GWM_EVOLVE_EVERY:-0}"
export VLLM_GWM_JUDGE_TIMEOUT="${VLLM_GWM_JUDGE_TIMEOUT:-300}"
export VLLM_GWM_COLLECT=opt_in
export VLLM_GWM_SHOW_GRAPH="${VLLM_GWM_SHOW_GRAPH:-1}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    echo "stopping vLLM pid=$SERVER_PID"
    pkill -9 -P "$SERVER_PID" 2>/dev/null || true
    kill -9 "$SERVER_PID" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.openai.api_server .* --port ${PORT}" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
}
trap cleanup EXIT

SERVE_EXTRA=()
if [[ "$MODEL" == *[Qq]wen* ]]; then
  SERVE_EXTRA+=(--gdn-prefill-backend triton)
fi

if [[ "$ATTACH" == "0" ]]; then
  echo "starting $MODEL on :$PORT (physical GPU $GPU -> cuda:0) ..."
  CUDA_VISIBLE_DEVICES="$GPU" \
    "$VLLM_PY" -c "import torch; assert torch.cuda.is_available(); print('cuda:0', torch.cuda.get_device_name(0))"
  CUDA_VISIBLE_DEVICES="$GPU" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --language-model-only \
    --max-model-len "${VLLM_GWM_MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${VLLM_GWM_GPU_MEM_UTIL:-0.88}" \
    --enable-prefix-caching \
    --default-chat-template-kwargs "$CHAT_KWARGS" \
    ${SERVE_EXTRA[@]+"${SERVE_EXTRA[@]}"} \
    >"$LOGFILE" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$PIDFILE"
  echo "vLLM log: $LOGFILE pid=$SERVER_PID"
  VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
  export VLLM_BASE_URL
else
  echo "attach mode: $VLLM_BASE_URL"
fi

echo "running live mode tests ($MODEL) ..."
"$VLLM_PY" -m pytest "$HERE/tests/test_live_qwen36_modes.py" -m live -s --tb=short

echo "live_qwen36_modes: OK"
echo "report log: $LOGFILE"
