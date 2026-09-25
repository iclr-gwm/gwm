#!/usr/bin/env bash
# Live integration: start vLLM with GWM (or attach) and run scripts/live_e2e.py.
#
# Usage:
#   bash scripts/live_smoke.sh                          # launch + test
#   VLLM_BASE_URL=http://127.0.0.1:8000/v1 bash scripts/live_smoke.sh --attach
#   VLLM_GWM_LIVE_MODEL=google/gemma-4-E2B-it bash scripts/live_smoke.sh
#
# Env:
#   CUDA_VISIBLE_DEVICES   GPU slot (default 1 → cuda:0 in process)
#   VLLM_GWM_LIVE_PORT     server port (default 8765)
#   VLLM_GWM_LIVE_MODEL    HF model id
#   VLLM_PYTHON            python with vllm + vllm-gwm installed
#   VLLM_USE_PRECOMPILED   pass through for vLLM editable install (default 1)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

ATTACH=0
INSTALL_VLLM=0
for arg in "$@"; do
  case "$arg" in
    --attach) ATTACH=1 ;;
    --install-vllm) INSTALL_VLLM=1 ;;
  esac
done

PORT="${VLLM_GWM_LIVE_PORT:-8765}"
MODEL="${VLLM_GWM_LIVE_MODEL:-google/gemma-4-E2B-it}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
JSON_OUT="${VLLM_GWM_LIVE_JSON:-$HERE/.gwm_data/live_e2e_report.json}"
PIDFILE="${VLLM_GWM_LIVE_PIDFILE:-$HERE/.gwm_data/vllm_live.pid}"
LOGFILE="${VLLM_GWM_LIVE_LOGFILE:-$HERE/.gwm_data/vllm_live.log}"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-$NO_PROXY}"

echo "== vllm-gwm live_smoke =="
echo "model=$MODEL port=$PORT gpu=$GPU attach=$ATTACH"

bash "$HERE/scripts/build_env.sh"

GWM_PY="$HERE/.venv/bin/python"
resolve_vllm_python() {
  if [[ -n "${VLLM_PYTHON:-}" && -x "$VLLM_PYTHON" ]]; then
    echo "$VLLM_PYTHON"
    return
  fi
  if [[ -x "$VLLM_REPO/.venv/bin/python" ]] && "$VLLM_REPO/.venv/bin/python" -c "import vllm" 2>/dev/null; then
    echo "$VLLM_REPO/.venv/bin/python"
    return
  fi
  if command -v vllm >/dev/null 2>&1; then
    echo "$(command -v python3)"
    return
  fi
  echo ""
}

VLLM_PY="$(resolve_vllm_python)"
if [[ -z "$VLLM_PY" && "$INSTALL_VLLM" == "1" ]]; then
  echo "installing vLLM editable into extension venv (precompiled)..."
  export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
  "$GWM_PY" -m pip install -e "$VLLM_REPO" --torch-backend=auto
  VLLM_PY="$GWM_PY"
fi
if [[ -z "$VLLM_PY" ]] || ! "$VLLM_PY" -c "import vllm" 2>/dev/null; then
  if [[ "$ATTACH" == "1" ]]; then
    VLLM_PY="$GWM_PY"
  else
    echo "ERROR: vLLM not found. Either:"
    echo "  1) bash scripts/live_smoke.sh --install-vllm   # slow first-time install"
    echo "  2) export VLLM_PYTHON=/path/to/venv/with/vllm"
    echo "  3) bash scripts/live_smoke.sh --attach  # with server already running"
    exit 1
  fi
fi

# Ensure GWM plugin is visible to the vLLM python env
if ! "$VLLM_PY" -c "import vllm_gwm" 2>/dev/null; then
  echo "installing vllm-gwm into $VLLM_PY env"
  "$VLLM_PY" -m pip install -e "$HERE[test,model,server]"
fi

mkdir -p "$(dirname "$PIDFILE")" "$(dirname "$JSON_OUT")"
TINY="$HERE/tests/fixtures/tiny_graph"
CRM="$HERE/graphs/crm/full"
EOPS="$HERE/graphs/eops/full"
[[ -d "$CRM" ]] || "$HERE/examples/fetch_graphs.sh" "$HERE/graphs"

export VLLM_PLUGINS=gwm
export VLLM_GWM_MODULES="tiny=${TINY},crm=${CRM},eops=${EOPS}"
export VLLM_GWM_DATA_ROOT="${VLLM_GWM_DATA_ROOT:-$HERE/.gwm_data}"
export VLLM_GWM_PRESET_DIR="$HERE/tests/fixtures/presets"
export VLLM_GWM_SHOW_GRAPH="${VLLM_GWM_SHOW_GRAPH:-1}"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping vLLM pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
}
trap cleanup EXIT

if [[ "$ATTACH" == "0" ]]; then
  echo "starting vLLM on :$PORT (GPU $GPU) ..."
  CUDA_VISIBLE_DEVICES="$GPU" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len "${VLLM_GWM_MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization "${VLLM_GWM_GPU_MEM_UTIL:-0.45}" \
    >"$LOGFILE" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$PIDFILE"
  echo "vLLM log: $LOGFILE"
  BASE_URL="http://127.0.0.1:${PORT}/v1"
else
  echo "attach mode: $BASE_URL"
fi

echo "running live_e2e.py ..."
"$GWM_PY" "$HERE/scripts/live_e2e.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --json "$JSON_OUT"

echo "live_smoke: OK"
echo "report: $JSON_OUT"
