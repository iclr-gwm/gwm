#!/usr/bin/env bash
# Live three-mode GWM test: frozen | build | evolve, each on its own GPU, in
# parallel, over both suites. Serves one vLLM+GWM instance per mode, runs the
# matching runner, collects JSON, tears down.
#
# Usage:
#   MODEL=google/gemma-4-E2B-it MODEL_SLUG=gemma4-e2b \
#   ROLLOUTS_ROOT=/path/to/gemma4-rollouts \
#   bash scripts/three_mode_live.sh
#
# Env knobs: GPUS ("0 2 3"), PORTS ("8801 8802 8803"), RATIO (20p), N_TEST (12),
#   OUT (results dir), HF_HOME, VENV, REPO_GRAPHS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MODEL="${MODEL:?set MODEL, e.g. google/gemma-4-E2B-it}"
MODEL_SLUG="${MODEL_SLUG:-$(echo "$MODEL" | tr '/A-Z' '_a-z' | tr -cd 'a-z0-9._-')}"
ROLLOUTS_ROOT="${ROLLOUTS_ROOT:?set ROLLOUTS_ROOT (dir with <suite>/<ratio>/out/rollouts*.jsonl.gz)}"
VENV="${VENV:-$HERE/.venv-live}"
PY="$VENV/bin/python"
REPO_GRAPHS="${REPO_GRAPHS:-$HERE/graphs}"
RATIO="${RATIO:-20p}"
N_TEST="${N_TEST:-12}"
SUITES="${SUITES:-eops,crm}"
OUT="${OUT:-$HERE/.gwm_live/$MODEL_SLUG}"
read -r -a GPU_ARR <<< "${GPUS:-0 2 3}"
read -r -a PORT_ARR <<< "${PORTS:-8801 8802 8803}"
MODES=(frozen build evolve)

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export NO_PROXY="127.0.0.1,localhost"; export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

MODULES="eops=${REPO_GRAPHS}/eops/full,crm=${REPO_GRAPHS}/crm/full"
CHAT_KWARGS='{"enable_thinking": false}'
SERVE_EXTRA=()
[[ "$MODEL" == *[Qq]wen* ]] && SERVE_EXTRA+=(--gdn-prefill-backend triton)

mkdir -p "$OUT"
PIDS=()
cleanup() {
  echo "== teardown =="
  for p in "${PIDS[@]:-}"; do
    [[ -n "$p" ]] || continue
    pkill -9 -P "$p" 2>/dev/null || true
    kill -9 "$p" 2>/dev/null || true
  done
  for port in "${PORT_ARR[@]}"; do
    pkill -9 -f "api_server .* --port ${port}" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "== three_mode_live: $MODEL =="
for i in 0 1 2; do
  mode="${MODES[$i]}"; gpu="${GPU_ARR[$i]}"; port="${PORT_ARR[$i]}"
  droot="$OUT/data/$mode"; mkdir -p "$droot/rollouts"
  log="$OUT/vllm_${mode}.log"
  echo "[$mode] GPU $gpu -> :$port  data=$droot  log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  VLLM_PLUGINS=gwm \
  VLLM_GWM_MODULES="$MODULES" \
  VLLM_GWM_DATA_ROOT="$droot" \
  VLLM_GWM_EMBED_DEVICE=cpu \
  VLLM_GWM_BUILD_EMBED_DEVICE=cpu \
  VLLM_GWM_MAX_GRAPHS=8 \
  VLLM_GWM_COLLECT=opt_in \
  VLLM_GWM_EVOLVE_EVERY=0 \
  VLLM_GWM_SHOW_GRAPH=1 \
    "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$port" \
    --language-model-only --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
    --enable-prefix-caching \
    --default-chat-template-kwargs "$CHAT_KWARGS" \
    ${SERVE_EXTRA[@]+"${SERVE_EXTRA[@]}"} \
    > "$log" 2>&1 &
  PIDS+=("$!")
done

echo "== waiting for servers to become ready =="
for port in "${PORT_ARR[@]}"; do
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "  :$port ready"; break
    fi
    sleep 5
  done
done

echo "== running 3 modes in parallel =="
RPIDS=()
for i in 0 1 2; do
  mode="${MODES[$i]}"; port="${PORT_ARR[$i]}"
  "$PY" "$HERE/scripts/three_mode_runner.py" \
    --base "http://127.0.0.1:${port}/v1" --model "$MODEL" \
    --mode "$mode" --suites "$SUITES" \
    --rollouts-root "$ROLLOUTS_ROOT" --ratio "$RATIO" \
    --built-root "$OUT/built" --data-root "$OUT/data/$mode" \
    --n-test "$N_TEST" --k-values "${K_VALUES:-1,4}" \
    --out "$OUT/${mode}.json" > "$OUT/runner_${mode}.log" 2>&1 &
  RPIDS+=("$!")
done

FAIL=0
for rp in "${RPIDS[@]}"; do wait "$rp" || FAIL=1; done

echo "== results =="
for mode in "${MODES[@]}"; do
  echo "--- $mode ($OUT/${mode}.json) ---"
  "$PY" - "$OUT/${mode}.json" <<'PY' || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  (no result:", e, ")"); raise SystemExit
for suite, ent in d.get("suites", {}).items():
    ev = ent.get("eval", {})
    extra = ""
    if "build" in ent: extra = f" built_states={ent['build'].get('n_states')} activate={ent['build'].get('activate_status')}"
    if "evolve" in ent: extra = f" evolve_ok={ent['evolve'].get('ok')} seeded={ent['evolve'].get('seeded')}"
    print(f"  {suite}: n={ev.get('n_tasks')} injected={ev.get('n_injected')} "
          f"avg_latency_ms={ev.get('avg_latency_ms')} gold_hit={ev.get('gold_hit_rate')}{extra}")
PY
done
echo "three_mode_live: done (fail=$FAIL) -> $OUT"
exit "$FAIL"
