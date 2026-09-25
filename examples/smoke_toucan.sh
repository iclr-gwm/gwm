#!/usr/bin/env bash
# Toucan live checks against an already-served vLLM GWM endpoint (GPU slot 1).
# Does not start vLLM — attach only, so it will not steal a running eval/gen job.
#
#   export VLLM_BASE_URL=http://127.0.0.1:8497/v1
#   bash examples/smoke_toucan.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${VLLM_BASE_URL:-http://127.0.0.1:8497/v1}"
GRAPH="${TOUCAN_GWM_GRAPH:-${HOME}/repo/gwm_artifacts/toucan/full}"

echo "Toucan GWM smoke (attach-only)"
echo "  base=${BASE}"
echo "  graph=${GRAPH}"
echo
echo "Serve (GPU 1, positional model is mapped to --model by vllm-gwm CLI):"
cat <<EOF
export CUDA_VISIBLE_DEVICES=1
export VLLM_PLUGINS=gwm
export VLLM_GWM_MODULES="toucan=${GRAPH}"
export VLLM_GWM_PRESET_DEFAULT=toucan
export VLLM_GWM_SHOW_GRAPH=1
python -m vllm_gwm.cli serve --gwm-show-graph \\
  google/gemma-4-31b-it --served-model-name gemma4-31b-it \\
  --host 127.0.0.1 --port 8497 --language-model-only
# preflight: curl -s ${BASE%/}/models | jq '.data[0].root'
# must be google/gemma-4-31b-it (not the default Qwen/Qwen3-0.6B)
EOF
echo

if ! curl -sf -m 5 "${BASE}/models" >/dev/null; then
  echo "no server at ${BASE} — printed serve recipe only"
  exit 0
fi

echo "=== GET /v1/gwm/info ==="
curl -sf "${BASE}/gwm/info" | python3 -m json.tool | head -40
echo
echo "=== POST /v1/gwm/advise (first-turn harvest flow; often abstain) ==="
curl -sf -m 60 "${BASE}/gwm/advise" \
  -H 'Content-Type: application/json' \
  -d '{"conversation_flow":[{"type":"user_message","content":"What is the weather in Paris?"}],"preset":"toucan","adapter":"toucan","episode_id":"docs-demo-1"}' \
  | python3 -m json.tool
echo
echo "=== vllm-gwm inspect ==="
if [[ -d "${GRAPH}" ]]; then
  "${ROOT}/.venv/bin/python" -m vllm_gwm.cli inspect "${GRAPH}" || true
fi
echo "smoke_toucan: OK"
