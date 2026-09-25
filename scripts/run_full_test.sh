#!/usr/bin/env bash
# Build env and run the full vllm-gwm test matrix with latency reporting.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-$NO_PROXY}"

echo "========== 1/4 build_env =========="
bash "$HERE/scripts/build_env.sh"

PY="$HERE/.venv/bin/python"
JSON_OUT="${VLLM_GWM_E2E_JSON:-$HERE/.gwm_data/e2e_report.json}"
mkdir -p "$(dirname "$JSON_OUT")"

echo ""
echo "========== 2/4 pytest (cpu) =========="
T0=$(date +%s%3N)
"$PY" -m pytest -m "not model and not gpu" -v --tb=short 2>&1 | tee "$HERE/.gwm_data/pytest_cpu.log"
T1=$(date +%s%3N)
echo "pytest_cpu_ms=$((T1 - T0))"

echo ""
echo "========== 3/4 e2e_benchmark (fake scorer + mock vLLM) =========="
T0=$(date +%s%3N)
"$PY" "$HERE/scripts/e2e_benchmark.py" --json "$JSON_OUT"
T1=$(date +%s%3N)
echo "e2e_fake_ms=$((T1 - T0))"

echo ""
echo "========== 4/4 e2e_benchmark (real MiniLM, GPU slot 1 if available) =========="
T0=$(date +%s%3N)
if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "$PY" "$HERE/scripts/e2e_benchmark.py" --model --json "${JSON_OUT%.json}_model.json"
  T1=$(date +%s%3N)
  echo "e2e_model_ms=$((T1 - T0)) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1})"
else
  echo "skip model e2e: torch/CUDA not available in venv"
fi

echo ""
echo "========== DONE =========="
echo "reports: $JSON_OUT"
