#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "skip: torch not installed in $PY"
  exit 0
fi
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required for GPU smoke"
assert torch.cuda.current_device() == 0, "expected physical GPU slot mapped to cuda:0"
print("gpu smoke ok on", torch.cuda.get_device_name(0))
PY
echo "GPU slot 1 smoke passed (cuda:0 inside process)"
