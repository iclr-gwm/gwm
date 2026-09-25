#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$ROOT/tests/fixtures/tiny_graph"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi
export VLLM_PLUGINS=gwm
export VLLM_GWM_DATA_ROOT="${VLLM_GWM_DATA_ROOT:-$ROOT/.gwm_data}"
echo "Smoke: validate tiny graph adapter"
"$PY" - <<PY
from pathlib import Path
from vllm_gwm.runtime.adapter import GraphAdapter
info = GraphAdapter.validate(Path("$FIX"))
print("valid", info.name, info.path)
PY
echo "Run CPU tests"
cd "$ROOT" && "$PY" -m pytest -m "not model and not gpu" -q
