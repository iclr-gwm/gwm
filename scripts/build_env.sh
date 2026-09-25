#!/usr/bin/env bash
# Bootstrap the vllm-gwm extension environment and optional extras.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

EXTRAS="${VLLM_GWM_INSTALL_EXTRAS:-test,model,server}"
VENV="$HERE/.venv"
PY="${VLLM_GWM_PYTHON:-$VENV/bin/python}"

echo "== vllm-gwm build_env =="
echo "root: $HERE"

if [[ ! -x "$PY" ]]; then
  echo "creating venv at $VENV"
  python3 -m venv "$VENV"
  PY="$VENV/bin/python"
fi

"$PY" -m pip install --quiet --upgrade pip
IFS=',' read -ra PARTS <<< "$EXTRAS"
if ((${#PARTS[@]})); then
  SPEC=".[$(IFS=,; echo "${PARTS[*]}")]"
else
  SPEC="."
fi
echo "installing editable package: $SPEC"
"$PY" -m pip install -e "$SPEC"

# graph bundles (skip large examples.json)
if [[ -x "$HERE/examples/fetch_graphs.sh" ]]; then
  "$HERE/examples/fetch_graphs.sh" "$HERE/graphs" || true
fi

echo "python: $("$PY" --version)"
"$PY" -c "import vllm_gwm; print('vllm_gwm', vllm_gwm.__version__)"
"$PY" -c "
import importlib.metadata as m
eps = [e for e in m.entry_points(group='vllm.endpoint_plugins') if e.name=='gwm']
print('entrypoint:', eps[0].value if eps else 'MISSING')
"

echo "build_env: OK"
