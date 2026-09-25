#!/usr/bin/env bash
# Build a fresh, reproducible environment for the live three-mode GWM tests.
#
# Creates an isolated venv and installs:
#   - vLLM (from $VLLM_SRC, reusing precompiled kernels), and
#   - this vllm-gwm tree with [test,model,server,build] extras.
#
# vLLM must come from the source tree the GWM plugin targets (its frontend API
# paths are newer than any PyPI release), so point $VLLM_SRC at that checkout.
#
# Usage:
#   VLLM_SRC=/path/to/vllm bash scripts/fresh_env.sh
#   VENV=.venv-live VLLM_SRC=/path/to/vllm bash scripts/fresh_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VENV="${VENV:-$HERE/.venv-live}"
PY_VERSION="${PY_VERSION:-3.12}"
VLLM_SRC="${VLLM_SRC:?set VLLM_SRC to the vLLM source checkout the plugin targets}"

echo "== fresh_env =="
echo "venv=$VENV python=$PY_VERSION vllm_src=$VLLM_SRC"

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PY_VERSION" "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
else
  python${PY_VERSION} -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi

echo "installing vLLM (precompiled) from $VLLM_SRC ..."
VLLM_USE_PRECOMPILED=1 "${PIP[@]}" -e "$VLLM_SRC"

echo "installing vllm-gwm[test,model,server,build] ..."
"${PIP[@]}" -e "$HERE[test,model,server,build]"

echo "== versions =="
"$VENV/bin/python" - <<'PY'
import vllm, vllm_gwm, torch
print("vllm     ", vllm.__version__)
print("torch    ", torch.__version__, "cuda", torch.cuda.is_available())
print("vllm_gwm ", vllm_gwm.__version__)
PY
echo "fresh_env: OK -> $VENV"
