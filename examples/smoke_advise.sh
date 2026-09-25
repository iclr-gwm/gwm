#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/examples/fetch_graphs.sh" "$ROOT/graphs"
export VLLM_PLUGINS=gwm
echo "Example: vllm serve MODEL --gwm-modules eops=$ROOT/graphs/eops/full --gwm-modules crm=$ROOT/graphs/crm/full"
