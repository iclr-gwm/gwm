#!/usr/bin/env bash
# Thin wrapper over the in-plugin mining pipeline (vllm_gwm.build.pipeline).
#
#   build_graph.sh <rollouts.jsonl[.gz]> <out_graph_dir> [extra pipeline flags]
#
# <rollouts.jsonl> is ingested into <out_graph_dir>/out/rollouts.jsonl.gz and the
# full DAG (prefix_expand -> embed -> concat -> discover -> mine -> precheck ->
# examples) runs in place, leaving a bundle GraphAdapter.validate() accepts.
#
# Configured through the same knobs the pipeline reads:
#   WM_EMBED_MODEL, WM_EMBED_BATCH, WM_EMBED_TRUST_REMOTE_CODE, EMB_DEVICE,
#   WM_REUSE_STEPS, WM_WATCHED_TOOLS, DISCOVER_ARGS
# Needs the build extra: pip install vllm-gwm[build]
set -euo pipefail
INPUT="${1:?input rollouts jsonl required}"
OUT="${2:?output dir required}"
shift 2
PY="${PYTHON:-python3}"
exec "$PY" -m vllm_gwm.build.pipeline --input "$INPUT" --workdir "$OUT" "$@"
