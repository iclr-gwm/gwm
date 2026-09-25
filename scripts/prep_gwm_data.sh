#!/usr/bin/env bash
# Copy frozen graphs + rollout corpora into a sibling gwm_data tree so live
# Qwen3.6 runs (frozen / build / self-evolve) have everything in one place.
#
# Default destination: <vllm-repo>/../gwm_data
# (or <api_vllm>/../gwm_data when the plugin lives under an api_vllm wrap).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ROOT="$(cd "$HERE/../.." && pwd)"
OUTER="$(cd "$VLLM_ROOT/.." && pwd)"
if [[ "$(basename "$OUTER")" == "api_vllm" ]]; then
  DEFAULT_DATA="$(cd "$OUTER/.." && pwd)/gwm_data"
  REPO_PARENT="$(cd "$OUTER/.." && pwd)"
else
  DEFAULT_DATA="$OUTER/gwm_data"
  REPO_PARENT="$OUTER"
fi
DEST="${GWM_DATA_ROOT:-$DEFAULT_DATA}"
ART="${GWM_ARTIFACTS:-$REPO_PARENT/gwm_artifacts}"
GWM_REPO="${GWM_GRAPH_SRC:-$REPO_PARENT/gwm/graphs}"

TINY="$HERE/tests/fixtures/tiny_graph"
CRM="$HERE/graphs/crm/full"
EOPS="$HERE/graphs/eops/full"
TOUCAN10="$ART/toucan/qwen36/10p"

echo "== prep_gwm_data =="
echo "dest=$DEST"

mkdir -p "$DEST/graphs" "$DEST/rollouts/toucan_qwen36_10p" "$DEST/fixtures/presets" \
  "$DEST/builds" "$DEST/live"

rsync -a --delete "$TINY/" "$DEST/graphs/tiny/"
echo "copied graphs/tiny (frozen test fixture)"

if [[ ! -d "$CRM" ]]; then
  "$HERE/examples/fetch_graphs.sh" "$HERE/graphs" || true
fi
if [[ -d "$CRM" ]]; then
  mkdir -p "$DEST/graphs/crm/full"
  rsync -a --exclude 'reports/examples.json' "$CRM/" "$DEST/graphs/crm/full/"
  echo "copied graphs/crm/full"
fi
if [[ -d "$EOPS" ]]; then
  mkdir -p "$DEST/graphs/eops/full"
  rsync -a --exclude 'reports/examples.json' "$EOPS/" "$DEST/graphs/eops/full/"
  echo "copied graphs/eops/full"
elif [[ -d "$GWM_REPO/eops/full" ]]; then
  mkdir -p "$DEST/graphs/eops/full"
  rsync -a --exclude 'reports/examples.json' "$GWM_REPO/eops/full/" "$DEST/graphs/eops/full/"
  echo "copied graphs/eops/full from gwm repo"
fi

if [[ -d "$TOUCAN10" ]]; then
  rsync -a "$TOUCAN10/" "$DEST/graphs/toucan_qwen36_10p/"
  echo "copied graphs/toucan_qwen36_10p (910 train rollouts, PRECHECK GO)"
  for f in rollouts.jsonl.gz rollouts_test.jsonl.gz rollouts_all.jsonl.gz; do
    if [[ -f "$TOUCAN10/out/$f" ]]; then
      cp -a "$TOUCAN10/out/$f" "$DEST/rollouts/toucan_qwen36_10p/$f"
    fi
  done
else
  echo "WARN: toucan qwen36 10p artifacts missing at $TOUCAN10" >&2
fi

cp -a "$HERE/tests/fixtures/crm_retry_flow.json" "$DEST/fixtures/"
cp -a "$HERE/tests/fixtures/eops_retry_flow.json" "$DEST/fixtures/"
cp -a "$HERE/tests/fixtures/presets/tiny.toml" "$DEST/fixtures/presets/"

PY="${GWM_PYTHON:-$HERE/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi
"$PY" - "$DEST" <<'PY'
import gzip
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
src = dest / "rollouts/toucan_qwen36_10p/rollouts.jsonl.gz"
seed = dest / "rollouts/toucan_qwen36_10p/seed_collection.jsonl"
sample = dest / "fixtures/toucan_prefix_flow.json"
n = 0
sample_flow = None
if src.is_file():
    with gzip.open(src, "rt", encoding="utf-8") as inf, seed.open(
        "w", encoding="utf-8"
    ) as out:
        for line in inf:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = str(rec.get("uid") or rec.get("task_id") or f"r{n}")
            flow = rec.get("conversation_flow") or rec.get("flow") or []
            if sample_flow is None and flow:
                sample_flow = flow
            row = {
                "id": uid,
                "episode_id": uid,
                "uid": uid,
                "task_id": rec.get("task_id") or uid,
                "flow": flow,
                "conversation_flow": flow,
                "success": rec.get("overall_success"),
                "overall_success": rec.get("overall_success"),
                "pass_rate": rec.get("pass_rate"),
                "domain": rec.get("domain") or "toucan",
                "split": rec.get("split") or "train",
                "finalized": True,
                "source": "seed",
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    if sample_flow is not None:
        sample.write_text(json.dumps(sample_flow, indent=2), encoding="utf-8")
manifest = {
    "dest": str(dest),
    "frozen_graphs": ["tiny", "crm/full", "eops/full", "toucan_qwen36_10p"],
    "min_train_rollouts_for_hdbscan": 50,
    "toucan_qwen36_10p_train_rollouts": n,
    "note": (
        "HDBSCAN default min_cluster_size=50; toucan qwen36 10p (910 train) "
        "is the smallest in-tree corpus that already produced a PRECHECK GO graph."
    ),
}
(dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"seed_collection.jsonl rows={n}")
print(f"wrote {dest / 'MANIFEST.json'}")
PY

echo "prep_gwm_data: OK -> $DEST"
cat "$DEST/MANIFEST.json"
