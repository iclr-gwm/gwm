#!/usr/bin/env bash
# Fetch full CRM/EOPS graph bundles from a local graph-artifacts directory.
# Set GWM_GRAPH_SRC to the directory that holds <suite>/full bundles.
set -euo pipefail
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/graphs}"
SRC="${GWM_GRAPH_SRC:?set GWM_GRAPH_SRC to the source graphs directory}"
mkdir -p "$DEST"
for suite in crm eops; do
  if [[ -d "$SRC/$suite/full" ]]; then
    mkdir -p "$DEST/$suite/full"
    rsync -a --exclude 'reports/examples.json' "$SRC/$suite/full/" "$DEST/$suite/full/"
    echo "synced $suite/full -> $DEST/$suite/full"
  fi
done
