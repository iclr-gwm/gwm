# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from vllm_gwm.runtime.adapter import GraphAdapter

ROOT = Path(__file__).resolve().parents[1]


def test_crm_graph_bundle_valid():
    path = ROOT / "graphs" / "crm" / "full"
    if not path.is_dir():
        return
    info = GraphAdapter.validate(path, name="crm")
    assert info.name == "crm"
    assert (path / "reports/transitions.json").exists()
