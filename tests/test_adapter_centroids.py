# SPDX-License-Identifier: Apache-2.0
"""Graph bundles must validate at any embedding dimension, not just 384."""

import json

import numpy as np
import pytest

from vllm_gwm.runtime.adapter import AdapterError, GraphAdapter, find_centroids


def _bundle(root, dim):
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "out/centroids").mkdir(parents=True, exist_ok=True)
    (root / "reports/transitions.json").write_text(json.dumps({"d": {}}))
    np.save(root / f"out/centroids/centroids_{dim}.npy", np.zeros((2, dim), dtype=np.float32))
    (root / "out/centroids/centroid_meta.json").write_text(
        json.dumps({"state_ids": ["d:0", "d:1"], "p90": {}, "model": "m", "dim": dim})
    )
    return root


@pytest.mark.parametrize("dim", [384, 768, 4096])
def test_validates_any_centroid_dimension(tmp_path, dim):
    root = _bundle(tmp_path / f"g{dim}", dim)
    assert GraphAdapter.validate(root, name="g").path == root.resolve()
    assert find_centroids(root).name == f"centroids_{dim}.npy"


def test_missing_centroid_matrix_is_reported(tmp_path):
    root = _bundle(tmp_path / "g", 384)
    (root / "out/centroids/centroids_384.npy").unlink()
    with pytest.raises(AdapterError, match=r"centroids_\*\.npy"):
        GraphAdapter.validate(root, name="g")


def test_generated_manifest_checksums_the_discovered_matrix(tmp_path):
    root = _bundle(tmp_path / "g", 4096)
    info = GraphAdapter.validate(root, name="g")
    assert "out/centroids/centroids_4096.npy" in info.manifest["files"]
