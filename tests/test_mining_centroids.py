# SPDX-License-Identifier: MIT
"""Stage 6 (precheck): the on-disk centroid contract the runtime reads.

``out/centroids/centroids_<dim>.npy`` + ``centroid_meta.json`` are what
``GraphAdapter`` validates and ``graph_sidecar.Scorer`` loads, so the key set,
the dimension-named matrix and the embedding provenance block are asserted here
rather than only end-to-end.
"""

import gzip
import json

import numpy as np
import pytest

from vllm_gwm.build.mining.precheck import parse_tools, precheck
from vllm_gwm.runtime.adapter import find_centroids

DIM = 8


def _write_bundle_inputs(work, *, dim=DIM, with_test=True, embed_sidecar=None):
    out = work / "out"
    reports = work / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    rows = []
    vectors = []
    centers = {c: rng.normal(size=dim) for c in ("toy:0", "toy:1")}
    for c, center in centers.items():
        for i in range(6):
            v = center + 0.05 * rng.normal(size=dim)
            vectors.append(v / np.linalg.norm(v))
            rows.append({
                "step_uid": f"{c.replace(':', '')}-tr{i}", "uid": f"u-{c}-{i}",
                "task_id": f"t{i}", "domain": "toy", "split": "train",
                "step_idx": 1, "n_steps": 1, "tool_names": ["execute"],
                "obs_outcome": "ok", "overall_success": i % 2 == 0,
                "cluster": c,
            })
    if with_test:
        for c, center in centers.items():
            v = center + 0.05 * rng.normal(size=dim)
            vectors.append(v / np.linalg.norm(v))
            rows.append({
                "step_uid": f"{c.replace(':', '')}-te", "uid": f"u-{c}-test",
                "task_id": "t-test", "domain": "toy", "split": "test",
                "step_idx": 1, "n_steps": 1, "tool_names": ["execute"],
                "obs_outcome": "error", "overall_success": False,
                "cluster": c,
            })

    np.savez(out / "emb_st_all.npz", X=np.array(vectors, dtype=np.float32))
    with gzip.open(out / "states_all.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (reports / "transitions.json").write_text(json.dumps({
        "toy": {"base_fail_rate": 0.5, "n_rollouts": 14,
                "state_fail": [{"state": "toy:0", "fail_rate": 0.9, "n_rollouts": 25},
                               {"state": "toy:1", "fail_rate": 0.1, "n_rollouts": 25}]}
    }))
    if embed_sidecar is not None:
        (out / "emb_st.config.json").write_text(json.dumps(embed_sidecar))
    return work


def test_centroid_meta_contract(tmp_path):
    work = _write_bundle_inputs(tmp_path / "g")
    res = precheck(work, watched_tools="execute", embed_model="acme/embed-8",
                   embed_device="cpu", embed_batch=32)

    cent_path = find_centroids(work)
    assert cent_path.name == f"centroids_{DIM}.npy"        # named for the dimension
    cents = np.load(cent_path)
    meta = json.loads((work / "out/centroids/centroid_meta.json").read_text())

    assert set(meta) >= {"state_ids", "p90", "top_tool", "model", "dim", "embedding",
                         "tail_chars", "max_tool_result_chars", "max_flow_chars"}
    assert meta["state_ids"] == ["toy:0", "toy:1"]         # sorted, noise dropped
    assert cents.shape == (2, DIM) and cents.dtype == np.float32
    assert meta["dim"] == DIM and meta["model"] == "acme/embed-8"
    assert meta["top_tool"] == {"toy:0": "execute", "toy:1": "execute"}
    assert set(meta["p90"]) == {"toy:0", "toy:1"}
    assert meta["max_tool_result_chars"] == 2000 and meta["max_flow_chars"] == 60000
    # centroids are unit vectors (cosine space)
    assert np.allclose(np.linalg.norm(cents, axis=1), 1.0, atol=1e-5)
    # provenance mirror, standalone
    emb = json.loads((work / "reports/EMBEDDING.json").read_text())
    assert emb["centroids"] == f"centroids_{DIM}.npy" and emb["n_states"] == 2
    assert res["n_states"] == 2 and res["dim"] == DIM
    assert (work / "reports/PRECHECK.md").exists()


def test_stage2_sidecar_is_authoritative_for_the_embedding_block(tmp_path):
    sidecar = {"model": "sidecar/model", "model_slug": "sidecar-model", "dim": DIM,
               "window": "tail", "tail_chars": 512, "normalize_embeddings": True}
    work = _write_bundle_inputs(tmp_path / "g", embed_sidecar=sidecar)
    precheck(work, embed_model="ignored/because-sidecar-wins")
    meta = json.loads((work / "out/centroids/centroid_meta.json").read_text())
    assert meta["model"] == "sidecar/model"
    assert meta["tail_chars"] == 512
    assert "stage2 sidecar" in meta["embedding"]["source"]


def test_no_test_split_still_writes_centroids(tmp_path):
    work = _write_bundle_inputs(tmp_path / "g", with_test=False)
    res = precheck(work)
    assert res["verdict"] == "REVIEW" and res["test_steps"] == 0
    assert find_centroids(work) is not None
    assert "No test split" in (work / "reports/PRECHECK.md").read_text()


def test_all_noise_states_is_a_hard_error(tmp_path):
    work = _write_bundle_inputs(tmp_path / "g")
    states = work / "out/states_all.jsonl.gz"
    with gzip.open(states, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        r["cluster"] = "noise"
    with gzip.open(states, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="zero"):
        precheck(work)


def test_parse_tools_accepts_strings_and_iterables():
    assert parse_tools("a, b  c") == {"a", "b", "c"}
    assert parse_tools(["a", "b"]) == {"a", "b"}
    assert parse_tools("") == set()
    assert parse_tools(None) == set()
