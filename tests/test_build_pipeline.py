# SPDX-License-Identifier: MIT
"""End-to-end build DAG on a tiny synthetic corpus.

Runs all 7 stages of :func:`vllm_gwm.build.pipeline.build_graph` with a stub
embedder (no model download, no GPU) and a k-means stand-in for UMAP+HDBSCAN,
then asserts the produced directory is a real bundle: ``GraphAdapter.validate()``
accepts it and ``graph_sidecar.Scorer`` loads and scores against it.
"""

import gzip
import json

import numpy as np
import pytest

from vllm_gwm.build import pipeline
from vllm_gwm.runtime.adapter import GraphAdapter, find_centroids

pytestmark = pytest.mark.build

DISCOVER = {"within_domain": True, "min_cluster_size": 2, "n_components": 2,
            "n_neighbors": 3, "seed": 17}


def _flow(*, error_first: bool, answer: str):
    """A 3-action flow: query -> (maybe repair) -> answer."""
    flow = [
        {"type": "system_message", "content": "You are a data agent."},
        {"type": "user_message", "content": f"Find {answer}."},
        {"type": "ai_message", "content": "", "tool_calls": [
            {"name": "execute", "args": {"q": f"SELECT * FROM T WHERE x='{answer}'"}}]},
        {"type": "tool_result", "tool_name": "execute",
         "result": ({"success": False, "error": "no such table: T"} if error_first
                    else {"success": True, "result": {"rows": [answer]}})},
        {"type": "ai_message", "content": "", "tool_calls": [
            {"name": "describe", "args": {"table": "T"}}]},
        {"type": "tool_result", "tool_name": "describe",
         "result": {"success": True, "result": {"columns": ["x"]}}},
        {"type": "ai_message", "content": f"The answer is {answer}.", "tool_calls": []},
    ]
    return flow


def _write_rollouts(work, path="out/rollouts_all.jsonl.gz"):
    """5 rollouts (4 train / 1 test), mixed outcomes, one domain."""
    specs = [
        ("r0", "train", True, False, "acme"),
        ("r1", "train", False, True, "globex"),
        ("r2", "train", True, False, "initech"),
        ("r3", "train", False, True, "umbrella"),
        ("r4", "test", True, False, "stark"),
    ]
    target = work / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as f:
        for uid, split, ok, err, name in specs:
            f.write(json.dumps({
                "uid": uid, "task_id": f"task-{uid}", "domain": "toy",
                "split": split, "overall_success": ok,
                "pass_rate": 1.0 if ok else 0.0,
                "conversation_flow": _flow(error_first=err, answer=name),
            }) + "\n")
    return target


@pytest.fixture
def built(tmp_path, stub_sentence_transformers, stub_labeller):
    work = tmp_path / "graph"
    _write_rollouts(work)
    res = pipeline.build_graph(
        work, embed_model="stub/tiny-embedder", embed_device="cpu", embed_batch=4,
        watched_tools="execute", discover_args=DISCOVER, examples_per=3)
    return work, res


def test_all_stages_run_and_write_their_artifacts(built):
    work, res = built
    assert list(res["stages"]) == list(pipeline.STAGES)
    for rel in (
        "out/rollouts.jsonl.gz", "out/rollouts_test.jsonl.gz",
        "out/steps.jsonl.gz", "out/steps_test.jsonl.gz",
        "out/emb_st.npz", "out/emb_st.meta.jsonl", "out/emb_st.config.json",
        "out/emb_st_all.npz", "out/emb_st_all.meta.jsonl",
        "out/states_all.jsonl.gz", "out/centroids/centroid_meta.json",
        "reports/transitions.json", "reports/WORKFLOW_GRAPH.md",
        "reports/PRECHECK.md", "reports/EMBEDDING.json", "reports/STATES.md",
        "reports/examples.json", "reports/EXAMPLES.md", "MANIFEST.json",
    ):
        assert (work / rel).exists(), rel
    # stage 0b split the tagged corpus; 4 train + 1 test rollouts -> 3 steps each
    assert res["stages"]["prefix_expand"]["train"]["steps"] == 12
    assert res["stages"]["prefix_expand"]["test"]["steps"] == 3
    assert res["stages"]["concat"]["n"] == 15


def test_bundle_matches_the_runtime_contract(built):
    work, res = built
    cent_path = find_centroids(work)
    assert cent_path.name == "centroids_16.npy"        # stub embedder dimension
    cents = np.load(cent_path)
    meta = json.loads((work / "out/centroids/centroid_meta.json").read_text())
    assert set(meta) >= {"state_ids", "p90", "top_tool", "model", "dim", "embedding",
                         "tail_chars", "max_tool_result_chars", "max_flow_chars"}
    assert cents.shape == (len(meta["state_ids"]), 16)
    assert meta["model"] == "stub/tiny-embedder" and meta["dim"] == 16
    assert all(s.startswith("toy:") for s in meta["state_ids"])
    assert meta["embedding"]["window"] == "tail"
    assert meta["embedding"]["normalize_embeddings"] is True

    trans = json.loads((work / "reports/transitions.json").read_text())
    assert set(trans) == {"toy"}
    dom = trans["toy"]
    assert dom["n_rollouts"] == 5
    assert set(dom) >= {"base_fail_rate", "P_next", "divergence", "state_fail",
                        "trans_fail", "tool_edges"}
    assert {sf["state"] for sf in dom["state_fail"]} >= set(meta["state_ids"])
    # tool-conditioned edges keep the toolset key shape the runtime looks up
    assert any("execute" in acts for acts in dom["tool_edges"].values())

    ex = json.loads((work / "reports/examples.json").read_text())
    assert set(ex) == {"toy"}
    assert set(ex["toy"]) == {"states", "transitions", "tool_transitions", "neg_states"}
    assert all(k in meta["state_ids"] for k in ex["toy"]["states"])
    first = next(iter(v for v in ex["toy"]["states"].values() if v))[0]
    assert {"step_uid", "uid", "step_idx", "cos", "text"} <= set(first)
    # the errored first action of r1/r3 is mined as a contrastive negative
    assert sum(len(v) for v in ex["toy"]["neg_states"].values()) >= 1
    assert res["stages"]["examples"]["states_covered"] >= 1


def test_bundle_validates_and_loads_in_the_scorer(built, stub_sentence_transformers):
    work, _ = built
    info = GraphAdapter.validate(work, name="synthetic")
    assert info.path == work.resolve()
    assert "reports/transitions.json" in info.manifest["files"]

    from vllm_gwm.harness import graph_sidecar

    scorer = graph_sidecar.Scorer(
        str(work / "reports/transitions.json"), str(work / "out/centroids"),
        examples_path=str(work / "reports/examples.json"), log_requests=False)
    meta = json.loads((work / "out/centroids/centroid_meta.json").read_text())
    assert scorer.ids == meta["state_ids"]
    assert scorer.model_name == "stub/tiny-embedder"

    live = _flow(error_first=True, answer="acme")[:4]
    state, dist = scorer.classify(live, "toy")
    assert state is None or state in scorer.ids
    assert 0.0 <= dist <= 2.0
    res = scorer.score_one(live, "toy", "graph_full")
    assert set(res) >= {"state", "cos_dist", "abstain", "is_trap", "injected", "block"}


def test_manifest_checksums_survive_revalidation(built):
    work, _ = built
    manifest = json.loads((work / "MANIFEST.json").read_text())
    assert manifest["built_by"] == "vllm_gwm.build.pipeline"
    assert manifest["embedding"]["model"] == "stub/tiny-embedder"
    GraphAdapter.validate(work)                     # checksums match as written
    (work / "reports/transitions.json").write_text("{}")
    with pytest.raises(Exception, match="checksum mismatch"):
        GraphAdapter.validate(work)


def test_reuse_steps_skips_stage_1_and_reports_md5(tmp_path, stub_sentence_transformers,
                                                   stub_labeller, capsys):
    work = tmp_path / "graph"
    _write_rollouts(work)
    pipeline.build_graph(work, embed_model="stub/tiny-embedder", embed_device="cpu",
                         discover_args=DISCOVER, stages=["prefix_expand"])
    before = (work / "out/steps.jsonl.gz").read_bytes()
    res = pipeline.build_graph(work, reuse_steps=True, stages=["prefix_expand"])
    stage = res["stages"]["prefix_expand"]
    assert stage["reused"] == ["steps.jsonl.gz", "steps_test.jsonl.gz"]
    assert len(stage["md5"]["steps.jsonl.gz"]) == 32
    assert "md5=" in capsys.readouterr().out
    assert (work / "out/steps.jsonl.gz").read_bytes() == before   # not regenerated


def test_reuse_steps_without_steps_fails_loudly(tmp_path):
    with pytest.raises(pipeline.BuildError, match="reuse_steps"):
        pipeline.build_graph(tmp_path / "empty", reuse_steps=True,
                             stages=["prefix_expand"])


def test_missing_rollouts_names_the_unported_harvest_stage(tmp_path):
    with pytest.raises(pipeline.BuildError, match="harvest"):
        pipeline.build_graph(tmp_path / "empty", stages=["prefix_expand"])


def test_unknown_stage_is_rejected(tmp_path):
    with pytest.raises(pipeline.BuildError, match="unknown stage"):
        pipeline.build_graph(tmp_path / "empty", stages=["mine", "nope"])


def test_ingest_normalizes_collection_store_records(tmp_path):
    """The collect-store shape (id/episode_id/flow/success, one row per turn)."""
    snap = tmp_path / "rollouts.jsonl"
    flow = _flow(error_first=False, answer="acme")
    rows = [
        {"id": "1", "episode_id": "ep-a", "flow": flow[:4], "success": None,
         "finalized": False},
        {"id": "2", "episode_id": "ep-a", "flow": flow, "success": True,
         "finalized": True},
        {"id": "3", "episode_id": "ep-b", "flow": flow, "success": False,
         "finalized": True},
    ]
    snap.write_text("".join(json.dumps(r) + "\n" for r in rows))
    work = tmp_path / "w"
    info = pipeline.ingest_rollouts(snap, work, default_domain="crm")
    assert info == {"train": 2, "test": 0, "workdir": str(work.resolve())}
    assert not (work / "out/rollouts_test.jsonl.gz").exists()
    with gzip.open(work / "out/rollouts.jsonl.gz", "rt", encoding="utf-8") as f:
        out = [json.loads(line) for line in f]
    assert [r["uid"] for r in out] == ["ep-a", "ep-b"]        # de-duped per episode
    assert [len(r["conversation_flow"]) for r in out] == [7, 7]  # finalized row wins
    assert [r["overall_success"] for r in out] == [True, False]
    assert {r["domain"] for r in out} == {"crm"}
    assert {r["split"] for r in out} == {"train"}


def test_ingest_rejects_an_empty_corpus(tmp_path):
    snap = tmp_path / "empty.jsonl"
    snap.write_text("")
    with pytest.raises(pipeline.BuildError, match="no non-test rollouts"):
        pipeline.ingest_rollouts(snap, tmp_path / "w")


def test_discover_args_accept_the_shell_string_form():
    args = pipeline.parse_discover_args(
        "--within-domain --min-cluster-size 25 --n-components 5 --n-neighbors 7")
    assert args == {"within_domain": True, "min_cluster_size": 25,
                    "n_components": 5, "n_neighbors": 7}
    assert pipeline.parse_discover_args("--no-within-domain")["within_domain"] is False
    assert pipeline.parse_discover_args(None) == pipeline.DEFAULT_DISCOVER_ARGS
    with pytest.raises(pipeline.BuildError, match="unrecognized"):
        pipeline.parse_discover_args("--nonsense 3")


def test_build_extras_are_lazy_with_an_actionable_error():
    from vllm_gwm.build.mining import _deps

    with pytest.raises(ModuleNotFoundError, match=r"vllm-gwm\[build\]"):
        _deps.lazy_import("umap_definitely_not_installed")
