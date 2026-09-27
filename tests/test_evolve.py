# SPDX-License-Identifier: MIT
import gzip
import json
import threading

import numpy as np
import pytest

from vllm_gwm.build import pipeline
from vllm_gwm.build.evolve import BuildJob, EvolveManager
from vllm_gwm.collect.outcome import OutcomeResolver
from vllm_gwm.collect.store import RolloutStore
from vllm_gwm.config import GwmConfig
from vllm_gwm.harness.llm import StubChat
from vllm_gwm.runtime.registry import GraphRegistry
from vllm_gwm.runtime.service import AdvisorService

FLOW = [
    {"type": "user_message", "content": "x"},
    {"type": "ai_message", "content": "y", "tool_calls": []},
]


def _manager(tmp_path, **cfg_kw):
    cfg = GwmConfig(data_root=tmp_path, **cfg_kw)
    store = RolloutStore(tmp_path / "rollouts")
    reg = GraphRegistry({"demo": tmp_path / "demo"})
    (tmp_path / "demo/reports").mkdir(parents=True, exist_ok=True)
    advisor = AdvisorService(cfg, reg, StubChat())
    outcome = OutcomeResolver(cfg, advisor)
    return EvolveManager(cfg, store, reg, outcome), store, reg


def test_evolve_trigger_after_n(tmp_path, monkeypatch):
    evolve, store, _ = _manager(tmp_path, evolve_every_n=2)
    # maybe_trigger spawns a daemon thread; stub the build out so this test covers
    # the trigger arithmetic only and no background build outlives it.
    ran = threading.Event()
    monkeypatch.setattr(evolve, "_run_build", lambda job: ran.set())
    store.append(FLOW, episode_id="a", success=True, finalize=True)
    assert evolve.maybe_trigger("demo") is None
    store.append(FLOW, episode_id="b", success=True, finalize=True)
    job = evolve.maybe_trigger("demo")
    assert job is not None
    assert job.adapter == "demo"
    assert ran.wait(5)


def test_force_trigger_ignores_every_n(tmp_path, monkeypatch):
    evolve, store, _ = _manager(tmp_path, evolve_every_n=0)
    ran = threading.Event()
    monkeypatch.setattr(evolve, "_run_build", lambda job: ran.set())
    store.append(FLOW, episode_id="a", success=True, finalize=True)
    assert evolve.maybe_trigger("demo") is None
    job = evolve.trigger("demo", force=True)
    assert job is not None
    assert job.adapter == "demo"
    assert ran.wait(5)


def _fake_bundle(work):
    """Minimal bundle a fake build_graph leaves behind (the registry validates it)."""
    (work / "reports").mkdir(parents=True, exist_ok=True)
    (work / "out/centroids").mkdir(parents=True, exist_ok=True)
    (work / "reports/transitions.json").write_text(json.dumps(
        {"unknown": {"base_fail_rate": 0.5, "state_fail": []}}))
    np.save(work / "out/centroids/centroids_16.npy", np.zeros((1, 16), dtype=np.float32))
    (work / "out/centroids/centroid_meta.json").write_text(json.dumps(
        {"state_ids": ["unknown:0"], "p90": {"unknown:0": 0.5}, "dim": 16}))


def test_build_runs_the_in_process_mining_pipeline(tmp_path, monkeypatch):
    evolve, store, reg = _manager(
        tmp_path, evolve_every_n=0, build_embed_device="cpu",
        build_embed_model="stub/tiny-embedder", build_watched_tools="execute")
    store.append(FLOW, episode_id="a", success=True, finalize=True)
    calls = []

    def fake_build(workdir, **kw):
        calls.append((workdir, kw))
        _fake_bundle(workdir)
        return {}

    monkeypatch.setattr(pipeline, "build_graph", fake_build)
    job = BuildJob(id="job1", status="running", adapter="demo", started_at="now")
    evolve._run_build(job)

    assert job.status == "completed", job.error
    workdir, kw = calls[0]
    assert kw == {"embed_device": "cpu", "watched_tools": "execute",
                  "embed_model": "stub/tiny-embedder"}
    # stage 0 bridge: the collected rollouts were normalized into the workdir
    with gzip.open(workdir / "out/rollouts.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert [r["uid"] for r in rows] == ["a"]
    assert rows[0]["conversation_flow"] == FLOW
    assert reg.registered["demo"] == workdir.resolve()   # activated


def test_build_failure_fails_closed(tmp_path, monkeypatch):
    evolve, store, reg = _manager(tmp_path, evolve_every_n=0)
    store.append(FLOW, episode_id="a", success=True, finalize=True)
    before = reg.registered["demo"]

    def boom(workdir, **kw):
        raise RuntimeError("clustering blew up")

    monkeypatch.setattr(pipeline, "build_graph", boom)
    job = BuildJob(id="job2", status="running", adapter="demo", started_at="now")
    evolve._run_build(job)

    assert job.status == "failed"
    assert "clustering blew up" in job.error
    assert reg.registered["demo"] == before             # previous graph untouched


def test_minimal_placeholder_stays_opt_in(tmp_path, monkeypatch):
    evolve, store, _ = _manager(tmp_path, evolve_every_n=0)
    store.append(FLOW, episode_id="a", success=True, finalize=True)

    def unavailable(workdir, **kw):
        raise ModuleNotFoundError("no umap; pip install vllm-gwm[build]")

    monkeypatch.setattr(pipeline, "build_graph", unavailable)
    monkeypatch.delenv("VLLM_GWM_EVOLVE_ALLOW_MINIMAL", raising=False)
    job = BuildJob(id="job3", status="running", adapter="demo", started_at="now")
    evolve._run_build(job)
    assert job.status == "failed"                       # never silently minimal

    monkeypatch.setenv("VLLM_GWM_EVOLVE_ALLOW_MINIMAL", "1")
    job = BuildJob(id="job4", status="running", adapter="demo", started_at="now")
    evolve._run_build(job)
    assert job.status == "completed", job.error


@pytest.mark.build
def test_evolve_build_end_to_end_with_stubs(tmp_path, stub_sentence_transformers,
                                            stub_labeller):
    """A real (stubbed-embedder) build through EvolveManager -> activated bundle."""
    evolve, store, reg = _manager(tmp_path, evolve_every_n=0,
                                  build_embed_device="cpu",
                                  build_embed_model="stub/tiny-embedder")
    flow = [
        {"type": "user_message", "content": "find acme"},
        {"type": "ai_message", "content": "", "tool_calls": [
            {"name": "execute", "args": {"q": "select 1"}}]},
        {"type": "tool_result", "tool_name": "execute",
         "result": {"success": True, "result": {"rows": [1]}}},
        {"type": "ai_message", "content": "done", "tool_calls": []},
    ]
    for i in range(4):
        store.append(flow, episode_id=f"ep{i}", success=i % 2 == 0, finalize=True)
    job = BuildJob(id="job5", status="running", adapter="demo", started_at="now")
    evolve._run_build(job)
    assert job.status == "completed", job.error
    built = tmp_path / "builds/demo/job5"
    assert (built / "reports/transitions.json").exists()
    assert (built / "out/centroids/centroids_16.npy").exists()
    assert reg.registered["demo"] == built.resolve()
