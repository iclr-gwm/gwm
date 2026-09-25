# SPDX-License-Identifier: Apache-2.0
"""Env-override plumbing, per-request graph refs, and Mediator usage stats."""
import json
import types

from vllm_gwm.config import GwmConfig
from vllm_gwm.harness.harness import HarnessConfig, Mediator
from vllm_gwm.protocol import GwmRequestOptions
from test_harness_offline import FakeRender, FakeScorer, ScriptedLLM, flow_with_candidate


def test_judge_context_env_overrides(monkeypatch):
    monkeypatch.setenv("VLLM_GWM_SELECT_EXAMPLES", "2")
    monkeypatch.setenv("VLLM_GWM_SELECT_TAIL", "4000")
    monkeypatch.setenv("VLLM_GWM_SELECT_TOURNAMENT", "1")
    monkeypatch.setenv("VLLM_GWM_TOOL_DOCS_LIVE", "0")
    monkeypatch.setenv("VLLM_GWM_TOOL_DOCS_CHARS", "1500")
    monkeypatch.setenv("VLLM_GWM_LOG_CANDIDATES", "true")
    ov = GwmConfig.from_env().harness_overrides
    assert ov["select_examples"] == 2
    assert ov["select_tail_chars"] == 4000
    assert ov["select_tournament"] is True
    assert ov["tool_docs_live"] is False
    assert ov["tool_docs_chars"] == 1500
    assert ov["log_candidates"] is True


def test_xargs_graph_ref():
    opts = GwmRequestOptions.from_xargs(
        {"gwm.adapter": "toucan", "gwm.graph": "toucan_10p", "gwm.mode": "select"}
    )
    assert opts.adapter == "toucan"
    assert opts.graph == "toucan_10p"


def test_xargs_graph_ref_defaults_empty():
    opts = GwmRequestOptions.from_xargs({"gwm": {"adapter": "toucan"}})
    assert opts.graph == ""


def _mediator(cfg, reply):
    return Mediator(FakeScorer(), ScriptedLLM(reply), cfg, FakeRender(),
                    types.SimpleNamespace())


def test_stats_counts_skip_and_judged():
    med = _mediator(HarnessConfig(log_path=None), "{}")
    med.select_joint([flow_with_candidate("create_event", {"user": "alice"})] * 4,
                     "dom")
    st = med.stats_snapshot()
    assert st["events"] == 1
    assert st["by_kind"] == {"select_joint": 1}
    assert st["skips"] == {"identical_candidates": 1}
    assert st["judged"] == 0

    reply = json.dumps({"compare": "x", "best": 1, "scores": [0.2, 0.9]})
    med2 = _mediator(HarnessConfig(log_path=None, select_shuffle=False), reply)
    med2.select_joint(
        [flow_with_candidate("create_event", {"d": "1"}),
         flow_with_candidate("create_event", {"d": "2"})], "dom")
    st2 = med2.stats_snapshot()
    assert st2["events"] == 1
    assert st2["judged"] == 1
    assert st2["llm_calls"] >= 1
    assert st2["overrides"] == 1  # judge picked a non-first candidate
    assert st2["latency_ms_total"] >= 0


def test_stats_snapshot_is_copy():
    med = _mediator(HarnessConfig(log_path=None), "{}")
    snap = med.stats_snapshot()
    snap["events"] = 999
    assert med.stats_snapshot()["events"] == 0
