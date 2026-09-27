# SPDX-License-Identifier: MIT
"""Offline unit tests for Mediator.select_joint."""

import json
import types

from vllm_gwm.harness.harness import HarnessConfig, Mediator


class FakeScorer:
    trans = {}
    succ = {}
    lift = {}
    trap = set()
    top_tool = {}
    examples = {}
    mtr = 2000
    mfc = 60000

    def __init__(self):
        self._nn = types.SimpleNamespace(state="dom:7", thr=0.2)

    def classify(self, flow, domain):
        return None, 0.35

    def resolve_domain(self, state, domain):
        return domain

    def base_for(self, domain):
        return 0.4

    def action_score(self, flow, domain):
        return (0.4, None)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        return self.reply, {"prompt_tokens": 10, "completion_tokens": 10}


class FakeRender:
    @staticmethod
    def render_text(flow, mtr, mfc):
        return "SYSTEM: sys\nUSER: the task\n"


def flow_with_candidate(tool, args, content=""):
    return [
        {"type": "user_message", "content": "Create a meeting with alice."},
        {"type": "ai_message", "content": content, "tool_calls": [{"name": tool, "args": args}]},
    ]


def run(cfg, reply, flows):
    llm = ScriptedLLM(reply)
    med = Mediator(FakeScorer(), llm, cfg, FakeRender(), types.SimpleNamespace())
    return med.select_joint(flows, "dom"), llm


def test_identical_candidates_short_circuit():
    cfg = HarnessConfig(log_path=None, log_candidates=True)
    flows = [flow_with_candidate("create_event", {"user": "alice"})] * 4
    res, llm = run(cfg, "{}", flows)
    assert len(llm.prompts) == 0
    assert [r["success_score"] for r in res] == [0.5] * 4


def test_dedup_shuffle_best_mapping():
    cfg = HarnessConfig(log_path=None, log_candidates=True)
    flows = [
        flow_with_candidate("create_event", {"user": "alice", "date": "2026-07-30"}),
        flow_with_candidate("create_event", {"user": "alice", "date": "2026-07-31"}),
        flow_with_candidate("create_event", {"user": "alice", "date": "2026-07-30"}),
        flow_with_candidate("create_event", {"user": "alice", "date": "2026-07-31"}),
    ]
    reply = json.dumps({"compare": "dates differ", "best": 0, "scores": [0.9, 0.9]})
    res, llm = run(cfg, reply, flows)
    assert len(llm.prompts) == 1
    scores = [r["success_score"] for r in res]
    assert scores.count(max(scores)) == 1
