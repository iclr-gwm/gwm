#!/usr/bin/env python3
"""Regenerate the committed tiny synthetic graph fixture (tests/fixtures/tiny_graph).

A minimal but *real* graph dir the Scorer can load without any big artifact:
one domain ``toy`` with two states —

* ``toy:0`` — a KNOWN TRAP (fail_rate 0.9 over base 0.5 → lift 1.8 ≥ 1.3, n=25 ≥ 20),
  with tool_edges so ``next_actions`` renders (``execute`` low-success,
  ``describe`` higher-success);
* ``toy:1`` — a healthy state.

Centroids are real MiniLM embeddings of representative rendered flow tails, with
a generous p90 so live classification of the fixture flows never abstains. Run
this once (needs sentence-transformers + a one-time MiniLM download, possibly via
HTTPS_PROXY) and commit the outputs; tests then run with no network.

    python tests/fixtures/make_tiny_graph.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gwm._vendor import render
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
OUT = HERE / "tiny_graph"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Representative flows whose rendered tails define the two state centroids. The
# trap state toy:0 is the "stuck on a failing execute" shape; toy:1 is a healthy
# describe-then-respond shape.
TRAP_FLOW = [
    {"type": "system_message", "content": "You are a data agent."},
    {"type": "user_message", "content": "Find the owner of the Acme account."},
    {"type": "ai_message", "content": "", "tool_calls": [
        {"name": "execute", "args": {"query": "SELECT owner FROM Acct"}}]},
    {"type": "tool_result", "tool_name": "execute",
     "result": {"success": False, "error": "no such table: Acct"}},
]
HEALTHY_FLOW = [
    {"type": "system_message", "content": "You are a data agent."},
    {"type": "user_message", "content": "Find the owner of the Acme account."},
    {"type": "ai_message", "content": "", "tool_calls": [
        {"name": "describe", "args": {"table": "Account"}}]},
    {"type": "tool_result", "tool_name": "describe",
     "result": {"success": True, "result": {"columns": ["Id", "Name", "OwnerId"]}}},
    {"type": "ai_message", "content": "The owner is user 007.", "tool_calls": []},
]

TRANSITIONS = {
    "toy": {
        "base_fail_rate": 0.5,
        "n_rollouts": 50,
        "state_fail": [
            {"state": "toy:0", "fail_rate": 0.9, "n_rollouts": 25},
            {"state": "toy:1", "fail_rate": 0.2, "n_rollouts": 25},
        ],
        "P_next": {
            "toy:0": {"toy:0": 0.6, "toy:1": 0.4},
            "toy:1": {"toy:1": 1.0},
        },
        "tool_edges": {
            "toy:0": {
                "execute": {"n": 12, "succ": 0.1, "p_action": 0.7,
                            "next": {"toy:0": 0.8, "toy:1": 0.2}},
                "describe": {"n": 8, "succ": 0.7, "p_action": 0.3,
                             "next": {"toy:1": 0.9, "toy:0": 0.1}},
            },
            "toy:1": {
                "respond": {"n": 20, "succ": 0.9, "p_action": 1.0,
                            "next": {"toy:1": 1.0}},
            },
        },
    }
}

META = {
    "state_ids": ["toy:0", "toy:1"],
    "p90": {"toy:0": 0.9, "toy:1": 0.9},   # generous → fixture flows never abstain
    "top_tool": {"toy:0": "execute", "toy:1": "respond"},
    "model": MODEL,
    "tail_chars": 3000,
    "max_tool_result_chars": 2000,
    "max_flow_chars": 60000,
}


def main() -> None:
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "out" / "centroids").mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL, device="cpu")
    tails = [
        render.render_text(TRAP_FLOW, META["max_tool_result_chars"], META["max_flow_chars"])[-META["tail_chars"]:],
        render.render_text(HEALTHY_FLOW, META["max_tool_result_chars"], META["max_flow_chars"])[-META["tail_chars"]:],
    ]
    cents = model.encode(tails, normalize_embeddings=True).astype("float32")
    assert cents.shape == (2, 384), cents.shape

    (OUT / "reports" / "transitions.json").write_text(json.dumps(TRANSITIONS, indent=2))
    np.save(OUT / "out" / "centroids" / "centroids_384.npy", cents)
    (OUT / "out" / "centroids" / "centroid_meta.json").write_text(json.dumps(META, indent=2))
    (OUT / "MANIFEST.json").write_text(json.dumps(
        {"name": "tiny_graph", "domains": ["toy"], "n_states": 2, "n_traps": 1,
         "synthetic": True, "note": "test fixture; regenerate with make_tiny_graph.py"},
        indent=2))
    print(f"wrote tiny graph to {OUT}")


if __name__ == "__main__":
    main()
