#!/usr/bin/env python3
"""End-to-end benchmark for vllm-gwm: advise, select, latency, sample outputs.

Runs fully offline with MockVLLM + FakeScorer (default) or with the real MiniLM
classifier when --model is passed.

Usage:
  scripts/build_env.sh
  .venv/bin/python scripts/e2e_benchmark.py
  .venv/bin/python scripts/e2e_benchmark.py --model
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/e2e_benchmark.py --model --json /tmp/gwm_e2e.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))
from mock_vllm import MockVLLM  # noqa: E402

CASES = [
    {"preset": "crm", "fixture": FIX / "crm_retry_flow.json", "graph": "tiny"},
    {"preset": "eops", "fixture": FIX / "eops_retry_flow.json", "graph": "tiny"},
]


def _load_flow(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


class FakeScorer:
    def __init__(self, transitions_path, centroids_dir, examples_path=None, **_kw):
        self.trans = json.loads(Path(transitions_path).read_text())
        self.examples = {}
        self.unify_domains = False
        self.mtr, self.mfc = 2000, 60000
        self.succ, self.base_succ, self.lift, self.trap, self.top_tool = {}, {}, {}, set(), {}
        for dom, dd in self.trans.items():
            self.base_succ[dom] = 1.0 - (dd.get("base_fail_rate", 0.0) or 0.0)
            base = dd.get("base_fail_rate", 0) or 1e-9
            for sf in dd.get("state_fail", []):
                s = str(sf["state"])
                self.succ[s] = 1.0 - float(sf["fail_rate"])
                if sf["n_rollouts"] >= 20 and (sf["fail_rate"] / base) >= 1.3:
                    self.trap.add(s)

    def base_for(self, domain):
        return self.base_succ.get(domain, 0.3)

    def resolve_domain(self, state, domain):
        return domain

    def classify(self, flow, domain):
        cand = [s for s in self.succ if s.split(":")[0] == domain] or list(self.succ)
        has_fail = any(
            isinstance(ev, dict)
            and ev.get("type") == "tool_result"
            and isinstance(ev.get("result"), dict)
            and (ev["result"].get("success") is False or ev["result"].get("error"))
            for ev in flow
        )
        trap_states = [s for s in cand if s in self.trap]
        state = trap_states[0] if has_fail and trap_states else sorted(cand)[-1]
        return state, 0.1

    def action_score(self, flow, domain):
        return (0.3, None)


def _tiny_preset():
    from vllm_gwm.presets import Preset

    tiny = FIX / "tiny_graph"
    return Preset(name="tiny", default_graph=str(tiny), default_domain="toy")


def run_benchmark(*, use_model: bool, json_out: Path | None) -> dict[str, Any]:
    from vllm_gwm.config import GwmConfig
    from vllm_gwm.runtime.engine_chat import HttpChatClient
    from vllm_gwm.runtime.registry import GraphRegistry
    from vllm_gwm.runtime.service import AdvisorService

    if use_model:
        pytest = __import__("pytest")
        pytest.importorskip("sentence_transformers")

    with MockVLLM("mock-gwm") as base_url:
        cfg = GwmConfig(preset_default="tiny", audit_log=False)
        reg = GraphRegistry({"tiny": FIX / "tiny_graph"}, max_loaded=2)
        if not use_model:
            from vllm_gwm.harness import graph_sidecar

            graph_sidecar.Scorer = FakeScorer  # type: ignore[misc, assignment]
        llm = HttpChatClient(base_url, "mock-gwm")
        svc = AdvisorService(cfg, reg, llm)
        svc._presets["tiny"] = _tiny_preset()

        report: dict[str, Any] = {"cases": [], "select_k4": None, "mode": "model" if use_model else "fake_scorer"}
        t_all = time.perf_counter()

        for case in CASES:
            flow = _load_flow(case["fixture"])
            ep = f"e2e-{case['preset']}-{uuid.uuid4().hex[:6]}"
            t0 = time.perf_counter()
            res = svc.advise(flow, preset="tiny", graph="tiny", episode_id=ep)
            ms = int((time.perf_counter() - t0) * 1000)
            harness = res.get("harness") or {}
            entry = {
                "preset": case["preset"],
                "episode_id": ep,
                "latency_ms": ms,
                "state": res.get("state"),
                "injected": res.get("injected"),
                "success_score": res.get("success_score"),
                "harness_llm_calls": harness.get("llm_calls"),
                "harness_tokens": harness.get("tokens"),
                "harness_latency_ms": harness.get("latency_ms"),
                "advice_preview": (res.get("block") or "")[:240],
            }
            report["cases"].append(entry)
            print(f"\n=== advise/{case['preset']} ===")
            print(f"latency_ms={ms} injected={res.get('injected')} state={res.get('state')}")
            print(f"harness: llm_calls={harness.get('llm_calls')} tokens={harness.get('tokens')} latency_ms={harness.get('latency_ms')}")
            if res.get("block"):
                print("advice block:")
                print(res["block"][:500])

        # select k=4
        flow = _load_flow(CASES[0]["fixture"])
        tools = [
            ("describe", {"table": "Account"}),
            ("execute", {"query": "SELECT 1"}),
            ("describe", {"table": "Case"}),
            ("respond", {"answer": "007"}),
        ]
        cands = [
            flow + [{"type": "ai_message", "content": "", "tool_calls": [{"name": t, "args": a}]}]
            for t, a in tools
        ]
        t0 = time.perf_counter()
        sel = svc.select(cands, preset="tiny", graph="tiny", mode="joint")
        ms = int((time.perf_counter() - t0) * 1000)
        results = sel.get("results") or []
        scores = [r.get("success_score") for r in results]
        winner = scores.index(max(scores)) if scores else -1
        report["select_k4"] = {
            "latency_ms": ms,
            "scores": scores,
            "winner_index": winner,
            "score_via": [r.get("score_via") for r in results],
        }
        print("\n=== select/k=4 (joint) ===")
        print(f"latency_ms={ms} scores={scores} winner={winner}")

        report["total_ms"] = int((time.perf_counter() - t_all) * 1000)
        print(f"\nTOTAL e2e_ms={report['total_ms']}")

        if json_out:
            json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f" wrote {json_out}")
        return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="use real MiniLM classifier")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    run_benchmark(use_model=args.model, json_out=args.json)


if __name__ == "__main__":
    main()
