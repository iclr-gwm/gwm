#!/usr/bin/env python3
"""CPU sidecar: transition-graph world-model guidance for prompt injection.

No GPU, no torch-in-purple-venv: runs under workflow_mining/.venv and serves the
orchestrator over HTTP. Classifies the live conversation_flow to a discovered
state (tail-embed → cosine-NN to the graph's persisted centroids, abstain past
the train P90), looks up the mined transition graph, and returns the injection
block for the requested arm template.

POST /score  {conversations:[flow,...], domain, template}
  -> {results:[{state, cos_dist, abstain, is_trap, last_tool, injected, block}, ...]}
GET /health

Templates: graph_full | generic | trap_only | placebo  (see render_block / arms).

Run::
    .venv/bin/python graph_sidecar.py --port 8031 \
        --transitions reports/transitions.json --centroids out/centroids
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# VENDORED into the gwm repo: the sibling deps live next to this file, so the
# original sys.path hack (parents[4]/wm/lib) is replaced by package imports.
# See docs/SYNC.md for provenance.
from vllm_gwm.harness import gv_score, render, wm_prompt

HERE = Path(__file__).resolve().parent

GENERIC_BLOCK = (
    "[World-Model guidance — general best practice]\n"
    "Review the most recent tool result before your next action.\n"
    "Do not repeat a tool call that just failed with the same arguments — diagnose and change approach.\n"
    "Do not finish until you have verified every requested change actually took effect.\n"
    "Prefer making concrete progress toward completing all requested steps."
)


class Scorer:
    def __init__(self, transitions_path, centroids_dir, seed=17, examples_path=None,
                 max_example_chars=1200, max_block_chars=12000,
                 unify_domains=False, p90_scale=1.0, log_requests=False):
        from sentence_transformers import SentenceTransformer
        self.trans = json.loads(Path(transitions_path).read_text())
        # per-domain few-shot examples for grounding (states + transitions); optional.
        self.examples = {}
        if examples_path and Path(examples_path).exists():
            self.examples = json.loads(Path(examples_path).read_text())
        self.max_example_chars = int(max_example_chars)
        self.max_block_chars = int(max_block_chars)
        # serve-time domain unification (out-of-domain graphs): classify against ALL
        # centroid rows and resolve guidance from the matched state's home domain.
        self.unify_domains = bool(unify_domains)
        # abstain threshold relaxation: abstain iff dist > p90[s] * p90_scale
        self.p90_scale = float(p90_scale)
        self.log_requests = bool(log_requests)
        # nearest-neighbour details of the last classify() on this thread, so
        # abstained requests still log which state was closest and its threshold
        self._nn = threading.local()
        cdir = Path(centroids_dir)
        cent_files = sorted(cdir.glob("centroids_*.npy"))
        if not cent_files:
            raise FileNotFoundError(f"no centroids_*.npy under {cdir}")
        self.cents = np.load(cent_files[0])
        meta = json.loads((cdir / "centroid_meta.json").read_text())
        self.ids = meta["state_ids"]
        self.p90 = meta["p90"]
        self.top_tool = meta.get("top_tool", {})
        self.model_name = meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.tail = meta.get("tail_chars", 3000)
        self.mtr = meta.get("max_tool_result_chars", 2000)
        self.mfc = meta.get("max_flow_chars", 60000)
        self.dom_of = {s: s.split(":")[0] for s in self.ids}
        # per-state historical success rate (1 - fail_rate) and per-domain base, for best_of_n scoring
        self.succ = {}
        self.base_succ = {}
        for dom, dd in self.trans.items():
            self.base_succ[dom] = 1.0 - (dd.get("base_fail_rate", 0.0) or 0.0)
            for sf in dd.get("state_fail", []):
                self.succ[str(sf["state"])] = 1.0 - float(sf["fail_rate"])
        # pooled (n_rollouts-weighted) base success, used when the request domain
        # is not a graph domain (out-of-domain requests under --unify-domains).
        tot_w = tot = 0.0
        for dom, dd in self.trans.items():
            w = float(dd.get("n_rollouts")
                      or sum(sf.get("n_rollouts", 0) for sf in dd.get("state_fail", []))
                      or 1.0)
            tot_w += w
            tot += w * self.base_succ[dom]
        self.pooled_succ = (tot / tot_w) if tot_w else 0.3
        self.by_dom = defaultdict(list)
        for i, s in enumerate(self.ids):
            self.by_dom[self.dom_of[s]].append(i)
        # trap states: fail-rate lift >= 1.3 over domain base, n>=20
        self.trap = set()
        # per-state fail-lift (fail_rate / domain base_fail_rate); 1.0 == base. Used by
        # the graph_value scorer's trap penalty on the modal landing state.
        self.lift = {}
        for dom, dd in self.trans.items():
            base = dd.get("base_fail_rate", 0) or 1e-9
            for sf in dd.get("state_fail", []):
                lift = sf["fail_rate"] / base
                self.lift[str(sf["state"])] = lift
                if sf["n_rollouts"] >= 20 and lift >= 1.3:
                    self.trap.add(str(sf["state"]))
        # graph_value composite-scorer weights (env-overridable WM_GV_*).
        self.gv_weights = gv_score.weights_from_env()
        # placebo permutation: within-domain shuffle of state ids
        rng = random.Random(seed)
        self.perm = {}
        for dom, idxs in self.by_dom.items():
            states = [self.ids[i] for i in idxs]
            shuf = states[:]; rng.shuffle(shuf)
            self.perm.update(dict(zip(states, shuf)))
        # The embedder is whatever built the graph (recorded in centroid_meta),
        # so a graph mined with a different model classifies against that same
        # model. Large embedders are unusable on CPU at per-step latency, hence
        # the device override; remote code is opt-in per usual HF caution.
        self.embed_device = os.getenv("VLLM_GWM_EMBED_DEVICE", "cpu").strip() or "cpu"
        trust = os.getenv("VLLM_GWM_EMBED_TRUST_REMOTE_CODE", "").strip().lower() in (
            "1", "true", "yes", "on")
        kwargs = {"device": self.embed_device}
        if trust:
            kwargs["trust_remote_code"] = True
        self.model = SentenceTransformer(self.model_name, **kwargs)
        dim = getattr(self.model, "get_sentence_embedding_dimension", lambda: None)()
        if dim is not None and int(dim) != int(self.cents.shape[1]):
            raise ValueError(
                f"embedder {self.model_name!r} emits {dim}-d vectors but the graph "
                f"centroids are {self.cents.shape[1]}-d ({cent_files[0].name}). The "
                "graph must be rebuilt with the embedder named in centroid_meta.json."
            )

    def base_for(self, domain):
        """Domain base success; pooled average for domains absent from the graph."""
        return self.base_succ.get(domain, self.pooled_succ)

    def resolve_domain(self, state, domain):
        """Domain whose transition block describes `state` (its home domain when
        unified / when the request domain is not a graph domain)."""
        if state is not None and (self.unify_domains or domain not in self.trans):
            return self.dom_of.get(state, domain)
        return domain

    def classify(self, flow, domain):
        cand = self.by_dom.get(domain, [])
        if self.unify_domains:
            # rank against ALL centroid rows (serve-time domain unification; also
            # covers request domains with no candidates of their own)
            cand = list(range(len(self.ids)))
        if not cand:
            return None, 1.0
        text = render.render_text(flow, self.mtr, self.mfc)[-self.tail:]
        v = self.model.encode([text], normalize_embeddings=True)[0]
        sims = self.cents[cand] @ v
        j = int(np.argmax(sims)); s = self.ids[cand[j]]; dist = 1.0 - float(sims[j])
        self._nn.state = s
        self._nn.thr = self.p90.get(s, 1.0) * self.p90_scale
        # top-N neighborhood for soft-classification consumers (wm-harness
        # WMH_TOPN_STATES): [(state_id, cos_dist), ...] best-first. Additive —
        # existing callers keep using the argmax + abstain contract unchanged.
        order = np.argsort(-sims)[:5]
        self._nn.top = [(self.ids[cand[int(i)]], 1.0 - float(sims[int(i)]))
                        for i in order]
        if dist > self._nn.thr:
            return None, dist           # abstain
        return s, dist

    def block_for(self, domain, state, template, n_shot=0):
        dd = self.trans.get(domain, {})
        if template == "generic":
            return GENERIC_BLOCK
        ex = self.examples.get(domain)  # examples are keyed in the centroid state space
        kw = dict(examples=ex, n_shot=n_shot,
                  max_example_chars=self.max_example_chars,
                  max_block_chars=self.max_block_chars)
        if template == "placebo":
            # placebo permutes the state id; do NOT attach (real) examples — keep it a control.
            return wm_prompt.render_block(dd, self.perm.get(state, state), self.top_tool)
        if template == "graph_tools":
            return wm_prompt.render_block_tools(dd, state, self.top_tool, **kw)
        if template == "graph_value":
            return wm_prompt.render_block_value(dd, state, self.top_tool, **kw)
        if template == "graph_action":
            # minimal actionable block (may be "" -> no injection); example
            # grounding defaults ON since the whole point is one concrete step.
            return wm_prompt.render_block_action(
                dd, state, self.top_tool, examples=ex,
                n_shot=max(1, n_shot), max_example_chars=self.max_example_chars)
        if template == "trap_only" and state not in self.trap:
            return ""
        return wm_prompt.render_block(dd, state, self.top_tool, **kw)  # graph_full / trap_only(trap)

    def action_score(self, flow, domain):
        """Score a best-of-N candidate by the ADVANTAGE of the tool action it proposes,
        from the CURRENT state. `flow` is the live flow with the candidate ai_message
        appended (the wm_react best_of_n shape). We classify the pre-candidate flow to
        get the current state s, read the candidate's toolset, and return succ(action)
        from the mined tool_edges (ranking candidates from the same s by succ == ranking
        by advantage, since V(s) is constant across candidates). Falls back to the
        landing-state value for unseen toolsets / abstained current state."""
        base = self.base_for(domain)
        cand_tools, pre = [], flow
        if flow and isinstance(flow[-1], dict) and flow[-1].get("type") == "ai_message":
            cand_tools = sorted({tc.get("name") for tc in (flow[-1].get("tool_calls") or [])
                                 if tc.get("name")})
            pre = flow[:-1]
        s, dist = self.classify(pre, domain)
        tk = "|".join(cand_tools)
        te = self.trans.get(self.resolve_domain(s, domain), {}).get("tool_edges", {})
        if s is not None:
            a = te.get(s, {}).get(tk)
            if a is not None and a.get("n", 0) >= 3:
                return float(a["succ"]), s, dist, tk, "action"
        # fallback: value of the state the full (flow+candidate) lands in
        s2, _ = self.classify(flow, domain)
        return float(self.succ.get(s2, base)), (s or "UNKNOWN"), dist, tk, "landing"

    def graph_value_score(self, flow, domain):
        """Score a best-of-N candidate with the composite graph-value advantage
        (gv_score.graph_value_step): EB-shrunk action success + one-step lookahead
        advantage - trap penalty - frequency penalty. Same classify->edge-lookup path
        as action_score; falls back to landing-state value for unseen edges / abstain."""
        base = self.base_for(domain)
        cand_tools, pre = [], flow
        if flow and isinstance(flow[-1], dict) and flow[-1].get("type") == "ai_message":
            cand_tools = sorted({tc.get("name") for tc in (flow[-1].get("tool_calls") or [])
                                 if tc.get("name")})
            pre = flow[:-1]
        s, dist = self.classify(pre, domain)
        tk = "|".join(cand_tools)
        te = self.trans.get(self.resolve_domain(s, domain), {}).get("tool_edges", {})
        if s is not None:
            edge = te.get(s, {}).get(tk)
            if edge is not None:
                r = gv_score.graph_value_step(edge, s, self.succ, base,
                                              self.lift, self.trap, self.gv_weights)
                return float(r["score"]), s, dist, tk, "graph_value", r.get("s_land", s)
        # fallback: value of the state the full (flow+candidate) lands in
        s2, _ = self.classify(flow, domain)
        return float(self.succ.get(s2, base)), (s or "UNKNOWN"), dist, tk, "landing", (s2 or "UNKNOWN")

    def score_one(self, flow, domain, template, n_shot=0, select_mode="state"):
        if select_mode == "action":
            ss, state, dist, tk, how = self.action_score(flow, domain)
            return {"state": state, "cos_dist": dist, "abstain": state == "UNKNOWN",
                    "is_trap": state in self.trap, "last_tool": tk, "injected": False,
                    "block": "", "success_score": ss, "probability": 1.0, "score_via": how}
        if select_mode == "graph_value":
            ss, state, dist, tk, how, s_land = self.graph_value_score(flow, domain)
            return {"state": state, "cos_dist": dist, "abstain": state == "UNKNOWN",
                    "is_trap": s_land in self.trap, "last_tool": tk, "injected": False,
                    "block": "", "success_score": ss, "probability": 1.0, "score_via": how}
        return self._score_one_state(flow, domain, template, n_shot)

    def _score_one_state(self, flow, domain, template, n_shot=0):
        state, dist = self.classify(flow, domain)
        last_tool = ""
        for e in reversed(flow):
            if isinstance(e, dict) and e.get("type") == "ai_message" and e.get("tool_calls"):
                last_tool = (e["tool_calls"][0].get("name") or "")
                break
        base = self.base_for(domain)
        if state is None:
            nn = getattr(self._nn, "state", None)
            self._log_request(domain, f"UNKNOWN(nearest={nn})", "UNKNOWN", dist,
                              getattr(self._nn, "thr", None), False)
            return {"state": "UNKNOWN", "cos_dist": dist, "abstain": True,
                    "is_trap": False, "last_tool": last_tool, "injected": False, "block": "",
                    "success_score": base, "probability": 1.0}
        dom = self.resolve_domain(state, domain)
        block = self.block_for(dom, state, template, n_shot)
        self._log_request(domain, state, dom, dist,
                          self.p90.get(state, 1.0) * self.p90_scale, bool(block))
        return {"state": state, "cos_dist": dist, "abstain": False,
                "is_trap": state in self.trap, "last_tool": last_tool,
                "injected": bool(block), "block": block,
                "success_score": self.succ.get(state, base), "probability": 1.0}

    def _log_request(self, domain_in, state, resolved_domain, dist, p90_thr, injected):
        if not self.log_requests:
            return
        thr = "nan" if p90_thr is None else f"{p90_thr:.4f}"
        print(f"[score] domain_in={domain_in} resolved_state={state} "
              f"resolved_domain={resolved_domain} dist={dist:.4f} "
              f"p90_threshold={thr} injected={injected}", file=sys.stderr, flush=True)


def make_handler(scorer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/health", "/healthz"):
                n_ex = sum(1 for d in scorer.examples.values()
                           for v in (d.get("states") or {}).values() if v)
                self._send(200, {"status": "ok", "n_states": len(scorer.ids),
                                 "n_traps": len(scorer.trap), "model": scorer.model_name,
                                 "n_states_with_examples": n_ex})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/score":
                self._send(404, {"error": "not found"}); return
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode()) if n else {}
            convs = req.get("conversations") or []
            domain = req.get("domain") or "unknown"
            template = (req.get("template") or "graph_full").strip()
            n_shot = int(req.get("n_shot") or 0)
            select_mode = (req.get("select_mode") or "state").strip()
            results = [scorer.score_one(c, domain, template, n_shot, select_mode) for c in convs]
            self._send(200, {"results": results})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--transitions", default=str(HERE / "reports" / "transitions.json"))
    ap.add_argument("--centroids", default=str(HERE / "out" / "centroids"))
    ap.add_argument("--examples", default=str(HERE / "reports" / "examples.json"),
                    help="few-shot grounding examples (per-domain states/transitions)")
    ap.add_argument("--max-example-chars", type=int,
                    default=int(os.getenv("WM_EXAMPLE_MAX_CHARS", "1200")))
    ap.add_argument("--max-block-chars", type=int,
                    default=int(os.getenv("WM_BLOCK_MAX_CHARS", "12000")))
    ap.add_argument("--unify-domains", action="store_true",
                    default=os.getenv("WM_UNIFY_DOMAINS", "").lower() in ("1", "true", "yes", "on"),
                    help="classify against ALL states (ignore the request domain) and "
                         "resolve guidance from the matched state's home domain — for "
                         "serving an out-of-domain graph")
    ap.add_argument("--p90-scale", type=float,
                    default=float(os.getenv("WM_P90_SCALE", "1.0")),
                    help="relax the abstain threshold: abstain iff dist > p90*scale")
    ap.add_argument("--no-request-log", action="store_true",
                    help="disable the per-/score stderr log line")
    args = ap.parse_args()
    scorer = Scorer(args.transitions, args.centroids, examples_path=args.examples,
                    max_example_chars=args.max_example_chars,
                    max_block_chars=args.max_block_chars,
                    unify_domains=args.unify_domains, p90_scale=args.p90_scale,
                    log_requests=not args.no_request_log)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(scorer))
    n_ex = sum(1 for d in scorer.examples.values() for v in (d.get('states') or {}).values() if v)
    print(f"[graph_sidecar] ready on :{args.port}  states={len(scorer.ids)} "
          f"traps={len(scorer.trap)} ex_states={n_ex} "
          f"unify_domains={scorer.unify_domains} p90_scale={scorer.p90_scale}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
