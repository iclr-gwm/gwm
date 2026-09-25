"""Graph-guided harness mediator (H1): an LLM agent over the mined workflow graph.

Sits between the workflow-graph artifacts (states / transitions / examples, built
by ``wm/build_all.sh``) and a policy agent, behind the existing graph-sidecar HTTP
contract (see server.py). Per request it:

1. classifies the live ``conversation_flow`` to a discovered state (reusing the
   deterministic ``graph_sidecar.Scorer`` classifier: MiniLM tail-embed → cosine-NN
   → p90 abstain);
2. detects live retry-loops directly from the trajectory (instance-level signal,
   independent of the classifier);
3. decides whether to intervene at all (default trigger: known trap state OR live
   retry-loop; hard per-episode budget; never twice for the same state);
4. if triggered, runs a small tool-loop with the harness LLM (find_similar /
   success_example / interpret_state over the graph) and returns its final short
   guidance as the injectable ``block`` — or ``""`` for NO ADVICE.

It also scores best-of-K candidates (``select``): jointly for per-step candidates
(EnterpriseOps wm_react shape) or singly for whole-rollout candidates (CRM shape).

Every decision is appended to a JSONL log for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"


def _fill(template: str, **vars: Any) -> str:
    """Placeholder substitution by literal replace — prompt files may contain
    JSON braces, so ``str.format`` is unusable here."""
    for k, v in vars.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def load_prompt(name: str, **vars: Any) -> str:
    return _fill((PROMPTS / f"{name}.md").read_text(encoding="utf-8"), **vars)


def parse_json_reply(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from an LLM reply (tolerates
    <think> traces, prose, and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def salvage_advice(text: str, cap: int) -> str:
    """Recover advice from a reply that failed strict JSON parsing.

    The advise LLM sometimes emits its guidance as bare prose (no JSON wrapper)
    or as JSON whose ``advice`` string contains unescaped quotes — both are good
    guidance that must not be dropped. Try, in order: (1) a lenient
    ``"advice": "..."`` field extraction; (2) if the reply is plain prose (no
    ``"tool"`` intent, doesn't start with a JSON object), use the cleaned prose
    verbatim. Returns "" when the reply looks like a tool call or is empty.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    if not text:
        return ""
    m = re.search(r'"advice"\s*:\s*"(.+?)"\s*[}\n]', text, flags=re.S)
    if m:
        return m.group(1).strip()[:cap]
    if '"tool"' in text or text.lstrip().startswith("{"):
        return ""  # a (malformed) tool-call attempt — do not treat as advice
    return text[:cap]


# Advice content that DIAGNOSIS §2.3 showed was actively harmful. Rejected advice
# is dropped (NO ADVICE) rather than injected.
_BAD_ADVICE_PATTERNS = [
    # hallucinated affordances / non-existent tool syntax (§2.3: 9.6%)
    (re.compile(r"information_schema", re.I), "information_schema"),
    (re.compile(r"\blist_tables\b", re.I), "list_tables"),
    (re.compile(r"\bPRAGMA\b", re.I), "pragma"),
    (re.compile(r'describe\s*\(\s*[{\"]', re.I), "json_pseudo_describe"),
    (re.compile(r'"table"\s*:', re.I), "json_pseudo_syntax"),
    # false "schema migration" narrative (§2.3: 22.5%) — the DB schema is
    # unchanged; drift only rewrites prompt text. Do not send agents chasing it.
    (re.compile(r"schema\s+drift|database\s+migrat|migration\s+in\s+progress"
                r"|(?:table|column|field)s?\s+(?:were|was|has\s+been|have\s+been)"
                r"\s+renamed|renamed\s+to", re.I), "schema_migration_theory"),
    # "no intervention needed" replies leaked verbatim as guidance (§2.3)
    (re.compile(r"no\s+(?:intervention|advice|guidance)\s+(?:needed|required|"
                r"necessary)|no\s+action\s+needed", re.I), "no_intervention_leak"),
]


def validate_advice(advice: str, *, salvaged: bool) -> str | None:
    """Return a rejection reason if ``advice`` is one of the known-harmful shapes
    (DIAGNOSIS §2.3, §7.3), else None. Empty advice is fine (NO ADVICE)."""
    a = (advice or "").strip()
    if not a:
        return None
    for pat, name in _BAD_ADVICE_PATTERNS:
        if pat.search(a):
            return name
    # truncated salvage: bare-prose recovery cut mid-token by the char cap.
    if salvaged and a[-1:].isalnum() and len(a.split()[-1]) >= 12:
        return "truncated_salvage"
    return None


# --------------------------------------------------------------------------- flow


def _actions_from_flow(flow: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(toolset, ok, result_digest) per executed action, in order."""
    out: list[dict[str, Any]] = []
    pending_tools: str | None = None
    for ev in flow:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "ai_message":
            tools = sorted({tc.get("name") for tc in (ev.get("tool_calls") or [])
                            if tc.get("name")})
            pending_tools = "|".join(tools) if tools else None
        elif ev.get("type") == "tool_result":
            res = ev.get("result")
            ok = True
            if isinstance(res, dict) and "success" in res:
                ok = bool(res.get("success"))
            txt = json.dumps(res, default=str)[:600] if res is not None else ""
            if ok and re.search(r'"(error|failed)"\s*:\s*(?!null|false|""|\[\]|\{\})', txt):
                ok = False
            out.append({"tools": pending_tools or ev.get("tool_name") or "",
                        "ok": ok,
                        "digest": hashlib.md5(txt.encode()).hexdigest()[:10]})
    return out


def detect_retry_loop(flow: list[dict[str, Any]]) -> dict[str, Any]:
    """Live, classifier-independent loop detection on the trajectory tail.

    Sparse by design (DIAGNOSIS §2.1 / §7.2): the old ">=2 consecutive failures"
    rule fired on ~92 % of episodes because ordinary schema probing under drift
    produces two *different* failures. A genuine stuck loop is the agent
    repeating the same call and getting the *same* result. So we now fire only
    when the tail shows >=3 consecutive failing calls of the same toolset whose
    result is unchanged (identical digest at least 3x), or >=3 same-toolset
    calls in the last 4 with an identical result at least 3x (a hard stall).
    """
    acts = _actions_from_flow(flow)
    if not acts:
        return {"detected": False}
    tail_tool = acts[-1]["tools"]
    fails = 0
    fail_digests: list[str] = []
    for a in reversed(acts):
        if a["tools"] == tail_tool and not a["ok"]:
            fails += 1
            fail_digests.append(a["digest"])
        else:
            break
    if fails >= 3 and fail_digests and max(
            fail_digests.count(d) for d in set(fail_digests)) >= 3:
        return {"detected": True, "kind": "consecutive_failures",
                "tool": tail_tool, "count": fails}
    last4 = acts[-4:]
    same = [a for a in last4 if a["tools"] == tail_tool]
    if len(same) >= 3:
        digests = [a["digest"] for a in same]
        if max(digests.count(d) for d in set(digests)) >= 3:
            return {"detected": True, "kind": "stall_repeat",
                    "tool": tail_tool, "count": len(same)}
    return {"detected": False}


def episode_key(flow: list[dict[str, Any]]) -> str:
    """Stable per-episode key: an explicit ``task_metadata.episode`` hint when
    present (used by replay to keep sibling rollouts of one task distinct),
    else a hash of the first user_message (the task prompt)."""
    for ev in flow:
        if isinstance(ev, dict) and ev.get("type") == "task_metadata" and ev.get("episode"):
            return str(ev["episode"])[:40]
    for ev in flow:
        if isinstance(ev, dict) and ev.get("type") == "user_message":
            return hashlib.md5(str(ev.get("content", "")).encode()).hexdigest()[:16]
    return hashlib.md5(json.dumps(flow[:2], default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------ config


@dataclass
class HarnessConfig:
    trigger: str = "trap_or_loop"      # trap_or_loop | confident | always
    max_tool_calls: int = 3
    max_advice_chars: int = 700
    max_per_episode: int = 2
    max_flow_tail_chars: int = 2500
    max_example_chars: int = 900
    llm_retries: int = 1
    log_path: str | None = None
    # -- judge context & selection protocol (EOPS_HARNESS_DIAGNOSIS.md fixes) --
    task_goal: bool = True             # F1: show the task excerpt in every prompt
    max_task_chars: int = 1000         # F1: task-goal excerpt cap
    cand_args_chars: int = 700         # F2: per-candidate tool-args render budget
    cand_content_chars: int = 400      # F2: per-candidate content render budget
    select_dedup: bool = True          # F2: judge only distinct candidates
    select_sem_dedup: bool = False     # dedup on the tool call (name+args) only,
    #                                    ignoring free-text / JSON-whitespace so
    #                                    functionally identical actions collapse
    select_ranking: bool = True        # F3: comparative prompt, explicit best index
    select_shuffle: bool = True        # F4: shuffle judge order (position bias)
    nearest_on_abstain: bool = True    # F5: low-confidence nearest-state context
    action_space: str = ""             # F6: benchmark action-space description
    advice_rules: str = ""             # F6: benchmark-specific advice rules
    log_candidates: bool = False       # F7: log rendered candidates to the jsonl
    # -- PERF_BOOST.md fixes (F8-F10): richer judge context ------------------
    tool_docs: str = ""                # F8: static tool-catalog text (env fallback)
    tool_docs_chars: int = 2000        # F8: cap for the rendered tool catalog
    tool_docs_live: bool = True        # F8: prefer the flow's live `tools` event
    select_examples: int = 0           # F9: successful exemplars in select prompts
    select_neg_examples: int = 0       # H4: contrastive step-labeled negatives
    #                                    (obs_outcome=error at the same state) shown
    #                                    to the judge as "failed attempts here"
    select_tail_chars: int = 0         # F10: select-only tail budget (0 = advise's)
    # -- PERF_BOOST2.md fixes (J/G levers) -----------------------------------
    select_tournament: bool = False    # J3: pairwise single-elim bracket (>2 cands)
    select_votes: int = 1              # J2: judge votes (reshuffled), majority best
    select_min_margin: float = 0.2     # H2: min winner−cand0 score margin to
    #                                    override candidate 0 (the greedy anchor);
    #                                    below it, re-pin cand0 (TOUCAN E4)
    select_min_margin_abstain: float = -1.0  # H23: separate (larger) margin used
    #                                    on abstain/UNKNOWN steps where judge
    #                                    evidence is thin; <0 = reuse select_min_margin
    select_graph_tiebreak: bool = False      # H25: when the margin gate fires,
    #                                    instead of always keeping cand0, prefer the
    #                                    candidate whose tool matches the classified
    #                                    state's highest-succ graph edge
    select_examples_rank: bool = False       # H26: order state exemplars by stored
    #                                    prototypicality (cos) instead of mining order
    topn_states: int = 1               # G3: soft top-N state context on abstain
    edge_min_n: int = 3                # G4: min edge support in next_actions
    rollout_tail_chars: int = 1200     # J1: per-rollout tail in joint rollout mode
    rollout_answer_chars: int = 400    # J1: per-rollout final-answer render cap
    extra: dict[str, Any] = field(default_factory=dict)


# F6 fallback when the benchmark config exports no WMH_ACTION_SPACE.
GENERIC_ACTION_SPACE = (
    "The policy agent acts by calling tools from the catalog visible in its own "
    "context, or finishes by replying without a tool call. Any advice MUST only "
    "reference tools or identifiers that appear in the trajectory.")


class Mediator:
    """The harness agent. ``scorer`` is a ``graph_sidecar.Scorer`` (owns the
    classifier + graph + examples); ``llm`` is a ``llm.ChatClient``-like object."""

    BLOCK_PREFIX = "[World-Model guidance]\n"

    def __init__(self, scorer: Any, llm: Any, cfg: HarnessConfig, render_mod: Any,
                 wm_prompt_mod: Any):
        self.scorer = scorer
        self.llm = llm
        self.cfg = cfg
        self.render = render_mod
        self.wm_prompt = wm_prompt_mod
        self._lock = threading.Lock()
        self._episode_counts: OrderedDict[str, int] = OrderedDict()
        self._episode_states: OrderedDict[str, set] = OrderedDict()
        self._log_lock = threading.Lock()
        # cumulative usage counters, aggregated by the /v1/gwm/stats route
        self._stats_lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "events": 0, "by_kind": {}, "skips": {}, "unknown_state": 0,
            "judged": 0, "llm_calls": 0, "tokens": 0, "overrides": 0,
            "fallbacks": 0, "injected": 0, "latency_ms_total": 0}
        # serializes classify + the nearest-state read (scorer._nn is shared
        # mutable state; ThreadingHTTPServer handles requests concurrently)
        self._cls_lock = threading.Lock()

    # ------------------------------------------------------------ judge context

    def _classify(self, flow: list[dict[str, Any]], domain: str
                  ) -> tuple[str | None, float, str | None, list]:
        """classify + (on abstain) the nearest state id + top-N neighborhood,
        race-free. ``top`` is [(state, cos_dist), ...] best-first when the
        scorer exposes it (graph_sidecar PERF_BOOST2), else []."""
        with self._cls_lock:
            state, dist = self.scorer.classify(flow, domain)
            nn = getattr(self.scorer, "_nn", None)
            nearest = None
            if state is None and self.cfg.nearest_on_abstain:
                nearest = getattr(nn, "state", None)
            top = list(getattr(nn, "top", None) or [])
        return state, dist, nearest, top

    def _task_goal(self, flow: list[dict[str, Any]]) -> str:
        """The task instruction (first user_message). The flow tail alone loses
        it: rendered flows exceed the tail budget after a step or two, so the
        judge would otherwise score 'progress toward the task' blind (F1)."""
        if not self.cfg.task_goal:
            return "(not shown)"
        for ev in flow:
            if isinstance(ev, dict) and ev.get("type") == "user_message":
                txt = re.sub(r"\s+", " ", str(ev.get("content") or "")).strip()
                if txt:
                    cap = self.cfg.max_task_chars
                    return txt[:cap] + (" …" if len(txt) > cap else "")
        return "(task instruction not present in the flow)"

    def _state_context(self, domain: str, state: str | None,
                       nearest: str | None, top: list | None = None
                       ) -> tuple[str, str, str]:
        """(state_label, state_stats, next_actions); on abstain fall back to the
        nearest state, clearly labelled low-confidence, instead of nothing (F5).
        With ``topn_states`` > 1 and a top-N neighborhood available (G3), the
        abstain context blends the top states instead of betting on one — finer
        graphs (eops 1029s) abstain on ~half the steps, and the single nearest
        state is often only marginally closer than its runner-up."""
        if state:
            return (state, self.interpret_state(domain, state),
                    self.next_actions(domain, state))
        k = self.cfg.topn_states
        if k > 1 and top:
            picks = top[:k]
            label = "UNKNOWN (soft: " + ", ".join(
                f"{s} d={d:.2f}" for s, d in picks) + ", LOW CONFIDENCE)"
            stats = "Low-confidence stats of the NEAREST states (classifier " \
                    "abstained; may not apply):\n" + "\n".join(
                        f"- {self.interpret_state(domain, s)}" for s, _ in picks)
            acts = "\n".join(f"[from {s}]\n{self.next_actions(domain, s)}"
                             for s, _ in picks)
            return label, stats, acts
        if nearest:
            return (f"UNKNOWN (nearest: {nearest}, LOW CONFIDENCE)",
                    "Low-confidence stats of the NEAREST state (classifier "
                    "abstained; may not apply): "
                    + self.interpret_state(domain, nearest),
                    self.next_actions(domain, nearest))
        return "UNKNOWN", "(unclassified)", self.next_actions(domain, None)

    def _render_candidates(self, flows: list[list[dict[str, Any]]]) -> list[dict[str, str]]:
        """Render each candidate's final ai_message. Budgets are config-driven
        (F2): the old 300-char caps hid argument differences between eops MCP
        candidates, forcing judge ties. The digest is over the FULL action, so
        dedup never confuses candidates that differ past the render cap."""
        out: list[dict[str, str]] = []
        for fl in flows:
            ev = fl[-1] if fl and isinstance(fl[-1], dict) and fl[-1].get("type") == "ai_message" else {}
            tools = ", ".join(sorted({tc.get("name") for tc in (ev.get("tool_calls") or [])
                                      if tc.get("name")})) or "(no tool — finishing)"
            args = json.dumps([tc.get("args") for tc in (ev.get("tool_calls") or [])],
                              default=str, sort_keys=True)
            content = str(ev.get("content") or "")
            out.append({
                "tools": tools,
                "args": args[: self.cfg.cand_args_chars],
                "content": content[: self.cfg.cand_content_chars],
                "digest": hashlib.md5("\x00".join((tools, args, content)).encode()
                                      ).hexdigest()[:10],
                # digest over the executed action only (tool name(s) + normalized
                # args); ignores the surrounding free text and JSON whitespace so
                # candidates that call the same tool the same way collapse.
                "sem_digest": hashlib.md5("\x00".join((tools, args)).encode()
                                          ).hexdigest()[:10],
            })
        return out

    def _render_rollouts(self, flows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Render whole-rollout candidates (J1): final answer + quick health
        stats + a per-rollout trajectory tail. Dedup digest is over the
        NORMALIZED final answer — sibling rollouts that reached the same answer
        are one candidate (self-consistency for free), and an answer-identical
        set skips the judge entirely."""
        out: list[dict[str, Any]] = []
        for fl in flows:
            ans = ""
            for ev in reversed(fl):
                if (isinstance(ev, dict) and ev.get("type") == "ai_message"
                        and str(ev.get("content") or "").strip()):
                    # only a tool-call-free closing message is a real final
                    # answer; a trailing tool action means the rollout ended
                    # without answering — don't mislabel its args as one.
                    if not ev.get("tool_calls"):
                        ans = re.sub(r"\s+", " ", str(ev["content"])).strip()
                    break
            n_res = n_err = 0
            for ev in fl:
                if isinstance(ev, dict) and ev.get("type") == "tool_result":
                    n_res += 1
                    res = ev.get("result")
                    if isinstance(res, dict) and (res.get("error")
                                                  or res.get("success") is False):
                        n_err += 1
            tail = self.render.render_text(fl, self.scorer.mtr, self.scorer.mfc
                                           )[-self.cfg.rollout_tail_chars:]
            out.append({
                "answer": ans[: self.cfg.rollout_answer_chars] or "(no final answer)",
                "steps": n_res, "errors": n_err, "tail": tail,
                "digest": hashlib.md5(ans.lower().encode()).hexdigest()[:10],
            })
        return out

    def _tool_docs_block(self, flow: list[dict[str, Any]]) -> str:
        """F8: a '[Policy tool catalog]' section for the judge/advisor prompts.

        Prefers the LIVE tool catalog the executor put in the flow (eops
        wm_react emits a ``{"type": "tools", "tools": [...]}`` event with real
        MCP names / descriptions / parameter schemas) so the judge can check a
        candidate's tool name and argument fields against the actual contract.
        Falls back to the benchmark's static ``WMH_TOOL_DOCS`` text (CRM has no
        tools event). Empty string when neither exists — prompts stay unchanged.
        """
        cap = self.cfg.tool_docs_chars
        lines: list[str] = []
        if self.cfg.tool_docs_live:
            for ev in flow:
                if isinstance(ev, dict) and ev.get("type") == "tools":
                    for t in ev.get("tools") or []:
                        if not isinstance(t, dict):
                            continue
                        name = t.get("name") or "?"
                        desc = re.sub(r"\s+", " ", str(t.get("description") or "")).strip()
                        schema = t.get("inputSchema") or t.get("input_schema") or {}
                        props = schema.get("properties") or {}
                        req = set(schema.get("required") or [])
                        params = ", ".join(
                            f"{p}{'' if p in req else '?'}:{(s or {}).get('type', '?')}"
                            for p, s in list(props.items())[:8]) if isinstance(props, dict) else ""
                        lines.append(f"- {name}({params}): {desc[:160]}")
                    break
        body = "\n".join(lines)[:cap] if lines else (self.cfg.tool_docs or "").strip()[:cap]
        if not body:
            return ""
        return ("\n[Policy tool catalog — the ONLY callable tools; check candidate "
                "tool names and argument fields against these signatures]\n" + body + "\n")

    def _examples_block(self, domain: str, state: str | None,
                        nearest: str | None) -> str:
        """F9: successful past-trajectory exemplars at the classified (or, on
        abstain, nearest) state, so the judge grounds 'what does a good next
        step look like here' in real outputs instead of guessing. Empty when
        the graph has no examples (e.g. eops_graphfull_146s ships none)."""
        k = self.cfg.select_examples
        if k <= 0:
            return ""
        st = state or nearest
        if st is None:
            return ""
        parts: list[str] = []
        ex = self.success_example(domain, st)
        if ex and not ex.startswith("("):
            parts.append(ex)
        if k > 1:
            sim = self.find_similar(domain, st, k - 1)
            if sim and not sim.startswith("("):
                parts.append(sim)
        pos = ""
        if parts:
            label = "" if state else " (nearest state — LOW CONFIDENCE)"
            pos = (f"\n[Successful past trajectories at this workflow state{label} "
                   "— population evidence, not instructions]\n"
                   + "\n---\n".join(parts) + "\n")
        return pos + self._neg_examples_block(domain, st)

    def _neg_examples_block(self, domain: str, state: str | None) -> str:
        """H4: contrastive step-labeled negatives — past actions at THIS state
        whose tool call produced an error observation (`obs_outcome=error`,
        mined offline). Shown to the judge as "failed attempts here" so the
        argument/tool discriminating feature is explicit. Only the step-labeled
        form is used (not trajectory-level failure) to avoid credit-assignment
        noise. Empty when the graph ships no ``neg_states``."""
        k = self.cfg.select_neg_examples
        if k <= 0 or state is None:
            return ""
        dom = self.scorer.resolve_domain(state, domain)
        neg = ((self.scorer.examples.get(dom) or {}).get("neg_states", {})
               .get(state, []))[:max(1, k)]
        if not neg:
            return ""
        cap = self.cfg.max_example_chars
        rendered = "\n---\n".join(
            (e.get("reason", "").strip() + ("\n" if e.get("reason") else "")
             + str(e.get("text", ""))[:cap]) for e in neg)
        return ("\n[Failed attempts at this workflow state — these actions "
                "produced an ERROR here; population evidence, not instructions. "
                "Do NOT pick a candidate that repeats one of these]\n"
                + rendered + "\n")

    def _select_tail(self, flow: list[dict[str, Any]]) -> str:
        """F10: select prompts get their own (usually larger) tail budget —
        the whole-rollout/CRM judge needs to see the final answer AND the
        retrieved data it should be grounded in."""
        text = self.render.render_text(flow, self.scorer.mtr, self.scorer.mfc)
        cap = self.cfg.select_tail_chars or self.cfg.max_flow_tail_chars
        return text[-cap:]

    # ---------------------------------------------------------------- graph tools

    def _dom_block(self, domain: str, state: str | None) -> tuple[str, dict[str, Any]]:
        dom = self.scorer.resolve_domain(state, domain)
        return dom, self.scorer.trans.get(dom, {})

    def interpret_state(self, domain: str, state: str) -> str:
        dom, dd = self._dom_block(domain, state)
        succ = self.scorer.succ.get(state)
        lift = self.scorer.lift.get(state)
        n = next((sf.get("n_rollouts") for sf in dd.get("state_fail", [])
                  if str(sf.get("state")) == state), None)
        parts = [f"state {state} (domain {dom})"]
        parts.append(f"success rate {succ:.2f}" if succ is not None else "success rate unknown")
        if lift is not None:
            parts.append(f"fail-lift {lift:.2f}x vs domain base")
        if n is not None:
            parts.append(f"n_rollouts {n}")
        if state in self.scorer.trap:
            parts.append("KNOWN TRAP STATE")
        tool = self.scorer.top_tool.get(state)
        if tool:
            parts.append(f"characteristic tool: {tool}")
        nxt = dd.get("P_next", {}).get(state, {})
        if nxt:
            top = sorted(nxt.items(), key=lambda kv: -kv[1])[:4]
            parts.append("P(next): " + ", ".join(f"{k} {v:.2f}" for k, v in top))
        return "; ".join(parts)

    def next_actions(self, domain: str, state: str | None, k: int = 6) -> str:
        if state is None:
            return "(state unknown — no action table)"
        dom, dd = self._dom_block(domain, state)
        edges = dd.get("tool_edges", {}).get(state, {})
        v_s = self.scorer.succ.get(state, self.scorer.base_for(dom))
        rows = []
        for tk, e in edges.items():
            # G4: min edge support is config-driven — the 1029-state graph
            # splits the same corpus over 7x more states, so n>=3 starves many
            # states of any action table ("no edges with n>=3").
            if e.get("n", 0) < self.cfg.edge_min_n:
                continue
            top_next = sorted((e.get("next") or {}).items(), key=lambda kv: -kv[1])[:2]
            nxt = ", ".join(f"{s} {p:.2f}" for s, p in top_next) or "?"
            rows.append((float(e.get("succ", 0.0)),
                         f"- action [{tk or 'no-tool/finish'}]: p_action {e.get('p_action', 0):.2f}, "
                         f"succ {e.get('succ', 0):.2f} (n={e.get('n')}), advantage "
                         f"{float(e.get('succ', 0)) - v_s:+.2f}, lands in: {nxt}"))
        rows.sort(key=lambda r: -r[0])
        header = f"V(state)={v_s:.2f}"
        return header + "\n" + ("\n".join(r[1] for r in rows[:k]) if rows
                                else f"(no edges with n>={self.cfg.edge_min_n})")

    def _graph_pref_idx(self, domain: str, state: str, cands: list[dict[str, str]]
                        ) -> int | None:
        """H25: index of the candidate whose tool matches the classified state's
        highest-success outgoing edge (min support edge_min_n), else None. Loose
        containment match between the edge's tool key and the candidate's rendered
        tool list."""
        dom, dd = self._dom_block(domain, state)
        edges = dd.get("tool_edges", {}).get(state, {})
        best_tool, best_succ = None, -1.0
        for tk, e in edges.items():
            if not tk or e.get("n", 0) < self.cfg.edge_min_n:
                continue
            if float(e.get("succ", 0.0)) > best_succ:
                best_succ, best_tool = float(e.get("succ", 0.0)), tk
        if not best_tool:
            return None
        toks = [t for t in re.split(r"[|,\s]+", str(best_tool)) if t]
        for i, c in enumerate(cands):
            ct = c.get("tools", "")
            if any(t and t in ct for t in toks):
                return i
        return None

    def find_similar(self, domain: str, state: str | None, k: int = 2) -> str:
        if state is None:
            return "(state unknown — no examples)"
        dom, _ = self._dom_block(domain, state)
        ex_all = (self.scorer.examples.get(dom) or {}).get("states", {}).get(state, [])
        if self.cfg.select_examples_rank:
            # H26: most prototypical first (highest stored cos to the state
            # centroid) instead of arbitrary mining order.
            ex_all = sorted(ex_all, key=lambda e: -float(e.get("cos", 0.0)))
        ex = ex_all[:max(1, k)]
        if not ex:
            return "(no stored examples for this state)"
        cap = self.cfg.max_example_chars
        return "\n---\n".join(e.get("text", "")[:cap] for e in ex)

    def success_example(self, domain: str, state: str | None) -> str:
        if state is None:
            return "(state unknown — no example)"
        dom, dd = self._dom_block(domain, state)
        block = self.wm_prompt.render_block_action(
            dd, state, self.scorer.top_tool, examples=self.scorer.examples.get(dom),
            n_shot=1, max_example_chars=self.cfg.max_example_chars)
        return block or "(no concrete successful next step recorded)"

    def _run_tool(self, name: str, args: dict[str, Any], domain: str,
                  state: str | None) -> str:
        if name == "find_similar":
            return self.find_similar(domain, state, int(args.get("k", 2)))
        if name == "success_example":
            return self.success_example(domain, state)
        if name == "interpret_state":
            return self.interpret_state(domain, str(args.get("state") or state))
        return f"(unknown tool {name})"

    # ------------------------------------------------------------------ budgets

    def _budget_ok(self, ep: str, state: str | None) -> bool:
        with self._lock:
            if self._episode_counts.get(ep, 0) >= self.cfg.max_per_episode:
                return False
            # once per (episode, state) — None (unclassified) counts as a state
            # too, so e.g. the same retry-loop is never advised twice.
            if state in self._episode_states.get(ep, set()):
                return False
        return True

    def _budget_consume(self, ep: str, state: str | None) -> None:
        with self._lock:
            self._episode_counts[ep] = self._episode_counts.get(ep, 0) + 1
            self._episode_states.setdefault(ep, set()).add(state)
            for d in (self._episode_counts, self._episode_states):
                while len(d) > 4096:
                    d.popitem(last=False)

    # -------------------------------------------------------------------- advise

    def _flow_tail(self, flow: list[dict[str, Any]]) -> str:
        text = self.render.render_text(flow, self.scorer.mtr, self.scorer.mfc)
        return text[-self.cfg.max_flow_tail_chars:]

    def advise(self, flow: list[dict[str, Any]], domain: str) -> dict[str, Any]:
        t0 = time.time()
        state, dist, nearest, top = self._classify(flow, domain)
        retry = detect_retry_loop(flow)
        is_trap = state in self.scorer.trap
        ep = episode_key(flow)
        rec: dict[str, Any] = {
            "kind": "advise", "domain": domain, "state": state or "UNKNOWN",
            "cos_dist": round(float(dist), 4), "abstain": state is None,
            "is_trap": is_trap, "retry_loop": retry, "episode": ep,
            "triggered": False, "injected": False, "tool_calls": [], "llm_calls": 0,
            "tokens": 0,
        }
        trig = self.cfg.trigger
        triggered = (retry["detected"] or is_trap) if trig == "trap_or_loop" else (
            state is not None if trig == "confident" else True)
        if not triggered or not self._budget_ok(ep, state):
            rec["skip"] = "budget" if triggered else "trigger"
            return self._finish(rec, t0, block="")
        rec["triggered"] = True

        trap_line = (f"TRAP: this is a known trap state (fail-lift "
                     f"{self.scorer.lift.get(state, 0):.2f}x)." if is_trap else
                     "Not a known trap state.")
        retry_line = (f"RETRY-LOOP detected live: {retry.get('count')}x "
                      f"[{retry.get('tool')}] ({retry.get('kind')})."
                      if retry["detected"] else "No live retry-loop.")
        confidence = "abstained (population stats may not apply)" if state is None else "confident"
        st_label, st_stats, nxt = self._state_context(domain, state, nearest, top)
        user = load_prompt(
            "advise_user", domain=domain, state=st_label,
            cos_dist=f"{dist:.3f}", confidence=confidence, trap_line=trap_line,
            retry_line=retry_line, task_goal=self._task_goal(flow),
            state_stats=st_stats, next_actions=nxt,
            tool_docs_block=self._tool_docs_block(flow),
            flow_tail=self._flow_tail(flow))
        system = load_prompt("system", max_advice_chars=self.cfg.max_advice_chars,
                             max_tool_calls=self.cfg.max_tool_calls,
                             action_space=self.cfg.action_space or GENERIC_ACTION_SPACE,
                             advice_rules=self.cfg.advice_rules or
                             "- Suggest only actions the action space above allows.")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]

        advice = ""
        for _ in range(self.cfg.max_tool_calls + 1):
            reply, usage = self._chat(messages, rec)
            obj = parse_json_reply(reply)
            if obj is not None and "tool" in obj and "advice" not in obj:
                name = str(obj.get("tool"))
                rec["tool_calls"].append(name)
                result = self._run_tool(name, obj.get("args") or {}, domain, state)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"[tool result: {name}]\n{result}"})
                continue
            if obj is not None and "advice" in obj:
                advice = str(obj.get("advice") or "").strip()
            else:
                # bare-prose or malformed-JSON advice — salvage rather than drop
                # the (often high-value, trap/retry-triggered) guidance.
                advice = salvage_advice(reply, self.cfg.max_advice_chars)
                if advice:
                    rec["salvaged"] = True
                else:
                    rec["parse_error"] = reply[:200]
            break
        else:
            rec["tool_budget_exhausted"] = True

        advice = advice[: self.cfg.max_advice_chars].strip()
        reason = validate_advice(advice, salvaged=bool(rec.get("salvaged")))
        if advice and reason:
            rec["rejected"] = reason
            advice = ""
        block = self.BLOCK_PREFIX + advice if advice else ""
        if block:
            self._budget_consume(ep, state)
        rec["injected"] = bool(block)
        rec["advice"] = advice
        return self._finish(rec, t0, block=block)

    # -------------------------------------------------------------------- select

    def select_joint(self, flows: list[list[dict[str, Any]]], domain: str,
                     episode_id: str | None = None) -> list[dict[str, Any]]:
        """Joint best-of-N comparison. Two shapes, auto-detected:

        * **step mode** (wm_react): flows share a prefix; each ends in one
          candidate ai_message — compare the candidate next actions.
        * **rollout mode** (PERF_BOOST2 J1, CRM `CRM_WM_SELECT_JOINT=1`): flows
          are K *whole rollouts* of the same task (no shared prefix) — compare
          final answers + per-rollout trajectory tails. Replaces K independent
          ``select_single`` calls, whose absolute 0/1-ish verifier scores tied
          on 17/28 measured headroom tasks (PERF_BOOST §10.2 evidence).

        Protocol (EOPS_HARNESS_DIAGNOSIS.md + PERF_BOOST2): dedup identical
        candidates (by full action in step mode, by final answer in rollout
        mode) and skip the LLM when only one distinct remains (F2); shuffle
        judge order (F4); comparative ranking with an explicit ``best`` (F3);
        task excerpt (F1), nearest/top-N state context on abstain (F5/G3),
        tool docs (F8), exemplars (F9). ``select_votes`` > 1 (J2) repeats the
        judgement with reshuffled candidate order and takes the majority
        ``best`` / mean scores — cancels residual position noise. Downstream
        picks argmax(success_score) with ties→lowest index, so when a best is
        named the returned scores are adjusted to make argmax==best; raw judge
        scores stay in the log (``scores_raw``)."""
        t0 = time.time()
        step_shape = bool(flows and flows[0]) and all(
            fl[:-1] == flows[0][:-1] for fl in flows)
        pre = (flows[0][:-1] if step_shape else (flows[0] or [])) if flows else []
        state, dist, nearest, top = self._classify(pre, domain)
        n = len(flows)
        rec: dict[str, Any] = {"kind": "select_joint", "domain": domain,
                               "mode": "step" if step_shape else "rollout",
                               "state": state or "UNKNOWN", "cos_dist": round(float(dist), 4),
                               "n_candidates": n, "llm_calls": 0, "tokens": 0,
                               "episode": episode_id}  # H7: task join key
        cands = (self._render_candidates(flows) if step_shape
                 else self._render_rollouts(flows))
        digests = [c["digest"] for c in cands]
        # dedup/skip identity: the semantic (action-only) digest when enabled,
        # so whitespace / free-text twins collapse instead of forcing the judge
        # to rank formatting noise.
        dkey = ([c.get("sem_digest", c["digest"]) for c in cands]
                if self.cfg.select_sem_dedup else digests)
        uniq = list(dict.fromkeys(dkey))
        rec["cand_digests"] = digests
        if self.cfg.select_sem_dedup:
            rec["cand_sem"] = dkey
        rec["n_distinct"] = len(uniq)

        scores: list[float] | None = None
        if self.cfg.select_dedup and len(uniq) == 1:
            # every candidate proposes the same action / final answer — nothing
            # to select; equal scores make the client keep candidate 0 (i.i.d.
            # sample = exactly the base arm's behavior), zero judge latency.
            rec["skip"] = "identical_candidates"
            scores = [0.5] * n
        else:
            base_order = ([dkey.index(d) for d in uniq] if self.cfg.select_dedup
                          else list(range(n)))
            st_label, st_stats, nxt = self._state_context(domain, state, nearest, top)
            common = dict(
                domain=domain, state=st_label, task_goal=self._task_goal(pre),
                state_stats=st_stats, next_actions=nxt,
                flow_tail=self._select_tail(pre),
                tool_docs_block=self._tool_docs_block(pre),
                examples_block=self._examples_block(domain, state, nearest))
            prompt_name = "select_joint" if step_shape else "select_joint_rollout"

            def _lines(order: list[int]) -> list[str]:
                if step_shape:
                    return [f"candidate {j}: tools=[{cands[i]['tools']}] "
                            f"args={cands[i]['args']} says: {cands[i]['content']}"
                            for j, i in enumerate(order)]
                return [f"candidate {j}: FINAL ANSWER: {cands[i]['answer']} "
                        f"[steps={cands[i]['steps']}, failed_calls={cands[i]['errors']}]\n"
                        f"--- candidate {j} trajectory tail ---\n{cands[i]['tail']}"
                        for j, i in enumerate(order)]

            judged_mean: dict[int, float] = {}
            best_orig: int | None = None
            if self.cfg.select_tournament and len(base_order) > 2:
                # J3 (PERF_BOOST3): the joint judge is strong at K=2 and
                # saturates beyond (judge_eff 0.534@2 → 0.200@8, PERF_BOOST2
                # §9) — run a single-elimination pairwise bracket instead of
                # one K-way comparison. ceil(log2 K) rounds; the final match
                # gets the full vote count, earlier matches one vote each.
                judged_mean, best_orig, tlog = self._run_tournament(
                    prompt_name, common, _lines, base_order, rec)
                rec["tournament"] = tlog
                if self.cfg.log_candidates:
                    rec["candidates"] = _lines(base_order)
            else:
                votes = max(1, self.cfg.select_votes)
                vote_scores: list[dict[int, float]] = []   # orig idx -> score
                vote_bests: list[int] = []
                for v in range(votes):
                    order = list(base_order)
                    if self.cfg.select_shuffle and len(order) > 1:
                        seed_src = "".join(digests) + ("" if v == 0 else f"|v{v}")
                        rng = random.Random(int(hashlib.md5(
                            seed_src.encode()).hexdigest()[:8], 16))
                        rng.shuffle(order)
                    if v == 0:
                        rec["judge_order"] = order
                        if self.cfg.log_candidates:
                            rec["candidates"] = _lines(order)
                    user = load_prompt(prompt_name, n=len(order),
                                       candidates="\n".join(_lines(order)), **common)
                    try:
                        reply, _ = self._chat([{"role": "user", "content": user}], rec)
                        obj = parse_json_reply(reply) or {}
                        raw = obj.get("scores")
                        if isinstance(raw, list) and len(raw) == len(order):
                            vote_scores.append({i: max(0.0, min(1.0, float(s)))
                                                for i, s in zip(order, raw)})
                            b = obj.get("best")
                            if (self.cfg.select_ranking and isinstance(b, (int, float))
                                    and 0 <= int(b) < len(order)):
                                vote_bests.append(order[int(b)])
                            if v == 0 and obj.get("compare"):
                                rec["compare"] = str(obj["compare"])[:200]
                    except Exception as exc:  # LLM down → graph fallback below
                        rec["error"] = str(exc)
                if vote_scores:
                    judged_mean = {i: statistics.mean([vs[i] for vs in vote_scores
                                                       if i in vs])
                                   for i in base_order
                                   if any(i in vs for vs in vote_scores)}
                    if vote_bests:
                        best_orig = max(set(vote_bests),
                                        key=lambda i: (vote_bests.count(i),
                                                       judged_mean.get(i, 0.0)))
                    if votes > 1:
                        rec["best_votes"] = vote_bests
            if judged_mean:
                # positional map; deduped duplicates inherit their
                # representative's score via the digest
                pos: list[float | None] = [None] * n
                for i, s in judged_mean.items():
                    pos[i] = s
                if self.cfg.select_dedup:
                    rep = {dkey[i]: s for i, s in judged_mean.items()}
                    scores = [rep.get(d, 0.0) if s is None else s
                              for s, d in zip(pos, dkey)]
                else:
                    scores = [0.0 if s is None else s for s in pos]
                rec["scores_raw"] = [round(s, 4) for s in scores]
                rec["best"] = best_orig
                if best_orig is not None:
                    m = max(scores)
                    scores = [m if i == best_orig else max(0.0, min(s, m - 0.01))
                              for i, s in enumerate(scores)]
        if scores is None:
            rec["fallback"] = "action_score"
            scores = []
            for fl in flows:
                ss, *_ = self.scorer.action_score(fl, domain)
                scores.append(float(ss))
        # H2: override margin gate. Only deviate from candidate 0 (the greedy
        # anchor, when VLLM_GWM_GREEDY_ANCHOR is on) if the judge's winner beats
        # it by a real score margin; otherwise re-pin cand0 so downstream argmax
        # keeps it. Kills the low-margin coin-flip overrides (TOUCAN E4/E2).
        # H23: use a larger margin on abstain steps (thin evidence). H25: when the
        # gate fires, optionally re-pin the graph-preferred candidate (tool matches
        # the state's best edge) instead of cand0.
        margin = self.cfg.select_min_margin
        if state is None and self.cfg.select_min_margin_abstain >= 0.0:
            margin = self.cfg.select_min_margin_abstain
        if (margin > 0.0 and scores and len(scores) > 1
                and rec.get("skip") != "identical_candidates"):
            if max(scores) - scores[0] < margin:
                keep = 0
                if (self.cfg.select_graph_tiebreak and step_shape
                        and state is not None):
                    gi = self._graph_pref_idx(domain, state, cands)
                    if gi is not None:
                        keep = gi
                m = max(scores)
                scores = [m if i == keep else min(s, m - 0.01)
                          for i, s in enumerate(scores)]
                rec["margin_gate"] = f"kept_cand{keep}"
                rec["margin_used"] = margin
        rec["scores"] = [round(s, 4) for s in scores]
        self._finish(rec, t0, block="")
        return [{"state": state or "UNKNOWN", "cos_dist": float(dist),
                 "abstain": state is None, "is_trap": state in self.scorer.trap,
                 "last_tool": "", "injected": False, "block": "",
                 "success_score": s, "probability": 1.0, "score_via": "harness_joint"}
                for s in scores]

    def _match(self, prompt_name: str, common: dict[str, Any], lines_fn,
               pair: tuple[int, int], rec: dict[str, Any], votes: int,
               tag: str) -> int:
        """One tournament match: judge the 2 candidates in ``pair`` (order
        shuffled per vote), return the winning ORIGINAL index. Judge silent /
        unparseable on every vote → deterministic advance (lower index =
        the earlier i.i.d. sample, matching the argmax tie rule)."""
        a, b = pair
        bests: list[int] = []
        for v in range(max(1, votes)):
            order = [a, b]
            if self.cfg.select_shuffle:
                rng = random.Random(int(hashlib.md5(
                    f"{tag}|{a}|{b}|{v}".encode()).hexdigest()[:8], 16))
                rng.shuffle(order)
            user = load_prompt(prompt_name, n=2,
                               candidates="\n".join(lines_fn(order)), **common)
            try:
                reply, _ = self._chat([{"role": "user", "content": user}], rec)
                obj = parse_json_reply(reply) or {}
                bst = obj.get("best")
                if isinstance(bst, (int, float)) and 0 <= int(bst) < 2:
                    bests.append(order[int(bst)])
                    continue
                raw = obj.get("scores")
                if (isinstance(raw, list) and len(raw) == 2
                        and float(raw[0]) != float(raw[1])):
                    bests.append(order[0] if float(raw[0]) > float(raw[1])
                                 else order[1])
            except Exception as exc:
                rec["error"] = str(exc)
        if not bests:
            return min(a, b)
        return sorted(set(bests), key=lambda i: (-bests.count(i), i))[0]

    def _run_tournament(self, prompt_name: str, common: dict[str, Any], lines_fn,
                        base_order: list[int], rec: dict[str, Any]
                        ) -> tuple[dict[int, float], int, list]:
        """J3: single-elimination pairwise bracket over the distinct candidates.
        Returns (per-candidate rank scores, winner, per-round match log).
        Scores encode elimination depth (winner 0.9, final loser 0.8, …) —
        downstream only argmax matters; the log keeps the full bracket."""
        alive = list(base_order)
        eliminated_at: dict[int, int] = {}
        tlog: list = []
        depth = 0
        while len(alive) > 1:
            depth += 1
            nxt: list[int] = []
            rnd: list = []
            for m in range(0, len(alive) - 1, 2):
                a, b = alive[m], alive[m + 1]
                votes = self.cfg.select_votes if len(alive) == 2 else 1
                win = self._match(prompt_name, common, lines_fn, (a, b), rec,
                                  votes, f"r{depth}m{m // 2}")
                eliminated_at[b if win == a else a] = depth
                nxt.append(win)
                rnd.append([a, b, win])
            if len(alive) % 2:
                nxt.append(alive[-1])   # odd pool: bye, advances unjudged
            tlog.append(rnd)
            alive = nxt
        winner = alive[0]
        judged = {i: max(0.1, 0.9 - 0.1 * (depth - eliminated_at[i] + 1))
                  for i in base_order if i in eliminated_at}
        judged[winner] = 0.9
        return judged, winner, tlog

    def select_single(self, flow: list[dict[str, Any]], domain: str) -> dict[str, Any]:
        """Whole-rollout scoring (CRM best-of-K shape): one score per finished flow."""
        t0 = time.time()
        state, dist, nearest, top = self._classify(flow, domain)
        rec: dict[str, Any] = {"kind": "select_single", "domain": domain,
                               "state": state or "UNKNOWN", "cos_dist": round(float(dist), 4),
                               "llm_calls": 0, "tokens": 0}
        st_label, st_stats, _ = self._state_context(domain, state, nearest, top)
        user = load_prompt(
            "select_single", domain=domain, state=st_label,
            task_goal=self._task_goal(flow), state_stats=st_stats,
            tool_docs_block=self._tool_docs_block(flow),
            examples_block=self._examples_block(domain, state, nearest),
            flow_tail=self._select_tail(flow))
        score = None
        try:
            reply, _ = self._chat([{"role": "user", "content": user}], rec)
            obj = parse_json_reply(reply) or {}
            if "score" in obj:
                score = max(0.0, min(1.0, float(obj["score"])))
                rec["reason"] = str(obj.get("reason", ""))[:200]
        except Exception as exc:
            rec["error"] = str(exc)
        if score is None:
            rec["fallback"] = "state_value"
            score = float(self.scorer.succ.get(state, self.scorer.base_for(domain)))
        rec["score"] = round(score, 4)
        self._finish(rec, t0, block="")
        return {"state": state or "UNKNOWN", "cos_dist": float(dist),
                "abstain": state is None, "is_trap": state in self.scorer.trap,
                "last_tool": "", "injected": False, "block": "",
                "success_score": score, "probability": 1.0, "score_via": "harness_single"}

    # ------------------------------------------------------------------ plumbing

    def _chat(self, messages: list[dict[str, str]], rec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        last_exc: Exception | None = None
        for _ in range(self.cfg.llm_retries + 1):
            try:
                reply, usage = self.llm.chat(messages)
                rec["llm_calls"] += 1
                rec["tokens"] += int(usage.get("prompt_tokens") or 0) + int(
                    usage.get("completion_tokens") or 0)
                return reply, usage
            except Exception as exc:
                last_exc = exc
        raise last_exc  # noqa: raise the final failure to the caller's handler

    def _finish(self, rec: dict[str, Any], t0: float, *, block: str) -> dict[str, Any]:
        rec["latency_ms"] = int((time.time() - t0) * 1000)
        self._count(rec)
        self._log(rec)
        if rec["kind"] != "advise":
            return rec
        return {"state": rec["state"], "cos_dist": rec["cos_dist"],
                "abstain": rec["abstain"], "is_trap": rec["is_trap"],
                "last_tool": (rec.get("retry_loop") or {}).get("tool", ""),
                "injected": rec["injected"], "block": block,
                "success_score": self.scorer.succ.get(
                    rec["state"], self.scorer.base_for(rec["domain"])),
                "probability": 1.0,
                "harness": {k: rec.get(k) for k in
                            ("triggered", "retry_loop", "tool_calls", "llm_calls",
                             "tokens", "latency_ms", "skip", "parse_error", "rejected")
                            if rec.get(k) is not None}}

    def _count(self, rec: dict[str, Any]) -> None:
        st = self.stats
        with self._stats_lock:
            st["events"] += 1
            kind = str(rec.get("kind"))
            st["by_kind"][kind] = st["by_kind"].get(kind, 0) + 1
            if rec.get("skip"):
                sk = str(rec["skip"])
                st["skips"][sk] = st["skips"].get(sk, 0) + 1
            if rec.get("state") in (None, "UNKNOWN"):
                st["unknown_state"] += 1
            calls = int(rec.get("llm_calls") or 0)
            if calls:
                st["judged"] += 1
            st["llm_calls"] += calls
            st["tokens"] += int(rec.get("tokens") or 0)
            scores = rec.get("scores")
            if (isinstance(scores, list) and len(set(scores)) > 1
                    and scores.index(max(scores)) != 0):
                st["overrides"] += 1
            if rec.get("fallback"):
                st["fallbacks"] += 1
            if rec.get("injected"):
                st["injected"] += 1
            st["latency_ms_total"] += int(rec.get("latency_ms") or 0)

    def stats_snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            return json.loads(json.dumps(self.stats))

    def _log(self, rec: dict[str, Any]) -> None:
        if not self.cfg.log_path:
            return
        line = json.dumps({"ts": round(time.time(), 3), **rec}, default=str)
        with self._log_lock:
            with open(self.cfg.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
