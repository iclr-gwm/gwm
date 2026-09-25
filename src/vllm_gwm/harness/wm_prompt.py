#!/usr/bin/env python3
"""Render a transition-graph "world-model" guidance block for prompt injection.

No neural scorer — the mined graph IS the world model. Given the agent's current
discovered state, we look up (from transitions.json + per-state success rates):
  - current state: name + historical success rate
  - P(next | current): the empirical next-state distribution
  - for each candidate next state: its historical success rate
  - failure traps: self-loops / early-END transitions with high failure lift
and format them as a [World-Model] message to inject before the agent acts.

Usage::

    python3 wm_prompt.py --domain teams --state teams:5
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def state_names(states_path: Path) -> dict:
    tools = {}
    if states_path.exists():
        agg: dict = {}
        for l in gzip.open(states_path, "rt"):
            r = json.loads(l)
            c = str(r.get("cluster"))
            agg.setdefault(c, Counter()).update(r.get("tool_names") or [])
        for c, cc in agg.items():
            tools[c] = cc.most_common(1)[0][0] if cc else "?"
    return tools


def _example_snippet(ex: dict, max_chars: int) -> str:
    """Trim a stored full-tail example to a render-time budget (keep the tail, which
    carries the outcome / most recent action+observation)."""
    t = (ex.get("text") or "").strip()
    if len(t) > max_chars:
        t = "…" + t[-max_chars:]
    return t


def _render_next_state_examples(examples, state, next_states, names,
                                max_example_chars, budget_left, max_shots=2):
    """Show what a SUCCESSFUL next state looks like — the landing step of edge
    state->nxt from past successful trajectories. Grounds 'where am I headed /
    which tool gets me there'. Only called for non-trap recommended transitions
    (the harvested transition examples are success-only, so they cannot illustrate
    a *failed* edge). Returns rendered lines (may be empty)."""
    if not examples or not next_states:
        return [], budget_left
    trans = examples.get("transitions") or {}

    def nm(s):
        return f"{s}({names[s]})" if s in names else s

    lines = []
    for nxt in next_states:
        ex = trans.get(f"{state}->{nxt}") or []
        if not ex:
            continue
        snip = _example_snippet(ex[0], max_example_chars)
        block = f"  → successful path to {nm(nxt)}:\n    {snip}"
        if len(block) > budget_left:
            break
        if not lines:
            lines.append("What a successful next step looks like (from past successful trajectories):")
        lines.append(block)
        budget_left -= len(block)
        if len(lines) - 1 >= max_shots:
            break
    return lines, budget_left


def _render_examples(examples, state, traps, names, n_shot, max_example_chars,
                     max_block_chars, used_chars, next_states=None):
    """Build the grounding 'Concrete examples' lines for n_shot>0.

    `examples` is the per-domain dict {"states":{id:[ex...]}, "transitions":{"a->b":[ex...]}}.
    For trap states we also surface up to a couple of transition examples for the
    flagged edges (showing the avoided behavior in context). Respects both the
    per-example and total-block char budgets; appends an omission note if truncated.
    """
    if not examples or n_shot <= 0:
        return []
    st_ex = (examples.get("states") or {}).get(state, [])[:n_shot]
    lines, budget_left = [], max_block_chars - used_chars
    if not st_ex:
        return []

    def nm(s):
        return f"{s}({names[s]})" if s in names else s

    lines.append(f"Concrete examples of state {nm(state)} (from past successful trajectories):")
    shown = 0
    for i, ex in enumerate(st_ex, 1):
        snip = _example_snippet(ex, max_example_chars)
        block = f"  [{i}] {snip}"
        if len(block) > budget_left and shown > 0:
            lines.append(f"  …[+{len(st_ex) - shown} more examples omitted for length]")
            break
        lines.append(block)
        budget_left -= len(block)
        shown += 1
    # Successful next-state shots: only for the NON-TRAP recommended transitions
    # passed in by render_block. Trap edges are deliberately excluded — the harvested
    # transition examples are success-only and cannot illustrate a failed edge.
    nxt_lines, budget_left = _render_next_state_examples(
        examples, state, next_states or [], names, max_example_chars, budget_left)
    lines += nxt_lines
    return lines


def render_block(dom_data: dict, state: str, names: dict, top: int = 4,
                 examples: dict | None = None, n_shot: int = 0,
                 max_example_chars: int = 1200, max_block_chars: int = 12000) -> str:
    P = dom_data["P_next"].get(state, {})
    succ = {s["state"]: 1.0 - s["fail_rate"] for s in dom_data["state_fail"]}
    base_succ = 1.0 - dom_data["base_fail_rate"]
    tfail = {(t["from"], t["to"]): t for t in dom_data["trans_fail"]}

    def nm(s):
        return f"{s}({names[s]})" if s in names else s

    cur_succ = succ.get(state, base_succ)
    lines = [
        "[World-Model guidance — empirical, from past trajectories]",
        f"Current workflow state: {nm(state)} — tasks here finish successfully "
        f"{cur_succ:.0%} of the time (domain avg {base_succ:.0%}).",
        "Most likely next states and their historical success:",
    ]
    ranked = sorted(P.items(), key=lambda x: -x[1])
    next_states: list[str] = []   # all non-trap, non-END, non-noise transitions, ranked by P
    for nxt, p in ranked:
        tf = tfail.get((state, nxt))
        trap_self = bool(tf and tf.get("self_loop")
                         and tf["fail_rate"] > base_succ + dom_data["base_fail_rate"])
        trap_end = bool(nxt == "END" and tf and tf["fail_rate"] > dom_data["base_fail_rate"])
        if nxt not in ("END", state, "noise") and not trap_self and not trap_end:
            next_states.append(nxt)
    for nxt, p in ranked[:top]:
        tag = ""
        tf = tfail.get((state, nxt))
        if tf and tf.get("self_loop") and tf["fail_rate"] > base_succ + dom_data["base_fail_rate"]:
            tag = "  ⚠ RETRY-LOOP: repeating this step usually FAILS"
        elif nxt == "END" and tf and tf["fail_rate"] > dom_data["base_fail_rate"]:
            tag = "  ⚠ premature finish here usually FAILS"
        sr = "" if nxt in ("END",) else f"→ success {succ.get(nxt, base_succ):.0%}"
        lines.append(f"  • {p:.0%} go to {nm(nxt)} {sr}{tag}")
    # explicit advice
    traps = [t for t in dom_data["trans_fail"][:6]
             if t["from"] == state and (t["self_loop"] or t["to"] == "END") and t["fail_rate"] > 0.6]
    if traps:
        lines.append("Avoid: " + "; ".join(
            f"re-running {nm(state)} (fails {t['fail_rate']:.0%})" if t["self_loop"]
            else f"ending now (fails {t['fail_rate']:.0%})" for t in traps))
    lines.append("Prefer transitions toward higher-success states; verify the last "
                 "tool result before retrying or finishing.")
    if n_shot > 0:
        used = sum(len(x) for x in lines) + len(lines)
        lines += _render_examples(examples, state, traps, names, n_shot,
                                  max_example_chars, max_block_chars, used,
                                  next_states=next_states)
    return "\n".join(lines)


def _render_tool_examples(examples, state, ranked_actions, names, n_shot,
                          max_example_chars, max_block_chars, used_chars):
    """Tool-conditioned grounding for graph_tools: for each ranked tool-action from
    `state`, show a real successful step where that toolset was called and the state
    it led to. Keyed by examples['tool_transitions'][state][toolset_key][next_state].
    Falls back silently to nothing if no tool-conditioned example exists."""
    if not examples or n_shot <= 0:
        return []
    tt = (examples.get("tool_transitions") or {}).get(state) or {}
    if not tt:
        return []

    def nm(s):
        return f"{s}({names[s]})" if s in names else s

    lines, budget_left, shown = [], max_block_chars - used_chars, 0
    for tk, action in ranked_actions:
        by_next = tt.get(tk) or {}
        if not by_next:
            continue
        # prefer the action's own modal next-state that we actually have an example for
        order = sorted(action.get("next", {}).items(), key=lambda x: -x[1])
        pick = next((nx for nx, _ in order if by_next.get(nx)), None)
        if pick is None:
            continue
        ex = by_next[pick][0]
        label = "[" + ", ".join(action.get("tools", [])) + "]" if action.get("tools") else "(no tool call)"
        snip = _example_snippet(ex, max_example_chars)
        block = f"  → calling {label} → {nm(pick)}:\n    {snip}"
        if len(block) > budget_left and shown > 0:
            break
        if not lines:
            lines.append("From this state, what each tool-action looks like "
                         "(real successful trajectories — which tool leads where):")
        lines.append(block)
        budget_left -= len(block)
        shown += 1
        if shown >= n_shot:
            break
    return lines


def render_block_value(dom_data: dict, state: str, names: dict, top_actions: int = 4,
                       top_next: int = 2, min_action_n: int = 3,
                       examples: dict | None = None, n_shot: int = 0,
                       max_example_chars: int = 1200, max_block_chars: int = 12000) -> str:
    """Value/advantage variant ("custom graph").

    The plain tool graph ranks actions by p_action*success, which rewards frequent
    in-success actions — usually self-loops/no-ops — not progress. Here we score each
    tool-action by its ADVANTAGE: A(s,a) = E_{s'}[V(s') | s,a] - V(s), where V(s) is
    the historical eventual-success rate of state s. We recommend the highest-advantage
    action and explicitly flag actions that do NOT advance (A<=0, e.g. retry loops)."""
    succ = {s["state"]: 1.0 - s["fail_rate"] for s in dom_data["state_fail"]}
    base_succ = 1.0 - dom_data["base_fail_rate"]
    actions = dom_data.get("tool_edges", {}).get(state, {})

    def nm(s):
        return f"{s}({names[s]})" if s in names else s

    def V(s):
        return succ.get(s, base_succ)

    # Advantage = P(task success | take action a from s) - V(s). This is the
    # decision-correct quantity (a["succ"] is the eventual-success of rollouts that
    # took this action), and avoids phantom credit from routing through END/noise
    # values. Self-loops are flagged separately since success credit on a repeated
    # step is noisy (a loop inside a winning trajectory still scores high).
    Vs = V(state)
    cand = [(tk, a) for tk, a in actions.items() if a["n"] >= min_action_n]
    for tk, a in cand:
        a["_adv"] = a["succ"] - Vs
        a["_self"] = max(a["next"].items(), key=lambda x: x[1])[0] == state
    ranked = sorted(cand, key=lambda x: (-x[1]["_adv"], x[1]["_self"]))
    ranked_actions = ranked[:top_actions]

    lines = [
        "[World-Model guidance — empirical, value-ranked, from past trajectories]",
        f"Current workflow state: {nm(state)} (historical success {Vs:.0%}, domain avg {base_succ:.0%}).",
        "Tool-actions ranked by ADVANTAGE — Δ = P(task succeeds | you take this action) − "
        "current success rate. Prefer the highest Δ; Δ≤0 means that action historically does "
        "NOT improve your odds. ⟳ marks repeating the same state (often a stall):",
    ]
    next_states: list[str] = []
    for tk, a in ranked_actions:
        label = "[" + ", ".join(a["tools"]) + "]" if a["tools"] else "(no tool call / stop)"
        adv = a["_adv"]
        flag = "  ✓ advances" if adv > 0.02 else ("  ⚠ does NOT advance" if adv <= 0 else "")
        loop = "  ⟳" if a["_self"] else ""
        lines.append(f"  • call {label}  (Δ {adv:+.0%}, used {a['p_action']:.0%} of the time){loop}{flag}")
        for nxt, p in sorted(a["next"].items(), key=lambda x: -x[1])[:top_next]:
            if nxt == "END":
                lines.append(f"      → {p:.0%} END")
                continue
            tag = "  (retry — same state)" if nxt == state else ""
            lines.append(f"      → {p:.0%} to {nm(nxt)} [success {V(nxt):.0%}]{tag}")
            if nxt not in ("noise", state) and nxt not in next_states:
                next_states.append(nxt)
    if not ranked:
        lines.append("  • (no tool-action history recorded for this state)")
    if ranked_actions and ranked_actions[0][1]["_adv"] <= 0:
        lines.append("No recorded action here improves success — re-read the last tool "
                     "result and the task; you may be missing a required step.")
    lines.append("Verify the last tool result before retrying or finishing.")
    if n_shot > 0:
        used = sum(len(x) for x in lines) + len(lines)
        st_lines = _render_examples(examples, state, [], names, min(n_shot, 1),
                                    max_example_chars, max_block_chars, used)
        lines += st_lines
        used += sum(len(x) for x in st_lines) + len(st_lines)
        tool_lines = _render_tool_examples(examples, state, ranked_actions, names,
                                           n_shot, max_example_chars, max_block_chars, used)
        if not tool_lines:
            tool_lines, _ = _render_next_state_examples(
                examples, state, next_states, names, max_example_chars,
                max_block_chars - used)
        lines += tool_lines
    return "\n".join(lines)


def render_block_tools(dom_data: dict, state: str, names: dict, top_actions: int = 4,
                       top_next: int = 3, min_action_n: int = 3,
                       examples: dict | None = None, n_shot: int = 0,
                       max_example_chars: int = 1200, max_block_chars: int = 12000) -> str:
    """Tool-conditioned variant of render_block.

    Instead of a single P(next | state) distribution, expose P(next | state, toolset):
    for each historical tool-action emitted from this state, show how often it is used,
    its rollout success rate, and where it leads. This lets the policy pick the *tool
    call* that empirically advances the workflow, not just the target state."""
    succ = {s["state"]: 1.0 - s["fail_rate"] for s in dom_data["state_fail"]}
    base_succ = 1.0 - dom_data["base_fail_rate"]
    actions = dom_data.get("tool_edges", {}).get(state, {})

    def nm(s):
        return f"{s}({names[s]})" if s in names else s

    cur_succ = succ.get(state, base_succ)
    lines = [
        "[World-Model guidance — empirical, tool-conditioned, from past trajectories]",
        f"Current workflow state: {nm(state)} — tasks here finish successfully "
        f"{cur_succ:.0%} of the time (domain avg {base_succ:.0%}).",
        "Your next tool call shapes where this goes. Historical tool-actions from here "
        "(share of actions, success rate, and most likely resulting state):",
    ]
    # rank actions by how much successful flow they carry: p_action * succ
    ranked = sorted(
        ((tk, a) for tk, a in actions.items() if a["n"] >= min_action_n),
        key=lambda x: -(x[1]["p_action"] * x[1]["succ"]))
    ranked_actions = ranked[:top_actions]
    next_states: list[str] = []
    for tk, a in ranked[:top_actions]:
        label = "[" + ", ".join(a["tools"]) + "]" if a["tools"] else "(no tool call)"
        lines.append(f"  • call {label}  ({a['p_action']:.0%} of actions, success {a['succ']:.0%})")
        for nxt, p in sorted(a["next"].items(), key=lambda x: -x[1])[:top_next]:
            if nxt == "END":
                warn = "  ⚠ premature finish here usually FAILS" if a["succ"] < base_succ else ""
                lines.append(f"      → {p:.0%} END{warn}")
                continue
            warn = ""
            if nxt == state and a["succ"] < base_succ:
                warn = "  ⚠ RETRY-LOOP: repeating this usually FAILS"
            sr = f" [success {succ.get(nxt, base_succ):.0%}]"
            lines.append(f"      → {p:.0%} to {nm(nxt)}{sr}{warn}")
            if nxt not in ("noise", state) and nxt not in next_states:
                next_states.append(nxt)
    if not ranked:
        lines.append("  • (no tool-action history recorded for this state)")
    lines.append("Prefer the tool-action with the highest success that advances the "
                 "workflow; verify the last tool result before retrying or finishing.")
    if n_shot > 0:
        # one state example for context, then tool-conditioned "which tool leads
        # where" examples (the focus of this arm). Tool examples fall back to the
        # state->next examples when no tool-keyed example was harvested.
        used = sum(len(x) for x in lines) + len(lines)
        st_lines = _render_examples(examples, state, [], names, min(n_shot, 1),
                                    max_example_chars, max_block_chars, used)
        lines += st_lines
        used += sum(len(x) for x in st_lines) + len(st_lines)
        tool_lines = _render_tool_examples(examples, state, ranked_actions, names,
                                           n_shot, max_example_chars, max_block_chars, used)
        if not tool_lines:   # fall back to state->next grounding if no tool-keyed examples
            tool_lines, _ = _render_next_state_examples(
                examples, state, next_states, names, max_example_chars,
                max_block_chars - used)
        lines += tool_lines
    return "\n".join(lines)


def render_block_action(dom_data: dict, state: str, names: dict,
                        examples: dict | None = None, n_shot: int = 1,
                        max_example_chars: int = 600,
                        max_block_chars: int = 1400) -> str:
    """Minimal actionable variant (CRM in-domain injection fix — see
    assets/crmarenapro/graphscale.md, 2026-07-15 diagnosis).

    graph_full re-injected an abstract block (state ids, percentage dumps,
    "ending now fails" advice) as the LAST user message on 60-80% of mid-flow
    turns, which measurably hurt a near-greedy policy. This variant emits at
    most three short elements, all concrete, and never advises against
    finishing:
      1. a trap warning phrased as behaviour ("change the query"), only when
         the current state has a high-fail self-loop;
      2. ONE successful next step from a real past trajectory (success-only
         harvested examples), if available;
      3. a schema-drift caveat so stale column names in the example are
         verified instead of trusted.
    Returns "" (no injection) when it has nothing concrete to say.
    """
    P = (dom_data.get("P_next") or {}).get(state, {})
    tfail = {(t["from"], t["to"]): t for t in dom_data.get("trans_fail", [])}
    base_fail = dom_data.get("base_fail_rate", 0.0) or 0.0

    lines: list[str] = []

    # 1. trap warning: high-fail self-loop on the current state only.
    self_tf = tfail.get((state, state))
    if self_tf and self_tf.get("self_loop") and self_tf.get("fail_rate", 0) > max(0.6, base_fail):
        lines.append(
            "Note: past runs that repeated the same query at this point usually "
            "failed — if the last result was empty or an error, change the query "
            "(different table, columns, or filters) instead of retrying it.")

    header = "[Hint from past successful runs on similar tasks]"
    footer = ("Column names in this org may differ from the example — verify with "
              "<describe> before reusing an exact query. Answer as soon as you "
              "have the information.")

    # 2. one concrete successful next step (ranked non-trap transitions),
    #    budgeted so the header/footer always survive whole.
    if examples and n_shot > 0:
        ex_budget = min(max_example_chars,
                        max_block_chars - len(header) - len(footer)
                        - sum(len(x) for x in lines) - 120)
        if ex_budget > 100:
            ranked = sorted(P.items(), key=lambda x: -x[1])
            next_states = [nxt for nxt, _ in ranked if nxt not in ("END", state, "noise")]
            trans_ex = (examples.get("transitions") or {})
            for nxt in next_states:
                ex = trans_ex.get(f"{state}->{nxt}") or []
                if not ex:
                    continue
                snip = _example_snippet(ex[0], ex_budget)
                # drop the exemplar's final answer — it belongs to a DIFFERENT
                # task instance and invites answer-copying
                cut = snip.find("<respond>")
                if cut > 0:
                    snip = snip[:cut].rstrip() + "\n  [that task's own final answer omitted]"
                lines.append("On a similar task, a successful next step looked like:\n"
                             f"  {snip}")
                break

    if not lines:
        return ""
    return "\n".join([header] + lines + [footer])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transitions", default=str(_HERE / "reports" / "transitions.json"))
    ap.add_argument("--states", default=str(_HERE / "out" / "states.jsonl.gz"))
    ap.add_argument("--domain", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--examples", default=str(_HERE / "reports" / "examples.json"))
    ap.add_argument("--n-shot", type=int, default=0)
    ap.add_argument("--graph-tools", action="store_true",
                    help="render tool-conditioned P(next|state,toolset) block")
    ap.add_argument("--graph-value", action="store_true",
                    help="render value/advantage-ranked tool-action block")
    ap.add_argument("--centroid-meta", default=str(_HERE / "out" / "centroids" / "centroid_meta.json"),
                    help="use centroid top_tool for state names (matches runtime)")
    args = ap.parse_args()

    data = json.loads(Path(args.transitions).read_text())
    if Path(args.centroid_meta).exists():
        names = json.loads(Path(args.centroid_meta).read_text()).get("top_tool", {})
    else:
        names = state_names(Path(args.states))
    examples = None
    if args.n_shot > 0 and Path(args.examples).exists():
        examples = json.loads(Path(args.examples).read_text()).get(args.domain)
    fn = (render_block_value if args.graph_value else
          render_block_tools if args.graph_tools else render_block)
    print(fn(data[args.domain], args.state, names,
             examples=examples, n_shot=args.n_shot))


if __name__ == "__main__":
    main()
