# SPDX-License-Identifier: Apache-2.0
"""Stage 5 — transition graph + success/fail divergence + failure taxonomy.

Reads ``states_all.jsonl.gz`` (per-step cluster assignments from
:mod:`vllm_gwm.build.mining.discover`), reconstructs each rollout's state
sequence, and mines — **per domain**, since states are domain-scoped:

- the transition graph ``P(s'|s)`` (with START/END sentinels),
- outcome-conditioned graphs ``P(s'|s, success)`` vs ``P(s'|s, fail)``,
- **divergence points**: states where the success vs fail next-state
  distributions differ most (total-variation distance, weighted by visit
  frequency) — i.e. where winning and losing trajectories split,
- a **failure taxonomy**: states & transitions whose rollouts fail far above the
  domain base rate (lift), ranked by frequency x lift; self-loops flagged as
  retry/stall,
- **tool-conditioned edges** ``P(next | state, toolset)`` with per-action
  distinct-rollout success rates (the ``tool_edges`` block the runtime scores
  best-of-N candidates with).

Outputs ``reports/transitions.json`` (the runtime contract) +
``reports/WORKFLOW_GRAPH.md``. Ported from
``benchmarks/wm/lib/mine_workflow_graph.py``; stdlib only.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vllm_gwm.build.mining.prefix_expand import read_jsonl


def toolset_key(tools: Any) -> str:
    """Canonical key for a step's tool calls: sorted, de-duplicated, '|'-joined.

    Empty (no tool call at this step) -> ``''``, rendered as a no-op action.
    """
    return "|".join(sorted({t for t in (tools or [])}))


def sequences_by_domain(states_path: str | Path):
    """domain -> {uid -> (ordered [state], ordered [toolset_key], overall_success)};
    plus state->top tools. ``toolset_key[i]`` is the tool action emitted *in* state
    ``state[i]`` (the action that drives the edge state[i]->state[i+1]). START/END
    sentinels carry an empty toolset."""
    rollouts: dict = defaultdict(dict)
    meta: dict = {}
    state_tools: dict = defaultdict(Counter)
    for r in read_jsonl(states_path):
        dom = r.get("domain") or "unknown"
        uid = r["uid"]
        rollouts[dom].setdefault(uid, []).append(
            (r.get("step_idx", 0), str(r.get("cluster")), toolset_key(r.get("tool_names"))))
        meta[uid] = bool(r.get("overall_success"))
        for t in r.get("tool_names") or []:
            state_tools[str(r.get("cluster"))][t] += 1
    out: dict = defaultdict(dict)
    for dom, by_uid in rollouts.items():
        for uid, steps in by_uid.items():
            ordered = sorted(steps)
            seq = ["START"] + [c for _, c, _ in ordered] + ["END"]
            tools = [""] + [tk for _, _, tk in ordered] + [""]
            out[dom][uid] = (seq, tools, meta[uid])
    return out, state_tools


def state_name(s: str, state_tools: dict) -> str:
    if s in ("START", "END", "noise"):
        return s
    top = state_tools.get(s, Counter()).most_common(1)
    return f"{s}({top[0][0]})" if top else s


def tv_distance(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def normalize(counter: Counter) -> dict:
    tot = sum(counter.values())
    return {k: v / tot for k, v in counter.items()} if tot else {}


def mine_domain(seqs: dict) -> dict[str, Any]:
    """Return the per-domain analysis dict for one domain's {uid: (seq, tools, success)}."""
    trans: Counter = Counter()                        # (s,s') -> n
    trans_succ: Counter = Counter()
    trans_fail: Counter = Counter()
    state_visits_rollouts: dict = defaultdict(lambda: [0, 0])  # state -> [through, fail]
    # tool-conditioned edges: (state_a, toolset_key) -> {next_state: n}, plus
    # DISTINCT-ROLLOUT success (not edge counts: a rollout that loops on an action
    # many times must count once, else self-loops get inflated success credit).
    tool_next: dict = defaultdict(Counter)  # (a, tk) -> Counter(next)  [edge-weighted]
    tool_uids: dict = defaultdict(set)      # (a, tk) -> {uid}
    tool_uids_succ: dict = defaultdict(set)  # (a, tk) -> {uid} that succeeded
    n_roll = n_fail = 0
    for uid, (seq, tools, ok) in seqs.items():
        n_roll += 1
        n_fail += int(not ok)
        for s in set(seq):
            if s in ("START", "END"):
                continue
            state_visits_rollouts[s][0] += 1
            state_visits_rollouts[s][1] += int(not ok)
        for i, (a, b) in enumerate(zip(seq, seq[1:], strict=False)):
            trans[(a, b)] += 1
            (trans_fail if not ok else trans_succ)[(a, b)] += 1
            tk = tools[i]                   # tool action emitted in state `a`
            tool_next[(a, tk)][b] += 1
            tool_uids[(a, tk)].add(uid)
            if ok:
                tool_uids_succ[(a, tk)].add(uid)

    base_fail = n_fail / max(n_roll, 1)

    # next-state distributions per source state (overall / success / fail)
    src_overall: dict = defaultdict(Counter)
    src_succ: dict = defaultdict(Counter)
    src_fail: dict = defaultdict(Counter)
    for (a, b), n in trans.items():
        src_overall[a][b] += n
    for (a, b), n in trans_succ.items():
        src_succ[a][b] += n
    for (a, b), n in trans_fail.items():
        src_fail[a][b] += n

    # divergence points: TV(success-next, fail-next) weighted by visits
    divergence = []
    for s in src_overall:
        if s == "START":
            continue
        tv = tv_distance(normalize(src_succ.get(s, Counter())),
                         normalize(src_fail.get(s, Counter())))
        visits = sum(src_overall[s].values())
        divergence.append({"state": s, "tv": tv, "visits": visits, "score": tv * visits})
    divergence.sort(key=lambda d: -d["score"])

    # failure-associated states (lift = fail-rate among rollouts through s / base)
    state_fail = []
    for s, (nr, nf) in state_visits_rollouts.items():
        fr = nf / max(nr, 1)
        state_fail.append({"state": s, "n_rollouts": nr, "fail_rate": fr,
                           "lift": fr / base_fail if base_fail else float("nan")})
    state_fail.sort(key=lambda d: -(d["lift"] * d["n_rollouts"]))

    # failure-associated transitions (incl. self-loops = retry/stall)
    trans_fail_rank = []
    for (a, b), n in trans.items():
        nf = trans_fail.get((a, b), 0)
        fr = nf / n
        trans_fail_rank.append({"from": a, "to": b, "n": n, "fail_rate": fr,
                                "lift": fr / base_fail if base_fail else float("nan"),
                                "self_loop": a == b})
    trans_fail_rank.sort(key=lambda d: -(d["lift"] * d["n"]))

    # tool-conditioned outgoing graph: per source state, group edges by the tool
    # action (toolset_key) used to leave it. Gives P(next | state, toolset),
    # P(toolset | state), and the per-action rollout success rate.
    tool_edges: dict = defaultdict(dict)
    src_action_total: Counter = Counter()   # state -> total distinct-rollout action uses
    for (a, tk), uids in tool_uids.items():
        src_action_total[a] += len(uids)
    for (a, tk), uids in tool_uids.items():
        if a in ("START", "END"):
            continue
        n = len(uids)                                      # distinct rollouts using it
        tool_edges[a][tk] = {
            "tools": tk.split("|") if tk else [],
            "n": n,
            "p_action": n / max(src_action_total[a], 1),   # P(toolset | a), rollout-weighted
            "succ": len(tool_uids_succ[(a, tk)]) / n,       # distinct-rollout success rate
            "next": normalize(tool_next[(a, tk)]),          # P(next | a, toolset)
        }

    return {
        "n_rollouts": n_roll, "base_fail_rate": base_fail,
        "P_next": {s: normalize(c) for s, c in src_overall.items()},
        "divergence": divergence, "state_fail": state_fail,
        "trans_fail": trans_fail_rank,
        "tool_edges": dict(tool_edges),
    }


def mermaid(analysis: dict, top: int = 12) -> list[str]:
    edges = sorted(
        ((a, b, n) for (a, b), n in
         Counter({(t["from"], t["to"]): t["n"] for t in analysis["trans_fail"]}).items()),
        key=lambda e: -e[2])[:top]
    lines = ["```mermaid", "graph LR"]
    for a, b, n in edges:
        sa, sb = a.replace(":", "_").replace("-", "_"), b.replace(":", "_").replace("-", "_")
        lines.append(f"  {sa} -->|{n}| {sb}")
    lines.append("```")
    return lines


def mine_workflow_graph(
    states_path: str | Path,
    out_dir: str | Path,
    *,
    top: int = 8,
) -> dict[str, Any]:
    """Mine the transition graph into ``<out_dir>/{transitions.json,WORKFLOW_GRAPH.md}``."""
    by_dom, state_tools = sequences_by_domain(states_path)
    reports = Path(out_dir)
    reports.mkdir(parents=True, exist_ok=True)

    full = {dom: mine_domain(seqs) for dom, seqs in by_dom.items()}

    # cross-domain failure taxonomy: high lift x frequency transitions.
    worst = []
    for dom, a in full.items():
        for t in a["trans_fail"][:8]:
            if t["n"] >= 40 and t["lift"] >= 1.3:
                tag = "RETRY-LOOP" if t["self_loop"] else ("DEAD-END" if t["to"] == "END" else "")
                worst.append((t["lift"] * t["n"], dom,
                              state_name(t["from"], state_tools),
                              state_name(t["to"], state_tools),
                              t["n"], t["fail_rate"], t["lift"], tag))
    worst.sort(reverse=True)

    lines = ["# Workflow graph & failure mining (A5)", ""]
    lines.append("States are within-domain workflow phases; the graph describes *flow*. "
                 "Outcome-conditioned divergence + failure lift highlight where losing "
                 "trajectories split from winning ones.\n")
    lines += ["## Top failure modes (cross-domain, lift≥1.3, n≥40)",
              "| domain | from → to | n | fail | lift | kind |", "|---|---|---|---|---|---|",
              *[f"| {dom} | {fr} → {to} | {n} | {frate:.2f} | {lift:.2f} | {tag} |"
                for _, dom, fr, to, n, frate, lift, tag in worst[:20]], ""]
    for dom in sorted(by_dom):
        a = full[dom]
        lines += [
            f"## {dom}",
            f"- rollouts: {a['n_rollouts']}, base fail rate: {a['base_fail_rate']:.3f}",
            "", "### Divergence points (success vs fail next-state, weighted)",
            "| state | TV | visits | score |", "|---|---|---|---|",
            *[f"| {d['state']} | {d['tv']:.3f} | {d['visits']} | {d['score']:.1f} |"
              for d in a["divergence"][:top]],
            "", "### Failure-associated states (lift = fail-rate / base)",
            "| state | n_rollouts | fail_rate | lift |", "|---|---|---|---|",
            *[f"| {s['state']} | {s['n_rollouts']} | {s['fail_rate']:.3f} | {s['lift']:.2f} |"
              for s in a["state_fail"][:top]],
            "", "### Failure-associated transitions (★ = self-loop / retry)",
            "| from | to | n | fail_rate | lift |", "|---|---|---|---|---|",
            *[f"| {t['from']} | {t['to']}{' ★' if t['self_loop'] else ''} | {t['n']} "
              f"| {t['fail_rate']:.3f} | {t['lift']:.2f} |"
              for t in a["trans_fail"][:top]],
            "",
            *mermaid(a),
            "",
        ]

    (reports / "WORKFLOW_GRAPH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports / "transitions.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(f"[mine] {len(by_dom)} domains -> {reports}/WORKFLOW_GRAPH.md , transitions.json")
    for dom in sorted(full):
        a = full[dom]
        top_div = a["divergence"][0] if a["divergence"] else {}
        print(f"  {dom}: rollouts={a['n_rollouts']} base_fail={a['base_fail_rate']:.2f} "
              f"top_divergence={top_div.get('state')} (tv={top_div.get('tv', 0):.2f})")
    return {
        "domains": sorted(full),
        "n_domains": len(full),
        "transitions": str(reports / "transitions.json"),
        "report": str(reports / "WORKFLOW_GRAPH.md"),
    }
