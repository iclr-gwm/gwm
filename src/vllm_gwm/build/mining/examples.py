# SPDX-License-Identifier: Apache-2.0
"""Stage 7 — harvest concrete few-shot examples for the graph world model.

Graph advice references opaque cluster ids (e.g. ``teams:5(create_call)``). The
agent has no idea what situation ``teams:5`` is, so the advice is
uninterpretable on its own. This stage harvests, for every discovered state and
for every state->state transition, up to ``per`` concrete examples drawn from
past **successful train** trajectories, so the guidance block can ground each
cluster id in real conversation tails.

Examples are keyed in the **runtime classifier space** (the persisted centroids /
``states_all.jsonl.gz`` cluster ids) — exactly the ``state`` value
``wm_prompt.render_block`` receives at runtime.

Selection: ``split == "train"`` and ``overall_success == True``, ranked by cosine
similarity of the step embedding to its state centroid (most representative
first), deduplicated by normalized text, capped at ``per``.

Output ``reports/examples.json``::

    { "<domain>": {
        "states":           { "teams:5": [ {step_uid, uid, step_idx, cos, text}, ... ] },
        "transitions":      { "teams:5->teams:11": [ ... ] },
        "tool_transitions": { "teams:5": { "create_call": { "teams:11": [ ... ] } } },
        "neg_states":       { "teams:5": [ {step_uid, reason, text}, ... ] } } }

plus ``reports/EXAMPLES.md`` (coverage summary).

Ported from ``benchmarks/wm/lib/harvest_examples.py``; ``neg_states`` (the
contrastive step-labeled negatives read by
``vllm_gwm.harness.harness.Mediator._neg_examples_block``) folds in the separate
``mine_neg_examples.py`` miner so one stage writes the whole examples contract.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vllm_gwm.build.mining.prefix_expand import read_jsonl

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip().lower()


def find_centroids(centroids_dir: str | Path) -> Path:
    """The centroid matrix, whatever embedding dimension built it."""
    hits = sorted(Path(centroids_dir).glob("centroids_*.npy"))
    if not hits:
        raise FileNotFoundError(f"no centroids_*.npy under {centroids_dir}")
    return hits[0]


def harvest_examples(
    *,
    states_path: str | Path,
    steps_path: str | Path,
    emb_path: str | Path,
    emb_meta_path: str | Path,
    centroids_dir: str | Path,
    out_path: str | Path,
    md_path: str | Path,
    per: int = 20,
    neg_per_state: int = 8,
) -> dict[str, Any]:
    """Harvest per-state / per-transition exemplars into ``out_path``."""
    # 1) centroids + state ids ------------------------------------------------
    cdir = Path(centroids_dir)
    cents = np.load(find_centroids(cdir)).astype(np.float32)
    cmeta = json.loads((cdir / "centroid_meta.json").read_text(encoding="utf-8"))
    state_ids = cmeta["state_ids"]
    top_tool = cmeta.get("top_tool", {})
    cent_of = {s: cents[i] for i, s in enumerate(state_ids)}
    valid_states = set(state_ids)            # the runtime classifier space

    # 2) embeddings aligned with meta order ----------------------------------
    X = np.load(emb_path)["X"].astype(np.float32)
    # normalize for cosine (centroids were saved from normalized embeds)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    # step_uid can repeat across splits in the concatenated meta; we only harvest
    # train, so index train rows (and fall back to any row if no train one exists).
    emb_row: dict[str, int] = {}
    for i, r in enumerate(read_jsonl(emb_meta_path)):
        suid = r["step_uid"]
        if r.get("split") == "train" or suid not in emb_row:
            emb_row[suid] = i

    # 3) per-step cluster / outcome / split (centroid space) -----------------
    info: dict[str, dict] = {}
    for r in read_jsonl(states_path):
        info[r["step_uid"]] = {
            "cluster": str(r.get("cluster")),
            "domain": r.get("domain"),
            "uid": r.get("uid"),
            "step_idx": r.get("step_idx", 0),
            "split": r.get("split"),
            "success": bool(r.get("overall_success")),
            "obs_outcome": r.get("obs_outcome"),
            "tool_names": r.get("tool_names") or [],
            "toolset": "|".join(sorted({t for t in (r.get("tool_names") or [])})),
        }

    # 4) text (train) ---------------------------------------------------------
    text_of: dict[str, str] = {}
    for r in read_jsonl(steps_path):
        text_of[r["step_uid"]] = r.get("text") or ""

    # ---- per-state candidates: train + success + has-centroid --------------
    state_cands: dict[str, list] = defaultdict(list)   # state -> [(cos, step_uid)]
    for suid, d in info.items():
        if d["split"] != "train" or not d["success"]:
            continue
        st = d["cluster"]
        if st not in valid_states:           # only states the classifier can emit
            continue
        if suid not in emb_row or suid not in text_of:
            continue
        cos = float(Xn[emb_row[suid]] @ cent_of[st])
        state_cands[st].append((cos, suid))

    def pick(cands: list, cap: int) -> list[dict]:
        out, seen = [], set()
        for cos, suid in sorted(cands, key=lambda x: -x[0]):
            t = text_of.get(suid, "")
            k = norm(t)[:4000]
            if not k or k in seen:
                continue
            seen.add(k)
            d = info[suid]
            out.append({"step_uid": suid, "uid": d["uid"], "step_idx": d["step_idx"],
                        "cos": round(cos, 4), "text": t})
            if len(out) >= cap:
                break
        return out

    states_by_dom: dict[str, dict] = defaultdict(dict)
    for st, cands in state_cands.items():
        states_by_dom[st.split(":")[0]][st] = pick(cands, per)

    # ---- per-transition candidates -----------------------------------------
    # reconstruct each train+success rollout's ordered state sequence, capture the
    # *landing* (to) step as the example for edge (from->to). Rank by cos of the
    # to-step to the to-centroid.
    seqs: dict[str, list] = defaultdict(list)   # uid -> [(step_idx, step_uid, cluster)]
    for suid, d in info.items():
        if d["split"] != "train" or not d["success"]:
            continue
        seqs[d["uid"]].append((d["step_idx"], suid, d["cluster"]))

    trans_cands: dict[tuple, list] = defaultdict(list)  # (dom, "from->to") -> [(cos, suid)]
    for steps in seqs.values():
        steps.sort()
        for (_i1, _s1, c1), (_i2, s2, c2) in zip(steps, steps[1:], strict=False):
            if c1 not in valid_states or c2 not in valid_states:
                continue
            if s2 not in emb_row or s2 not in text_of:
                continue
            cos = float(Xn[emb_row[s2]] @ cent_of[c2])
            trans_cands[(c2.split(":")[0], f"{c1}->{c2}")].append((cos, s2))

    trans_by_dom: dict[str, dict] = defaultdict(dict)
    for (dom, key), cands in trans_cands.items():
        trans_by_dom[dom][key] = pick(cands, per)

    # ---- per-(state, toolset, next) candidates -----------------------------
    # tool-conditioned grounding for the graph_tools arm: capture the *landing*
    # step of an edge, keyed by the tool action emitted in the SOURCE state. Lets
    # the block show "from <state>, calling [T] led to <next>" with a real tail.
    tooltrans_cands: dict[tuple, list] = defaultdict(list)  # (dom, from, tk, to)
    for steps in seqs.values():
        # steps already sorted above as [(step_idx, step_uid, cluster)]
        for (_i1, s1, c1), (_i2, s2, c2) in zip(steps, steps[1:], strict=False):
            if c1 not in valid_states or c2 not in valid_states:
                continue
            if s2 not in emb_row or s2 not in text_of:
                continue
            tk = info[s1].get("toolset", "")
            cos = float(Xn[emb_row[s2]] @ cent_of[c2])
            tooltrans_cands[(c2.split(":")[0], c1, tk, c2)].append((cos, s2))

    tooltrans_by_dom: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    for (dom, frm, tk, to), cands in tooltrans_cands.items():
        tooltrans_by_dom[dom][frm].setdefault(tk, {})[to] = pick(cands, per)

    # ---- contrastive negatives: steps whose observation ERRORED ------------
    # step-level label (NOT trajectory-level), grouped by the step's mined state.
    neg_by_dom: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    if neg_per_state > 0:
        seen_neg: Counter = Counter()
        for suid, d in sorted(info.items()):
            if str(d.get("obs_outcome")) != "error":
                continue
            st = d["cluster"]
            if st not in valid_states:
                continue
            dom = st.split(":")[0]
            if seen_neg[(dom, st)] >= neg_per_state:
                continue
            txt = text_of.get(suid, "")
            if not txt:
                continue
            tool_names = d.get("tool_names") or []
            reason = (f"[FAILED: tool {tool_names} errored at this state]"
                      if tool_names else "[FAILED here]")
            neg_by_dom[dom][st].append({"step_uid": suid, "reason": reason, "text": txt})
            seen_neg[(dom, st)] += 1

    # ---- assemble + write ---------------------------------------------------
    out: dict[str, dict] = {}
    domains = (set(states_by_dom) | set(trans_by_dom) | set(tooltrans_by_dom)
               | set(neg_by_dom))
    for dom in sorted(domains):
        tt = {frm: {tk: dict(tos) for tk, tos in acts.items()}
              for frm, acts in tooltrans_by_dom.get(dom, {}).items()}
        out[dom] = {"states": states_by_dom.get(dom, {}),
                    "transitions": trans_by_dom.get(dom, {}),
                    "tool_transitions": tt,
                    "neg_states": dict(neg_by_dom.get(dom, {}))}
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # ---- coverage report ----------------------------------------------------
    n_states_covered = sum(1 for d in out.values() for v in d["states"].values() if v)
    n_trans_covered = sum(1 for d in out.values() for v in d["transitions"].values() if v)
    n_neg = sum(len(v) for d in out.values() for v in d["neg_states"].values())
    lines = ["# Harvested few-shot examples (graph grounding)", "",
             f"- per-key cap: {per}",
             "- selection: train split, overall_success=True, ranked by "
             "cosine-to-centroid, deduped", ""]
    lines += [f"- states with >=1 example: {n_states_covered} / {len(valid_states)} "
              "centroid states",
              f"- transitions with >=1 example: {n_trans_covered}",
              f"- contrastive negatives (obs_outcome=error): {n_neg} "
              f"(cap {neg_per_state}/state)", ""]
    lines += ["| domain | states (cov/total) | mean ex/state | transitions | mean ex/trans |",
              "|---|---|---|---|---|"]
    for dom in sorted(out):
        s_map = out[dom]["states"]
        t_map = out[dom]["transitions"]
        dom_states = [s for s in valid_states if s.split(":")[0] == dom]
        cov = sum(1 for v in s_map.values() if v)
        msl = sum(len(v) for v in s_map.values()) / max(len(s_map), 1)
        mtl = sum(len(v) for v in t_map.values()) / max(len(t_map), 1)
        lines.append(f"| {dom} | {cov}/{len(dom_states)} | {msl:.1f} | {len(t_map)} "
                     f"| {mtl:.1f} |")
    # show one grounded example for the most-populated (most opaque) state, so the
    # spot check is benchmark-agnostic rather than tied to a fixed state id.
    spot_state, spot_ex = None, []
    for dom in sorted(out):
        for st, exs in out[dom]["states"].items():
            if len(exs) > len(spot_ex):
                spot_state, spot_ex = st, exs
    if spot_state:
        lines += ["", f"## Spot check: {spot_state} "
                      f"(modal tool = {top_tool.get(spot_state, '?')})", ""]
        lines += ["```", spot_ex[0]["text"][-1200:], "```"]
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[harvest] states covered {n_states_covered}/{len(valid_states)}; "
          f"negatives {n_neg}; wrote {out_file} and {md_path}")
    return {
        "domains": sorted(out),
        "states_covered": n_states_covered,
        "n_centroid_states": len(valid_states),
        "transitions_covered": n_trans_covered,
        "negatives": n_neg,
        "out": str(out_file),
        "md": str(md_path),
    }
