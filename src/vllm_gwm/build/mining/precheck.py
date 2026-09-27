# SPDX-License-Identifier: MIT
"""Stage 6 — runtime centroids + the offline GO/NO-GO gate.

This is the stage that writes the **runtime contract** consumed by
:class:`vllm_gwm.runtime.adapter.GraphAdapter` and
:class:`vllm_gwm.harness.graph_sidecar.Scorer`:

- ``out/centroids/centroids_<dim>.npy`` — per-state centroids, named for the
  embedding dimension that produced them (384 for MiniLM, 4096 for a 4096-d
  embedder), built from the TRAIN rows so centroids <-> discovery labels stay
  consistent,
- ``out/centroids/centroid_meta.json`` — ``state_ids``, ``p90`` (train cosine
  abstain gate), ``top_tool``, ``model``, ``dim``, ``embedding`` (the full
  stage-2 config block), ``tail_chars``, ``max_tool_result_chars``,
  ``max_flow_chars``,
- ``reports/EMBEDDING.json`` — the embedding provenance, standalone so an
  artifact dir can be named from it without parsing the (large) centroid meta,
- ``reports/PRECHECK.md`` — the gates:

  A. Classifier reproducibility — does cosine-NN-to-centroid (the cheap runtime
     method) reproduce the discovery labels on held-out test steps?
  B. Trap reachability — how many test rollouts pass through a flagged trap state.
  C. Behavior floor — baseline repeated-failing-tool-call / immediate-retry rates.

Ported from ``benchmarks/wm/lib/precheck.py``, with the module-level
``WM_WORKDIR`` / ``WM_WATCHED_TOOLS`` / ``WM_EMBED_*`` globals replaced by
explicit arguments and the ``render`` file-loading hack replaced by the plugin's
:mod:`vllm_gwm.harness.render`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vllm_gwm.build.mining.embed import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_TAIL_CHARS,
    model_slug,
)
from vllm_gwm.build.mining.prefix_expand import (
    MAX_FLOW_CHARS,
    MAX_TOOL_RESULT_CHARS,
    read_jsonl,
)
from vllm_gwm.harness import render

TRAP_FAIL = 0.6   # state fail-rate threshold for "trap"
TRAP_MIN_N = 20


def parse_tools(watched_tools: Any) -> set[str]:
    """Accept a comma/space separated string or any iterable of tool names."""
    if not watched_tools:
        return set()
    if isinstance(watched_tools, str):
        names = re.split(r"[,\s]+", watched_tools.strip())
    else:
        names = [str(t) for t in watched_tools]
    return {n for n in names if n}


def embedding_config(
    out_dir: Path,
    dim: int,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_device: str | None = None,
    embed_batch: int | None = None,
    trust_remote_code: bool = False,
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> dict[str, Any]:
    """Full embedding config for the bundle.

    Prefers the sidecar written by stage 2 (authoritative — it describes the run
    that actually produced the vectors); falls back to the values passed in when
    re-running precheck over an older embed with no sidecar.
    """
    # emb_st.config.json sorts before emb_st_test.config.json ("." < "_"), so the
    # train shard — the one the centroids come from — wins.
    for sidecar in sorted(out_dir.glob("emb_st*.config.json")):
        try:
            cfg = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cfg["source"] = f"stage2 sidecar ({sidecar.name})"
        cfg.setdefault("dim", dim)
        return cfg
    return {
        "model": embed_model,
        "model_slug": model_slug(embed_model),
        "dim": dim,
        "window": "tail",
        "tail_chars": int(tail_chars),
        "normalize_embeddings": True,
        "batch_size": int(embed_batch) if embed_batch else None,
        "device": embed_device or None,
        "trust_remote_code": bool(trust_remote_code),
        "source": "build_graph arguments (no stage-2 sidecar found)",
    }


def precheck(
    workdir: str | Path,
    *,
    watched_tools: Any = (),
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_device: str | None = None,
    embed_batch: int | None = None,
    trust_remote_code: bool = False,
    tail_chars: int = DEFAULT_TAIL_CHARS,
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
    max_flow_chars: int = MAX_FLOW_CHARS,
    trap_fail: float = TRAP_FAIL,
    trap_min_n: int = TRAP_MIN_N,
) -> dict[str, Any]:
    """Build the runtime centroids from ``<workdir>/out`` and run the offline gates."""
    work = Path(workdir).resolve()
    out_dir = work / "out"
    reports = work / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    tools = parse_tools(watched_tools)

    X = np.load(out_dir / "emb_st_all.npz")["X"]
    meta = list(read_jsonl(out_dir / "states_all.jsonl.gz"))
    if len(X) != len(meta):
        raise ValueError(f"emb/states length mismatch {len(X)} vs {len(meta)}")
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    split = np.array([m["split"] for m in meta])
    cluster = np.array([str(m["cluster"]) for m in meta])
    domain = np.array([m.get("domain") for m in meta])
    train = split == "train"

    # state -> top tool (for watched-state identification + naming)
    toolcnt: dict = defaultdict(lambda: defaultdict(int))
    for m in meta:
        for t in m.get("tool_names") or []:
            toolcnt[str(m["cluster"])][t] += 1
    top_tool = {s: max(c, key=c.get) for s, c in toolcnt.items() if c}
    watched_states = {s for s, t in top_tool.items() if t in tools}

    # ---- build + save centroids from TRAIN rows (consistent with discovery labels) ----
    cdir = out_dir / "centroids"
    cdir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    cent_rows: list[np.ndarray] = []
    p90: dict[str, float] = {}
    for s in sorted(set(cluster[train])):
        if s == "noise":
            continue
        idx = np.where(train & (cluster == s))[0]
        v = Xn[idx].mean(0)
        v = v / (np.linalg.norm(v) + 1e-9)
        d = 1.0 - Xn[idx] @ v
        ids.append(s)
        cent_rows.append(v)
        p90[s] = float(np.percentile(d, 90))
    if not ids:
        raise ValueError(
            "no non-noise train states discovered; the graph would have zero "
            "centroids (loosen min_cluster_size or collect more rollouts)"
        )
    cents = np.array(cent_rows, dtype=np.float32)
    # Name the matrix for its embedding dimension and record the model that
    # actually produced it: the serve-time classifier re-embeds live flows with
    # the model named here, so a wrong name silently classifies against the
    # wrong space.
    dim = int(cents.shape[1])
    np.save(cdir / f"centroids_{dim}.npy", cents)
    ecfg = embedding_config(
        out_dir, dim, embed_model=embed_model, embed_device=embed_device,
        embed_batch=embed_batch, trust_remote_code=trust_remote_code,
        tail_chars=tail_chars,
    )
    (cdir / "centroid_meta.json").write_text(json.dumps(
        {"state_ids": ids, "p90": p90, "top_tool": top_tool,
         "model": ecfg.get("model", embed_model), "dim": dim,
         "embedding": ecfg,
         "tail_chars": ecfg.get("tail_chars", tail_chars),
         "max_tool_result_chars": max_tool_result_chars,
         "max_flow_chars": max_flow_chars}, indent=2), encoding="utf-8")
    # Standalone copy so the embedding provenance is readable without parsing
    # the (large) centroid meta, and so artifact dirs can be named from it.
    (reports / "EMBEDDING.json").write_text(json.dumps(
        {**ecfg, "n_states": len(ids), "centroids": f"centroids_{dim}.npy"},
        indent=2), encoding="utf-8")
    print(f"[precheck] embedding model={ecfg.get('model')} dim={dim} "
          f"slug={ecfg.get('model_slug')} states={len(ids)} "
          f"(via {ecfg.get('source')})")
    print(f"[precheck] suggested artifact suffix: _{ecfg.get('model_slug')}")
    result: dict[str, Any] = {
        "n_states": len(ids), "dim": dim, "embedding": ecfg,
        "centroids": str(cdir / f"centroids_{dim}.npy"),
        "centroid_meta": str(cdir / "centroid_meta.json"),
    }

    id_index = {s: i for i, s in enumerate(ids)}
    dom_of = {s: s.split(":")[0] for s in ids}

    # ---- A. classifier reproducibility on TEST steps ----
    te = np.where(split == "test")[0]
    if len(te) == 0:
        # Train/valid-only harvest (no held-out test rows yet). Gates B/C need
        # test rollouts too, so emit centroids + a NO-TEST report and stop here.
        lines = [
            "# Offline GO/NO-GO gate — transition-graph WM experiment", "",
            f"Centroids: {len(ids)} states saved -> out/centroids/", "",
            "## No test split", "",
            'No `split=="test"` rows in this build (e.g. a train/valid-only harvest).',
            "Generate trajectories over the held-out test ids and re-harvest to run",
            "gates A (classifier reproducibility), B (trap reachability), C (behavior floor).",
            "", "## VERDICT: REVIEW (centroids built; test-dependent gates skipped)",
        ]
        (reports / "PRECHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        result.update({"verdict": "REVIEW", "gate_a": None, "test_steps": 0,
                       "report": str(reports / "PRECHECK.md")})
        return result

    pred_list: list[str] = []
    for i in te:
        d = domain[i]
        cand = [s for s in ids if dom_of[s] == d]
        if not cand:
            pred_list.append("abstain")
            continue
        sims = Xn[i] @ cents[[id_index[s] for s in cand]].T
        j = int(np.argmax(sims))
        s = cand[j]
        dist = 1.0 - float(sims[j])
        pred_list.append(s if dist <= p90[s] else "abstain")
    pred = np.array(pred_list)
    actual = cluster[te]
    nonnoise = actual != "noise"
    abstain_rate = float((pred == "abstain").mean())
    # abstain counts as a miss
    agree_nonnoise = float((pred[nonnoise] == actual[nonnoise]).mean()) if nonnoise.any() \
        else float("nan")
    fired = pred != "abstain"
    agree_fired = float((pred[fired & nonnoise] == actual[fired & nonnoise]).mean()) \
        if (fired & nonnoise).any() else float("nan")
    wmask = np.array([a in watched_states for a in actual])
    agree_watched = float((pred[wmask] == actual[wmask]).mean()) if wmask.any() \
        else float("nan")

    # ---- B. trap reachability on TEST ----
    trans = json.loads((reports / "transitions.json").read_text(encoding="utf-8"))
    trap_states = set()
    for _dom, dd in trans.items():
        for sf in dd.get("state_fail", []):
            if sf["fail_rate"] >= trap_fail and sf["n_rollouts"] >= trap_min_n:
                trap_states.add(str(sf["state"]))
    # per test rollout (uid): does its state sequence touch a trap state?
    roll_states: dict = defaultdict(set)
    roll_task: dict = {}
    for i in te:
        roll_states[meta[i]["uid"]].add(cluster[i])
        roll_task[meta[i]["uid"]] = meta[i]["task_id"]
    n_roll = len(roll_states)
    n_roll_trap = sum(1 for ss in roll_states.values() if ss & trap_states)
    tasks_trap = {roll_task[u] for u, ss in roll_states.items() if ss & trap_states}
    n_test_tasks = len(set(roll_task.values()))  # distinct held-out test tasks

    # ---- C. behavior floor on test rollouts ----
    rep_fail = imm_retry = n = 0
    test_rollouts = out_dir / "rollouts_test.jsonl.gz"
    if test_rollouts.exists():
        for r in read_jsonl(test_rollouts):
            flow = r.get("conversation_flow") or []
            ai = [e for e in flow if isinstance(e, dict) and e.get("type") == "ai_message"]
            seen_err: set[str] = set()
            prev_tools, prev_out = None, None
            for k, p in enumerate(ai):
                pos = flow.index(p)
                nxt = flow.index(ai[k + 1]) if k + 1 < len(ai) else len(flow)
                tools_at = tuple((tc.get("name") or "")
                                 for tc in (p.get("tool_calls") or []))
                outs = [render.tool_result_outcome(e) for e in flow[pos + 1:nxt]
                        if isinstance(e, dict) and e.get("type") == "tool_result"]
                out = "error" if "error" in outs else ("none" if not outs else "ok")
                n += 1
                # repeated-failing-tool-call: a watched tool re-called after it errored
                for t in tools_at:
                    if t in tools and t in seen_err:
                        rep_fail += 1
                        break
                if prev_tools == tools_at and prev_out == "error":
                    imm_retry += 1
                if out == "error":
                    for t in tools_at:
                        seen_err.add(t)
                prev_tools, prev_out = tools_at, out
    else:
        print(f"[precheck] no {test_rollouts.name}; gate C (behavior floor) skipped")

    go_a = (agree_nonnoise >= 0.70) and (abstain_rate <= 0.40)
    verdict = "GO" if go_a and n_roll_trap > 0 and (rep_fail + imm_retry) > 0 else "REVIEW"
    lines = [
        "# Offline GO/NO-GO gate — transition-graph WM experiment", "",
        f"Centroids: {len(ids)} states saved -> out/centroids/", "",
        "## A. Classifier reproducibility (test steps, cosine-NN vs discovery labels)",
        f"- test steps: {len(te)}; abstain rate: **{abstain_rate:.3f}**",
        f"- non-noise top-1 agreement (abstain=miss): **{agree_nonnoise:.3f}**",
        f"- agreement when it fires: {agree_fired:.3f}",
        f"- watched failure-state agreement: {agree_watched:.3f} (n={int(wmask.sum())})",
        f"- **GATE A: {'PASS' if go_a else 'FAIL'}** (need ≥0.70 agree & ≤0.40 abstain)", "",
        "## B. Trap reachability (test)",
        f"- flagged trap states: {len(trap_states)}",
        f"- test rollouts passing through a trap: {n_roll_trap}/{n_roll} "
        f"({n_roll_trap / max(n_roll, 1):.2f})",
        f"- distinct test tasks reaching a trap: {len(tasks_trap)}/{n_test_tasks}", "",
        "## C. Behavior floor (baseline test rollouts)",
        f"- steps: {n}; repeated-failing-tool-call steps: {rep_fail} "
        f"({rep_fail / max(n, 1):.4f})",
        f"- immediate-retry-after-error steps: {imm_retry} ({imm_retry / max(n, 1):.4f})", "",
        f"## VERDICT: {verdict}",
        "(GO requires A pass, traps reachable, and non-zero baseline failure behavior "
        "to reduce.)",
    ]
    (reports / "PRECHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    result.update({
        "verdict": verdict, "gate_a": bool(go_a), "test_steps": int(len(te)),
        "abstain_rate": abstain_rate, "agree_nonnoise": agree_nonnoise,
        "n_traps": len(trap_states), "trap_rollouts": n_roll_trap,
        "report": str(reports / "PRECHECK.md"),
    })
    return result
