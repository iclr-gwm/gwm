# SPDX-License-Identifier: Apache-2.0
"""Stage 4 — UMAP -> HDBSCAN state discovery + cluster diagnostics.

Fits the reducer + clusterer on the **train split only** (no leakage from
valid/test), then assigns every step to a state. In ``within_domain`` mode each
domain is clustered separately and labels become ``"<domain>:<local_id>"`` — the
state-id space the whole runtime (centroids, transitions, examples) is keyed on.

Two outputs:

- ``states_all.jsonl.gz`` — ``{step_uid, cluster, ...meta}`` for every step.
- ``states_report.md`` — cluster profiles (size, success-rate, top tools, mean
  step position, dominant obs_outcome/domain) AND the "what-do-clusters-encode"
  diagnostic: normalized mutual information between cluster id and
  ``{domain, step-position, outcome}``. If domain/position MI dominates outcome
  MI, the discovered "state" is mostly trajectory position — flagged so we can
  switch to within-domain clustering / position residualization.

Ported from ``benchmarks/wm/lib/discover_states.py``. The ``--save-centroids``
side path of the original is intentionally *not* ported: centroids are written by
:mod:`vllm_gwm.build.mining.precheck` alone, so there is exactly one writer of
the ``out/centroids`` contract.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vllm_gwm.build.mining._deps import lazy_import

DEFAULT_N_COMPONENTS = 10
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_CLUSTER_SIZE = 50
DEFAULT_SEED = 17


def read_meta(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def umap_hdbscan_labels(
    X: np.ndarray,
    train_mask: np.ndarray,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = None,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """UMAP+HDBSCAN fit on the train rows of a subset; labels for all rows (-1 = noise)."""
    umap = lazy_import("umap")
    hdbscan = lazy_import("hdbscan")

    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        metric="cosine", random_state=seed)
    reducer.fit(X[train_mask])
    Z = reducer.transform(X)
    # UMAP.transform can emit non-finite rows for points that map to
    # disconnected regions of the fitted manifold (intermittent, subset-
    # dependent); HDBSCAN's prediction KD-tree then aborts on NaN. Sanitize
    # to 0 — a no-op when Z is all-finite, so ratios that already built are
    # bit-identical; only the pathological rows are rescued.
    if not np.isfinite(Z).all():
        n_bad = int((~np.isfinite(Z).all(axis=1)).sum())
        print(f"[discover] WARN: UMAP produced {n_bad} non-finite row(s); "
              "sanitizing to 0 before HDBSCAN")
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                         metric="euclidean", prediction_data=True)
    cl.fit(Z[train_mask])
    lab = np.full(len(X), -1, dtype=int)
    lab[train_mask] = cl.labels_
    if (~train_mask).any():
        pred, _ = hdbscan.approximate_predict(cl, Z[~train_mask])
        lab[~train_mask] = pred
    return lab


def discover_states(
    emb_path: str | Path,
    out_states: str | Path,
    *,
    meta_path: str | Path | None = None,
    out_report: str | Path | None = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = None,
    within_domain: bool = True,
    seed: int = DEFAULT_SEED,
    label_fn: Any = None,
) -> dict[str, Any]:
    """Cluster the embeddings into workflow states.

    ``label_fn`` overrides the UMAP+HDBSCAN labeller (same signature as
    :func:`umap_hdbscan_labels`); the pipeline never sets it, it exists so the
    DAG can be exercised without the heavy clustering deps.
    """
    emb = Path(emb_path)
    meta_file = Path(meta_path) if meta_path else emb.with_suffix(".meta.jsonl")
    X = np.load(emb)["X"]
    meta = read_meta(meta_file)
    if len(meta) != len(X):
        raise ValueError(f"meta/emb length mismatch {len(meta)} vs {len(X)}")

    labeller = label_fn or umap_hdbscan_labels

    def cluster_subset(Xsub: np.ndarray, train_sub: np.ndarray) -> np.ndarray:
        return labeller(Xsub, train_sub, n_components=n_components,
                        n_neighbors=n_neighbors, min_cluster_size=min_cluster_size,
                        min_samples=min_samples, seed=seed)

    split = np.array([m.get("split") for m in meta])
    train_mask = split == "train"
    n_train = int(train_mask.sum())
    print(f"[discover] {len(X)} steps ({n_train} train); UMAP {X.shape[1]}->{n_components}")
    if n_train == 0:
        raise ValueError(
            "no rows with split=='train'; state discovery fits on the train split "
            "only (tag rollouts with split='train' before building)"
        )

    domains = np.array([m.get("domain") for m in meta])
    str_labels = np.empty(len(X), dtype=object)
    if within_domain:
        for d in sorted(set(domains)):
            idx = np.where(domains == d)[0]
            if not train_mask[idx].any():
                str_labels[idx] = "noise"
                continue
            sub = cluster_subset(X[idx], train_mask[idx])
            str_labels[idx] = [f"{d}:{c}" if c >= 0 else "noise" for c in sub]
    else:
        glob = cluster_subset(X, train_mask)
        str_labels[:] = [str(c) if c >= 0 else "noise" for c in glob]

    # train view used for profiling + MI
    tr_meta = [m for m, t in zip(meta, train_mask, strict=True) if t]
    cl = list(str_labels[train_mask])
    n_clusters = len({c for c in cl if c != "noise"})
    noise = float(np.mean([c == "noise" for c in cl]))

    # ---- cluster profiles (train) ----
    prof: dict[str, dict] = defaultdict(lambda: {"n": 0, "succ": 0, "pos": [],
                                                 "tools": Counter(), "obs": Counter(),
                                                 "dom": Counter()})
    for m, c in zip(tr_meta, cl, strict=True):
        p = prof[c]
        p["n"] += 1
        p["succ"] += int(bool(m.get("overall_success")))
        if m.get("n_steps"):
            p["pos"].append((m.get("step_idx", 1)) / max(m["n_steps"], 1))
        for tn in m.get("tool_names") or []:
            p["tools"][tn] += 1
        p["obs"][m.get("obs_outcome")] += 1
        p["dom"][m.get("domain")] += 1

    # ---- MI diagnostic (train) ----
    nmi = lazy_import("sklearn.metrics").normalized_mutual_info_score
    dom = [m.get("domain") for m in tr_meta]
    posb = [int(round(5 * (m.get("step_idx", 1) / max(m.get("n_steps", 1), 1))))
            for m in tr_meta]
    outc = [int(bool(m.get("overall_success"))) for m in tr_meta]
    mi = {
        "domain": float(nmi(dom, cl)),
        "position": float(nmi(posb, cl)),
        "outcome": float(nmi(outc, cl)),
    }
    # In within-domain mode the domain is baked into the label, so only compare
    # position vs outcome; globally, a high domain MI is itself the confound.
    if within_domain:
        state_is_position = mi["position"] >= mi["outcome"]
    else:
        state_is_position = (mi["domain"] >= mi["outcome"] or mi["position"] >= mi["outcome"])

    # ---- write states ----
    states_out = Path(out_states)
    states_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(states_out, "wt", encoding="utf-8") as f:
        for m, c in zip(meta, str_labels, strict=True):
            rec = dict(m)
            rec["cluster"] = c
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- report ----
    lines = [
        f"# A3 state discovery — {emb.name}", "",
        f"- steps: {len(X)} ({n_train} train); UMAP dim {n_components}, "
        f"HDBSCAN min_cluster_size={min_cluster_size}",
        f"- **{n_clusters} clusters**, noise fraction {noise:.3f}", "",
        "## What do clusters encode? (normalized MI, train)",
        f"- domain:   {mi['domain']:.3f}",
        f"- position: {mi['position']:.3f}",
        f"- outcome:  {mi['outcome']:.3f}",
        f"- **{'WARNING: state ~ domain/position, not control state' if state_is_position else 'OK: outcome MI is competitive'}**",
        "", "## Cluster profiles (train)",
        "| cluster | n | success | mean_pos | top_tools | obs | top_domain |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in sorted(prof):
        p = prof[c]
        mean_pos = float(np.mean(p["pos"])) if p["pos"] else float("nan")
        top_tools = ", ".join(f"{t}:{n}" for t, n in p["tools"].most_common(3))
        obs = ", ".join(f"{o}:{n}" for o, n in p["obs"].most_common())
        dom_top = ", ".join(f"{d}:{n}" for d, n in p["dom"].most_common(2))
        lines.append(f"| {c} | {p['n']} | {p['succ'] / max(p['n'], 1):.2f} | {mean_pos:.2f} "
                     f"| {top_tools} | {obs} | {dom_top} |")
    report_out = Path(out_report) if out_report else emb.parent / "states_report.md"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[discover] -> {states_out} , {report_out}")
    return {
        "steps": int(len(X)), "train": n_train, "n_clusters": n_clusters,
        "noise_fraction": noise, "mi": mi, "state_is_position": bool(state_is_position),
        "states": str(states_out), "report": str(report_out),
    }
