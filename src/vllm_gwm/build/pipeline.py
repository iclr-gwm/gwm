# SPDX-License-Identifier: MIT
"""Graph-bundle build pipeline (the in-plugin port of ``benchmarks/wm/build_all.sh``).

Runs the mining DAG over a working directory and leaves behind a bundle that
:class:`vllm_gwm.runtime.adapter.GraphAdapter` validates and
:class:`vllm_gwm.harness.graph_sidecar.Scorer` serves::

    <workdir>/out/rollouts.jsonl.gz          (input, stage 0 — NOT produced here)
    <workdir>/out/rollouts_test.jsonl.gz     (optional held-out split)
    <workdir>/out/steps{,_test}.jsonl.gz     1 prefix_expand
    <workdir>/out/emb_st{,_test}.npz         2 embed   (+ .meta.jsonl/.config.json)
    <workdir>/out/emb_st_all.npz             3 concat  (+ .meta.jsonl)
    <workdir>/out/states_all.jsonl.gz        4 discover(+ reports/STATES.md)
    <workdir>/reports/transitions.json       5 mine    (+ WORKFLOW_GRAPH.md)
    <workdir>/out/centroids/                 6 precheck(+ reports/{PRECHECK,EMBEDDING})
    <workdir>/reports/examples.json          7 examples(+ EXAMPLES.md)
    <workdir>/MANIFEST.json

**Stage 0 (harvest) is not part of this pipeline.** Turning a benchmark's raw
trajectories into ``conversation_flow`` rollouts is benchmark-specific; the
pipeline starts from ``out/rollouts*.jsonl.gz`` (or, with ``reuse_steps``, from
``out/steps*.jsonl.gz``) already present in the workdir. :func:`ingest_rollouts`
is the one bridge provided: it normalizes the plugin's own collected rollouts
(:class:`vllm_gwm.collect.store.RolloutStore` snapshots) into that input shape.

An ``out/rollouts_all.jsonl.gz`` with ``split`` tags is split into
train/test first (build_all.sh stage 0b).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vllm_gwm.build.mining import (
    concat_embeddings,
    discover_states,
    embed_steps,
    expand_steps,
    harvest_examples,
    mine_workflow_graph,
    precheck,
    split_by_tag,
)
from vllm_gwm.build.mining import discover as _discover
from vllm_gwm.build.mining.embed import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_TAIL_CHARS,
)
from vllm_gwm.build.mining.prefix_expand import read_jsonl

STAGES: tuple[str, ...] = (
    "prefix_expand",
    "embed",
    "concat",
    "discover",
    "mine",
    "precheck",
    "examples",
)

DEFAULT_DISCOVER_ARGS: dict[str, Any] = {
    "within_domain": True,
    "min_cluster_size": _discover.DEFAULT_MIN_CLUSTER_SIZE,
    "n_components": _discover.DEFAULT_N_COMPONENTS,
}


class BuildError(RuntimeError):
    """Raised when the build cannot proceed (missing inputs, bad stage name)."""


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_discover_args(spec: Any) -> dict[str, Any]:
    """Normalize ``discover_args`` into :func:`discover_states` keyword arguments.

    Accepts ``None`` (defaults), a dict of kwargs, or a shell-style string / argv
    list so the ``DISCOVER_ARGS`` knob of build_all.sh keeps working, e.g.
    ``"--within-domain --min-cluster-size 50 --n-components 10"``.
    """
    args = dict(DEFAULT_DISCOVER_ARGS)
    if spec is None:
        return args
    if isinstance(spec, dict):
        args.update(spec)
        return args
    argv = spec.split() if isinstance(spec, str) else [str(a) for a in spec]
    ap = argparse.ArgumentParser(prog="discover", add_help=False)
    ap.add_argument("--within-domain", dest="within_domain", action="store_true",
                    default=None)
    ap.add_argument("--no-within-domain", dest="within_domain", action="store_false")
    ap.add_argument("--n-components", dest="n_components", type=int)
    ap.add_argument("--n-neighbors", dest="n_neighbors", type=int)
    ap.add_argument("--min-cluster-size", dest="min_cluster_size", type=int)
    ap.add_argument("--min-samples", dest="min_samples", type=int)
    ap.add_argument("--seed", dest="seed", type=int)
    parsed, unknown = ap.parse_known_args(argv)
    if unknown:
        raise BuildError(f"unrecognized discover args: {unknown}")
    args.update({k: v for k, v in vars(parsed).items() if v is not None})
    return args


def ingest_rollouts(
    input_path: str | Path,
    workdir: str | Path,
    *,
    default_domain: str = "unknown",
    default_split: str = "train",
) -> dict[str, Any]:
    """Normalize a rollout jsonl into ``<workdir>/out/rollouts{,_test}.jsonl.gz``.

    Accepts both the mining shape (``uid`` / ``conversation_flow`` /
    ``overall_success``) and the plugin's collection shape (``id`` /
    ``episode_id`` / ``flow`` / ``success``). Collection stores append one row per
    turn, so rows are de-duplicated per episode keeping the finalized / longest
    flow. Untagged rows default to ``split=default_split`` because state discovery
    fits on the train split only.
    """
    out_dir = Path(workdir) / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, dict] = {}
    order: list[str] = []
    for i, rec in enumerate(read_jsonl(input_path)):
        flow = rec.get("conversation_flow") or rec.get("flow") or []
        key = str(rec.get("uid") or rec.get("episode_id") or rec.get("id") or f"r{i}")
        success = rec.get("overall_success")
        if success is None:
            success = rec.get("success")
        norm = {
            "uid": key,
            "task_id": str(rec.get("task_id") or rec.get("episode_id") or key),
            "domain": rec.get("domain") or default_domain,
            "split": rec.get("split") or default_split,
            "temp": rec.get("temp"),
            "conversation_flow": flow,
            "overall_success": bool(success),
            "pass_rate": rec.get("pass_rate"),
            "finalized": bool(rec.get("finalized")),
        }
        prev = best.get(key)
        if prev is None:
            order.append(key)
            best[key] = norm
            continue
        better = (norm["finalized"], len(flow)) >= (
            prev["finalized"], len(prev["conversation_flow"]))
        if better:
            best[key] = norm

    n_train = n_test = 0
    train_path = out_dir / "rollouts.jsonl.gz"
    test_path = out_dir / "rollouts_test.jsonl.gz"
    with gzip.open(train_path, "wt", encoding="utf-8") as tr, \
            gzip.open(test_path, "wt", encoding="utf-8") as te:
        for key in order:
            rec = dict(best[key])
            rec.pop("finalized", None)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if rec["split"] == "test":
                te.write(line)
                n_test += 1
            else:
                tr.write(line)
                n_train += 1
    if n_test == 0:
        test_path.unlink()
    print(f"[ingest] {n_train} train + {n_test} test rollouts -> {out_dir}")
    if n_train == 0:
        raise BuildError(f"no non-test rollouts found in {input_path}")
    return {"train": n_train, "test": n_test, "workdir": str(Path(workdir).resolve())}


def write_manifest(workdir: Path, *, adapter: str = "", extra: dict | None = None) -> Path:
    """Write ``MANIFEST.json`` with sha256s of the runtime-contract files."""
    from vllm_gwm.runtime.adapter import GraphAdapter, find_centroids

    rel_files = ["reports/transitions.json", "out/centroids/centroid_meta.json"]
    cent = find_centroids(workdir)
    if cent is not None:
        rel_files.append(str(cent.relative_to(workdir.resolve())))
    manifest: dict[str, Any] = {
        "adapter": adapter or workdir.name,
        "version": datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ"),
        "built_by": "vllm_gwm.build.pipeline",
        "files": {
            rel: {"sha256": GraphAdapter.sha256_file(workdir / rel)}
            for rel in rel_files
            if (workdir / rel).exists()
        },
    }
    if extra:
        manifest.update(extra)
    path = workdir / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_graph(
    workdir: str | Path,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_device: str | None = None,
    embed_batch: int = DEFAULT_BATCH_SIZE,
    trust_remote_code: bool = False,
    reuse_steps: bool = False,
    watched_tools: Any = (),
    discover_args: Any = None,
    stages: list[str] | tuple[str, ...] | None = None,
    tail_chars: int = DEFAULT_TAIL_CHARS,
    examples_per: int = 20,
    show_progress_bar: bool = False,
) -> dict[str, Any]:
    """Run the mining DAG in ``workdir`` and return a per-stage result dict.

    ``workdir`` must already contain ``out/rollouts.jsonl.gz`` (stage 0 harvest is
    benchmark-specific — see :func:`ingest_rollouts`) or, with
    ``reuse_steps=True``, ``out/steps.jsonl.gz``.

    ``reuse_steps`` mirrors ``WM_REUSE_STEPS``: stage 1 is skipped, the existing
    ``out/steps*.jsonl.gz`` are asserted present and their md5 printed (so a
    re-embed with a different ``embed_model`` provably reuses the same text).
    ``stages`` selects a subset of :data:`STAGES` (order is always the DAG order).
    """
    work = Path(workdir).resolve()
    out_dir = work / "out"
    reports = work / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    want = list(STAGES) if stages is None else [s for s in STAGES if s in set(stages)]
    unknown = sorted(set(stages or ()) - set(STAGES))
    if unknown:
        raise BuildError(f"unknown stage(s) {unknown}; known stages: {list(STAGES)}")

    dargs = parse_discover_args(discover_args)
    roll_train = out_dir / "rollouts.jsonl.gz"
    roll_test = out_dir / "rollouts_test.jsonl.gz"
    roll_all = out_dir / "rollouts_all.jsonl.gz"
    steps_train = out_dir / "steps.jsonl.gz"
    steps_test = out_dir / "steps_test.jsonl.gz"
    emb_train = out_dir / "emb_st.npz"
    emb_test = out_dir / "emb_st_test.npz"
    emb_all = out_dir / "emb_st_all.npz"
    states_all = out_dir / "states_all.jsonl.gz"

    started = datetime.now(tz=UTC)
    print(f"[build] workdir={work} embed={embed_model} device={embed_device} "
          f"reuse_steps={int(bool(reuse_steps))} start={started:%H:%M:%SZ}")
    results: dict[str, Any] = {"workdir": str(work), "stages": {}}
    total = len(STAGES)

    def stage_no(name: str) -> int:
        return STAGES.index(name) + 1

    # ---- stage 1: prefix_expand ------------------------------------------
    if "prefix_expand" in want:
        print(f"[build] {stage_no('prefix_expand')}/{total} prefix_expand")
        if reuse_steps:
            existing = sorted(out_dir.glob("steps*.jsonl.gz"))
            if not steps_train.exists():
                raise BuildError(
                    f"reuse_steps=True but {steps_train} is missing; run the "
                    "pipeline without reuse_steps to generate it"
                )
            digests = {p.name: _md5(p) for p in existing}
            for name, digest in digests.items():
                print(f"[build] reusing {name} md5={digest}")
            results["stages"]["prefix_expand"] = {
                "reused": list(digests), "md5": digests,
            }
        else:
            if roll_all.exists() and not roll_train.exists():
                print("[build] 0b split by 'split' tag -> rollouts{,_test}.jsonl.gz")
                if split_by_tag(roll_all, roll_train, roll_test)["test"] == 0:
                    roll_test.unlink()
            if not roll_train.exists():
                raise BuildError(
                    f"no rollouts at {roll_train}: stage 0 (harvest) is "
                    "benchmark-specific and is not part of this pipeline — write "
                    "out/rollouts.jsonl.gz (see ingest_rollouts) first"
                )
            st = {"train": expand_steps(roll_train, steps_train)}
            if roll_test.exists():
                st["test"] = expand_steps(roll_test, steps_test)
            results["stages"]["prefix_expand"] = st

    # ---- stage 2: embed ---------------------------------------------------
    if "embed" in want:
        print(f"[build] {stage_no('embed')}/{total} embed on {embed_device} "
              f"model={embed_model}")
        if not steps_train.exists():
            raise BuildError(f"missing {steps_train}; run the prefix_expand stage first")
        kw = dict(model=embed_model, device=embed_device, batch_size=embed_batch,
                  tail_chars=tail_chars, trust_remote_code=trust_remote_code,
                  show_progress_bar=show_progress_bar)
        emb = {"train": embed_steps(steps_train, emb_train, **kw)}
        if steps_test.exists():
            emb["test"] = embed_steps(steps_test, emb_test, **kw)
        results["stages"]["embed"] = emb

    # ---- stage 3: concat --------------------------------------------------
    if "concat" in want:
        print(f"[build] {stage_no('concat')}/{total} concat train+test -> emb_st_all")
        shards = [emb_train] + ([emb_test] if emb_test.exists() else [])
        results["stages"]["concat"] = concat_embeddings(shards, emb_all)

    # ---- stage 4: discover ------------------------------------------------
    if "discover" in want:
        print(f"[build] {stage_no('discover')}/{total} discover_states ({dargs})")
        results["stages"]["discover"] = discover_states(
            emb_all, states_all, out_report=reports / "STATES.md", **dargs)

    # ---- stage 5: mine ----------------------------------------------------
    if "mine" in want:
        print(f"[build] {stage_no('mine')}/{total} mine_workflow_graph")
        results["stages"]["mine"] = mine_workflow_graph(states_all, reports)

    # ---- stage 6: precheck ------------------------------------------------
    if "precheck" in want:
        print(f"[build] {stage_no('precheck')}/{total} precheck")
        results["stages"]["precheck"] = precheck(
            work, watched_tools=watched_tools, embed_model=embed_model,
            embed_device=embed_device, embed_batch=embed_batch,
            trust_remote_code=trust_remote_code, tail_chars=tail_chars)

    # ---- stage 7: harvest_examples ---------------------------------------
    if "examples" in want:
        print(f"[build] {stage_no('examples')}/{total} harvest_examples")
        results["stages"]["examples"] = harvest_examples(
            states_path=states_all, steps_path=steps_train, emb_path=emb_all,
            emb_meta_path=emb_all.with_suffix(".meta.jsonl"),
            centroids_dir=out_dir / "centroids",
            out_path=reports / "examples.json", md_path=reports / "EXAMPLES.md",
            per=examples_per)

    if (reports / "transitions.json").exists():
        embedding = ((results["stages"].get("precheck") or {}).get("embedding")) or {}
        results["manifest"] = str(write_manifest(
            work, extra={"embedding": embedding} if embedding else None))
    done = datetime.now(tz=UTC)
    print(f"[build] DONE {done:%H:%M:%SZ} ({(done - started).total_seconds():.0f}s) "
          f"-> {work}/{{out,reports}}")
    results["elapsed_s"] = (done - started).total_seconds()
    return results


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def main(argv: list[str] | None = None) -> int:
    """``python -m vllm_gwm.build.pipeline`` — build a bundle from a workdir/jsonl.

    Flag defaults mirror the ``WM_*`` knobs of ``benchmarks/wm/build_all.sh``
    (``WM_EMBED_MODEL``, ``WM_EMBED_BATCH``, ``WM_EMBED_TRUST_REMOTE_CODE``,
    ``EMB_DEVICE``, ``WM_REUSE_STEPS``, ``WM_WATCHED_TOOLS``, ``DISCOVER_ARGS``)
    so existing driver scripts keep working; nothing is read at import time.
    """
    ap = argparse.ArgumentParser(prog="vllm_gwm.build.pipeline", description=main.__doc__)
    ap.add_argument("--workdir", default=os.getenv("WM_WORKDIR", ""),
                    help="build directory holding out/ and reports/")
    ap.add_argument("--input", default="",
                    help="rollouts jsonl(.gz) to ingest into <workdir>/out/ first")
    ap.add_argument("--embed-model", default=os.getenv("WM_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    ap.add_argument("--embed-device", default=os.getenv("EMB_DEVICE") or None)
    ap.add_argument("--embed-batch", type=int,
                    default=int(os.getenv("WM_EMBED_BATCH", str(DEFAULT_BATCH_SIZE))))
    ap.add_argument("--trust-remote-code", action="store_true",
                    default=_env_flag("WM_EMBED_TRUST_REMOTE_CODE"))
    ap.add_argument("--reuse-steps", action="store_true", default=_env_flag("WM_REUSE_STEPS"))
    ap.add_argument("--watched-tools", default=os.getenv("WM_WATCHED_TOOLS", ""))
    ap.add_argument("--discover-args", default=os.getenv("DISCOVER_ARGS") or None)
    ap.add_argument("--stage", action="append", default=[],
                    help=f"run only these stages (repeatable): {list(STAGES)}")
    ap.add_argument("--domain", default="unknown", help="domain tag for --input rows")
    args = ap.parse_args(argv)

    if not args.workdir:
        ap.error("--workdir (or WM_WORKDIR) is required")
    work = Path(args.workdir)
    if args.input:
        ingest_rollouts(args.input, work, default_domain=args.domain)
    try:
        build_graph(
            work,
            embed_model=args.embed_model,
            embed_device=args.embed_device,
            embed_batch=args.embed_batch,
            trust_remote_code=args.trust_remote_code,
            reuse_steps=args.reuse_steps,
            watched_tools=args.watched_tools,
            discover_args=args.discover_args,
            stages=args.stage or None,
        )
    except BuildError as exc:
        print(f"[build] ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
