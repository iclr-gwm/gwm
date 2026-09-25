#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive one live GWM mode (frozen | build | evolve) across one or more suites.

Assumes a vLLM+GWM server is already serving ``--base`` with the relevant
adapters registered. Emits a JSON report to ``--out``.

Modes:
  frozen  adapter already points at a pre-mined bundle; run advise (K=1) and
          select (K>1) over held-out test tasks.
  build   mine a graph from a ratio of rollouts, activate it via /gwm/rollback,
          then run GWM inference over the held-out test split.
  evolve  force an EvolveManager rebuild from seeded rollouts, wait for it, then
          run GWM inference.

Data-path args accept any location; nothing here is environment-specific.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def http_json(method: str, url: str, body: dict | None = None, *, timeout: float = 600.0):
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def wait_ready(base: str, *, timeout_s: float = 1200.0) -> None:
    health = base.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            status, _ = http_json("GET", health, timeout=5.0)
            if status == 200:
                return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        time.sleep(3.0)
    raise RuntimeError(f"server not ready at {health}: {last}")


def read_jsonl_gz(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def read_uids(path: Path) -> set[str]:
    uids: set[str] = set()
    if not Path(path).is_file():
        return uids
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(rec.get("uid") or rec.get("task_id") or "")
            if uid:
                uids.add(uid)
    return uids


def flow_of(rec: dict) -> list[dict]:
    return rec.get("conversation_flow") or rec.get("flow") or []


def flow_to_chat(flow: list[dict]) -> list[dict]:
    msgs: list[dict] = []
    for ev in flow:
        t = ev.get("type")
        if t == "system_message":
            msgs.append({"role": "system", "content": str(ev.get("content") or "")})
        elif t == "user_message":
            msgs.append({"role": "user", "content": str(ev.get("content") or "")})
        elif t == "ai_message":
            parts = []
            if ev.get("content"):
                parts.append(str(ev["content"]))
            for tc in ev.get("tool_calls") or []:
                parts.append(
                    f"[tool_call] {tc.get('name')}("
                    f"{json.dumps(tc.get('args', {}), sort_keys=True)})"
                )
            msgs.append({"role": "assistant", "content": "\n".join(parts) or "(no content)"})
        elif t == "tool_result":
            res = ev.get("result")
            msgs.append({
                "role": "user",
                "content": f"[tool result: {ev.get('tool_name', '')}] "
                f"{json.dumps(res, default=str)[:800]}",
            })
    return msgs


def split_prefix_gold(flow: list[dict]) -> tuple[list[dict], str]:
    """Prefix = flow up to the last ai tool-call; gold = that tool name."""
    last_idx = -1
    gold = ""
    for i, ev in enumerate(flow):
        if ev.get("type") == "ai_message" and (ev.get("tool_calls") or []):
            last_idx = i
            gold = str((ev["tool_calls"][0] or {}).get("name") or "")
    if last_idx <= 0:
        return flow[:-1] if len(flow) > 1 else flow, gold
    return flow[:last_idx], gold


def chat_gwm(base, model, messages, *, adapter, preset, k, episode, max_tokens=96):
    mode = "advise" if k <= 1 else "select"
    t0 = time.perf_counter()
    status, body = http_json(
        "POST", f"{base}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0 if k <= 1 else 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {
                "gwm.adapter": adapter, "gwm.preset": preset,
                "gwm.mode": mode, "gwm.k": k, "gwm.episode_id": episode,
            },
        }, timeout=600.0,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    if status != 200:
        return {"k": k, "mode": mode, "ok": False, "error": body, "latency_ms": ms}
    choices = body.get("choices") or []
    text = (choices[0].get("message") or {}).get("content") or "" if choices else ""
    return {"k": k, "mode": mode, "ok": True, "latency_ms": ms, "text": text,
            "n_choices": len(choices), "finish": choices[0].get("finish_reason") if choices else None}


def admin_advise(base, flow, *, adapter, preset, episode):
    status, adv = http_json(
        "POST", f"{base}/gwm/advise",
        {"conversation_flow": flow, "preset": preset, "adapter": adapter,
         "episode_id": episode}, timeout=300.0)
    return status, adv


def do_build(base, suite, *, rollouts, workdir, embed_device):
    from vllm_gwm.build import pipeline
    from vllm_gwm.runtime.adapter import GraphAdapter
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ing = pipeline.ingest_rollouts(rollouts, workdir, default_domain=suite)
    discover_mode = "within_domain"
    try:
        res = pipeline.build_graph(workdir, embed_device=embed_device, embed_batch=64,
                                   watched_tools=(), show_progress_bar=False)
    except Exception as exc:
        # sparse ratios can't sustain per-domain clustering; pool globally
        discover_mode = f"global (fallback: {type(exc).__name__})"
        res = pipeline.build_graph(workdir, embed_device=embed_device, embed_batch=64,
                                   watched_tools=(), reuse_steps=True,
                                   discover_args={"within_domain": False},
                                   show_progress_bar=False)
    GraphAdapter.validate(workdir, name=suite)
    trans = json.loads((workdir / "reports/transitions.json").read_text())
    n_states = sum(len(dd.get("state_fail") or []) for dd in trans.values())
    status, body = http_json("POST", f"{base}/gwm/rollback",
                             {"adapter": suite, "version_path": str(workdir.resolve())},
                             timeout=120.0)
    return {"ingested": ing, "stages": list(res.get("stages", {})),
            "discover_mode": discover_mode,
            "domains": list(trans), "n_states": n_states,
            "activate_status": status, "activate_body": body}


def seed_store(data_root, rollouts):
    """Overwrite the server's rollout store with a suite's train rollouts."""
    store = Path(data_root) / "rollouts" / "rollouts.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(rollouts, "rt", encoding="utf-8") as inf, store.open("w", encoding="utf-8") as out:
        for line in inf:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = str(rec.get("uid") or rec.get("task_id") or f"r{n}")
            flow = flow_of(rec)
            out.write(json.dumps({
                "id": uid, "episode_id": uid, "uid": uid,
                "flow": flow, "conversation_flow": flow,
                "success": rec.get("overall_success"), "overall_success": rec.get("overall_success"),
                "domain": rec.get("domain") or "unknown", "split": "train",
                "finalized": True, "source": "seed",
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def do_evolve(base, suite, *, timeout_s, data_root=None, rollouts=None):
    seeded = None
    if data_root and rollouts and Path(rollouts).is_file():
        seeded = seed_store(data_root, rollouts)
    status, before = http_json("GET", f"{base}/gwm/builds", timeout=30.0)
    n_before = len((before or {}).get("jobs") or [])
    status, trig = http_json("POST", f"{base}/gwm/evolve",
                             {"adapter": suite, "force": True}, timeout=60.0)
    if status != 200 or not trig.get("ok"):
        return {"ok": False, "trigger": trig, "status": status}
    job_id = (trig.get("job") or {}).get("id")
    deadline = time.time() + timeout_s
    job = trig.get("job") or {}
    body: dict = {}
    while time.time() < deadline:
        status, body = http_json("GET", f"{base}/gwm/builds", timeout=30.0)
        jobs = {j.get("id"): j for j in (body.get("jobs") or [])}
        job = jobs.get(job_id) or job
        if job.get("status") in ("completed", "failed"):
            break
        time.sleep(5.0)
    return {"ok": job.get("status") == "completed", "job": job, "seeded": seeded,
            "n_before": n_before, "n_after": len((body or {}).get("jobs") or [])}


def eval_suite(base, model, suite, *, adapter, preset, eval_rollouts, exclude_rollouts,
               n_test, k_values):
    """Eval on held-out tasks: records in `eval_rollouts` (100p) not in the build
    ratio set `exclude_rollouts` (e.g. 20p) — i.e. the 'remaining ratio'."""
    exclude = read_uids(Path(exclude_rollouts)) if exclude_rollouts else set()
    pool = read_jsonl_gz(Path(eval_rollouts), limit=max(n_test * 40, 500)) \
        if Path(eval_rollouts).is_file() else []
    rows = []
    for rec in pool:
        uid = str(rec.get("uid") or rec.get("task_id") or "")
        if uid and uid in exclude:
            continue
        rows.append(rec)
        if len(rows) >= n_test:
            break
    tasks = []
    for rec in rows:
        flow = flow_of(rec)
        if not flow:
            continue
        prefix, gold = split_prefix_gold(flow)
        msgs = flow_to_chat(prefix)
        if not any(m["role"] == "user" for m in msgs):
            continue
        msgs.append({"role": "user", "content": "What is your single next step?"})
        ep = f"live-{suite}-{uuid.uuid4().hex[:6]}"
        st, adv = admin_advise(base, prefix, adapter=adapter, preset=preset, episode=ep + "-adm")
        task = {"uid": rec.get("uid") or rec.get("task_id"), "gold_next_tool": gold,
                "advise": {"state": (adv or {}).get("state"), "injected": (adv or {}).get("injected"),
                           "is_trap": (adv or {}).get("is_trap"),
                           "success_score": (adv or {}).get("success_score")} if st == 200 else {"error": adv},
                "gwm": []}
        for k in k_values:
            entry = chat_gwm(base, model, msgs, adapter=adapter, preset=preset, k=k, episode=f"{ep}-k{k}")
            entry["gold_hit"] = bool(gold) and gold.split("-")[-1].lower() in (entry.get("text") or "").lower()
            task["gwm"].append(entry)
        tasks.append(task)
    n = len(tasks)
    inj = sum(1 for t in tasks if (t.get("advise") or {}).get("injected"))
    lat = [e["latency_ms"] for t in tasks for e in t["gwm"] if e.get("ok")]
    hit = sum(1 for t in tasks for e in t["gwm"] if e.get("gold_hit"))
    tot = sum(1 for t in tasks for e in t["gwm"] if e.get("ok"))
    return {"suite": suite, "n_tasks": n, "n_injected": inj,
            "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
            "gold_hit_rate": round(hit / tot, 3) if tot else None, "tasks": tasks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["frozen", "build", "evolve"])
    ap.add_argument("--suites", required=True, help="comma list, e.g. eops,crm")
    ap.add_argument("--rollouts-root", default="")
    ap.add_argument("--ratio", default="20p", help="build ratio (also excluded from eval)")
    ap.add_argument("--eval-ratio", default="100p", help="pool to draw held-out eval tasks from")
    ap.add_argument("--built-root", default="/tmp/gwm_live/built")
    ap.add_argument("--data-root", default="", help="evolve server data root (to seed store)")
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--k-values", default="1,4")
    ap.add_argument("--evolve-timeout", type=float, default=1200.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    wait_ready(base)

    report: dict[str, Any] = {"mode": args.mode, "model": args.model, "base": base,
                              "ratio": args.ratio, "suites": {}, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for suite in suites:
        preset = suite
        adapter = suite
        entry: dict[str, Any] = {}
        try:
            if args.mode == "build":
                rollouts = Path(args.rollouts_root) / suite / args.ratio / "out" / "rollouts.jsonl.gz"
                entry["build"] = do_build(base, suite, rollouts=rollouts,
                                          workdir=Path(args.built_root) / f"{suite}_{args.ratio}",
                                          embed_device=args.embed_device)
            elif args.mode == "evolve":
                ev_roll = Path(args.rollouts_root) / suite / args.ratio / "out" / "rollouts.jsonl.gz"
                entry["evolve"] = do_evolve(base, suite, timeout_s=args.evolve_timeout,
                                            data_root=args.data_root, rollouts=ev_roll)
            eval_rollouts = Path(args.rollouts_root) / suite / args.eval_ratio / "out" / "rollouts.jsonl.gz"
            exclude_rollouts = Path(args.rollouts_root) / suite / args.ratio / "out" / "rollouts.jsonl.gz"
            entry["eval"] = eval_suite(base, args.model, suite, adapter=adapter, preset=preset,
                                       eval_rollouts=eval_rollouts, exclude_rollouts=exclude_rollouts,
                                       n_test=args.n_test, k_values=k_values)
            ev = entry["eval"]
            print(f"[{args.mode}/{suite}] n_tasks={ev['n_tasks']} injected={ev['n_injected']} "
                  f"avg_latency_ms={ev['avg_latency_ms']} gold_hit_rate={ev['gold_hit_rate']}")
        except Exception as exc:  # keep other suites/modes even if one fails
            import traceback
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{args.mode}/{suite}] FAILED: {entry['error']}")
            traceback.print_exc()
        report["suites"][suite] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
