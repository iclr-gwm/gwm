#!/usr/bin/env python3
"""Live integration test against a running vLLM server with the GWM plugin.

Exercises admin routes and GWM-gated ``/v1/chat/completions`` (advise + select),
recording latency and sample outputs.

Attach to an existing server:
  export VLLM_BASE_URL=http://127.0.0.1:8765/v1
  .venv/bin/python scripts/live_e2e.py --model google/gemma-4-E2B-it

Or use the wrapper (starts vLLM when available):
  bash scripts/live_smoke.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"

CASES = [
    {"preset": "crm", "fixture": FIX / "crm_retry_flow.json", "adapter": "tiny"},
    {"preset": "eops", "fixture": FIX / "eops_retry_flow.json", "adapter": "tiny"},
]


def _http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> tuple[int, dict[str, Any] | list[Any] | str, dict[str, str]]:
    data = None
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    if body is not None:
        data = json.dumps(body).encode()
    req = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            hdr_out = dict(resp.headers)
            if not raw.strip():
                return resp.status, {}, hdr_out
            try:
                return resp.status, json.loads(raw), hdr_out
            except json.JSONDecodeError:
                return resp.status, raw, hdr_out
    except HTTPError as exc:
        raw = exc.read().decode()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        return exc.code, payload, dict(exc.headers)


def wait_ready(base: str, *, timeout_s: float = 600.0, interval_s: float = 2.0) -> None:
    health = base.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            status, _, _ = _http_json("GET", health, timeout=5.0)
            if status == 200:
                return
        except (HTTPError, URLError, TimeoutError) as exc:
            last_err = str(exc)
        time.sleep(interval_s)
    raise RuntimeError(f"server not ready at {health} within {timeout_s}s: {last_err}")


def flow_to_chat(flow: list[dict[str, Any]]) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
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
                    f"[tool_call] {tc.get('name')}({json.dumps(tc.get('args', {}), sort_keys=True)})"
                )
            msgs.append({"role": "assistant", "content": "\n".join(parts) or "(no content)"})
        elif t == "tool_result":
            res = ev.get("result")
            msgs.append(
                {
                    "role": "user",
                    "content": f"[tool result: {ev.get('tool_name', '')}] "
                    f"{json.dumps(res, default=str)[:800]}",
                }
            )
    return msgs


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base_url.rstrip("/")
    model = args.model
    report: dict[str, Any] = {
        "base_url": base,
        "model": model,
        "cases": [],
        "baseline_chat_ms": None,
        "gwm_advise_chat_ms": None,
        "gwm_select_chat_ms": None,
    }

    print(f"waiting for {base} ...")
    wait_ready(base, timeout_s=args.ready_timeout)

    t0 = time.perf_counter()
    status, info, _ = _http_json("GET", f"{base}/gwm/info")
    report["gwm_info_ms"] = int((time.perf_counter() - t0) * 1000)
    if status != 200:
        raise RuntimeError(f"/v1/gwm/info failed: {status} {info}")
    graphs_info = info.get("graphs") or []
    graph_count = len(graphs_info) if isinstance(graphs_info, list) else "n/a"
    print(f"gwm/info ok presets={info.get('presets')} graphs={graph_count}")

    status, graphs, _ = _http_json("GET", f"{base}/gwm/graphs")
    if status != 200:
        raise RuntimeError(f"/v1/gwm/graphs failed: {status} {graphs}")
    loaded = graphs.get("loaded") if isinstance(graphs, dict) else None
    print(f"gwm/graphs ok loaded={loaded}")

    graph_url = base.rstrip("/").removesuffix("/v1") + "/v1/graph"
    status, graph_page, graph_hdrs = _http_json("GET", graph_url, timeout=30.0)
    report["graph_viewer_status"] = status
    if status == 200:
        blob = graph_page if isinstance(graph_page, str) else json.dumps(graph_page)
        print(
            f"v1/graph ok content-type={graph_hdrs.get('content-type', '')} "
            f"bytes={len(blob)}"
        )
    else:
        print(f"v1/graph skipped status={status}")

    for case in CASES:
        flow = json.loads(Path(case["fixture"]).read_text(encoding="utf-8"))
        ep = f"live-{case['preset']}-{uuid.uuid4().hex[:6]}"
        t0 = time.perf_counter()
        status, adv, _ = _http_json(
            "POST",
            f"{base}/gwm/advise",
            {
                "conversation_flow": flow,
                "preset": "tiny",
                "adapter": case["adapter"],
                "episode_id": ep,
            },
            timeout=args.request_timeout,
        )
        ms = int((time.perf_counter() - t0) * 1000)
        if status != 200:
            raise RuntimeError(f"/v1/gwm/advise failed: {status} {adv}")
        entry = {
            "preset": case["preset"],
            "episode_id": ep,
            "admin_advise_ms": ms,
            "state": adv.get("state"),
            "injected": adv.get("injected"),
            "success_score": adv.get("success_score"),
            "advice_preview": (adv.get("block") or "")[:240],
            "harness": adv.get("harness") or {},
        }
        report["cases"].append(entry)
        print(f"\n=== admin/advise/{case['preset']} ===")
        print(f"latency_ms={ms} state={adv.get('state')} injected={adv.get('injected')}")
        if adv.get("block"):
            print(adv["block"][:500])

    # baseline chat (no GWM)
    chat_msgs = flow_to_chat(json.loads((FIX / "crm_retry_flow.json").read_text()))
    chat_msgs.append({"role": "user", "content": "What is your single next step?"})
    t0 = time.perf_counter()
    status, baseline, _ = _http_json(
        "POST",
        f"{base}/chat/completions",
        {"model": model, "messages": chat_msgs, "max_tokens": args.max_tokens, "temperature": 0},
        timeout=args.request_timeout,
    )
    report["baseline_chat_ms"] = int((time.perf_counter() - t0) * 1000)
    if status != 200:
        raise RuntimeError(f"baseline chat failed: {status} {baseline}")
    baseline_text = ""
    if isinstance(baseline, dict):
        baseline_text = (
            (baseline.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        )
    report["baseline_sample"] = baseline_text[:400]
    print(f"\n=== chat/baseline (no GWM) ===")
    print(f"latency_ms={report['baseline_chat_ms']}")
    print(baseline_text[:400])

    # GWM advise via chat/completions
    t0 = time.perf_counter()
    status, gwm_out, _ = _http_json(
        "POST",
        f"{base}/chat/completions",
        {
            "model": model,
            "messages": chat_msgs,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "vllm_xargs": {
                "gwm.adapter": "tiny",
                "gwm.mode": "advise",
                "gwm.k": 1,
                "gwm.preset": "tiny",
                "gwm.episode_id": f"live-chat-{uuid.uuid4().hex[:6]}",
            },
        },
        timeout=args.request_timeout,
    )
    report["gwm_advise_chat_ms"] = int((time.perf_counter() - t0) * 1000)
    if status != 200:
        raise RuntimeError(f"GWM advise chat failed: {status} {gwm_out}")
    gwm_text = ""
    if isinstance(gwm_out, dict):
        gwm_text = (gwm_out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        report["gwm_advise_usage"] = gwm_out.get("usage")
    report["gwm_advise_sample"] = gwm_text[:400]
    print(f"\n=== chat/gwm-advise ===")
    print(f"latency_ms={report['gwm_advise_chat_ms']}")
    print(gwm_text[:400])

    # GWM select k=4
    t0 = time.perf_counter()
    status, sel_out, _ = _http_json(
        "POST",
        f"{base}/chat/completions",
        {
            "model": model,
            "messages": chat_msgs[:-1],  # drop "next step" prompt
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
            "vllm_xargs": {
                "gwm.adapter": "tiny",
                "gwm.mode": "select",
                "gwm.k": 4,
                "gwm.preset": "tiny",
                "gwm.episode_id": f"live-sel-{uuid.uuid4().hex[:6]}",
            },
        },
        timeout=args.request_timeout,
    )
    report["gwm_select_chat_ms"] = int((time.perf_counter() - t0) * 1000)
    if status != 200:
        raise RuntimeError(f"GWM select chat failed: {status} {sel_out}")
    sel_text = ""
    if isinstance(sel_out, dict):
        sel_text = (sel_out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        report["gwm_select_usage"] = sel_out.get("usage")
    report["gwm_select_sample"] = sel_text[:400]
    print(f"\n=== chat/gwm-select k=4 ===")
    print(f"latency_ms={report['gwm_select_chat_ms']}")
    print(sel_text[:400])

    report["total_ms"] = sum(
        x
        for x in (
            report.get("gwm_info_ms"),
            report["baseline_chat_ms"],
            report["gwm_advise_chat_ms"],
            report["gwm_select_chat_ms"],
        )
        if isinstance(x, int)
    )
    for c in report["cases"]:
        report["total_ms"] = (report.get("total_ms") or 0) + c.get("admin_advise_ms", 0)
    print(f"\nTOTAL live_e2e_ms≈{report['total_ms']}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Live vLLM + GWM integration test")
    ap.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8765/v1"),
    )
    ap.add_argument(
        "--model",
        default=os.getenv("VLLM_GWM_LIVE_MODEL", "google/gemma-4-E2B-it"),
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    report = run_live(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
