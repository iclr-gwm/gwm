# SPDX-License-Identifier: Apache-2.0
"""Live Qwen3.6 tests for the three GWM vLLM modes.

1. Frozen graph (tiny fixture) + live advise/select at K=1, 2, 10
2. Build a graph from copied trajectories, then live-infer against it
3. Self-evolve: seed the rollout store and force a background rebuild

Skipped unless ``RUN_GWM_LIVE=1`` or ``VLLM_BASE_URL`` is set. Launch with
``CUDA_VISIBLE_DEVICES=1 bash scripts/live_qwen36_modes.sh``.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _default_gwm_data() -> Path:
    """Sibling ``gwm_data`` next to the repo (``../gwm_data`` from api_vllm)."""
    vllm_root = ROOT.parents[1]
    outer = vllm_root.parent
    if outer.name == "api_vllm":
        return outer.parent / "gwm_data"
    return outer / "gwm_data"


GWM_DATA = Path(os.getenv("GWM_DATA_ROOT", str(_default_gwm_data()))).resolve()
# HDBSCAN default min_cluster_size=50; toucan qwen36 10p (910 train) is the
# smallest in-tree corpus that already produced a PRECHECK GO graph.
MIN_TRAIN_ROLLOUTS = 50
K_VALUES = (1, 2, 10)
IO_DIR = Path(os.getenv("GWM_LIVE_IO_DIR", str(GWM_DATA / "live" / "samples")))
_IO_SAMPLES: list[dict[str, Any]] = []
_CHAT_MAX_TOKENS = int(os.getenv("GWM_LIVE_MAX_TOKENS", "192"))


def _live_enabled() -> bool:
    if os.getenv("VLLM_BASE_URL", "").strip():
        return True
    return os.getenv("RUN_GWM_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.live,
    pytest.mark.skipif(
        not _live_enabled(),
        reason="set RUN_GWM_LIVE=1 or VLLM_BASE_URL to run live Qwen3.6 GWM modes",
    ),
]


def _http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 600.0,
) -> tuple[int, Any]:
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            if not raw.strip():
                return resp.status, {}
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except HTTPError as exc:
        raw = exc.read().decode()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        return exc.code, payload


def wait_ready(base: str, *, timeout_s: float = 900.0) -> None:
    health = base.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            status, _ = _http_json("GET", health, timeout=5.0)
            if status == 200:
                return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"server not ready at {health} within {timeout_s}s: {last}")


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
                    f"[tool_call] {tc.get('name')}("
                    f"{json.dumps(tc.get('args', {}), sort_keys=True)})"
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


def _load_flow(name: str) -> list[dict[str, Any]]:
    for cand in (
        GWM_DATA / "fixtures" / name,
        ROOT / "tests" / "fixtures" / name,
    ):
        if cand.is_file():
            raw = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("conversation_flow"), list):
                return raw["conversation_flow"]
            if isinstance(raw, list):
                return raw
    raise FileNotFoundError(name)


def _preview_flow(flow: list[dict[str, Any]], *, n: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in flow[:n]:
        row = {"type": ev.get("type")}
        if ev.get("type") == "user_message":
            row["content"] = str(ev.get("content") or "")[:400]
        elif ev.get("type") == "ai_message":
            tcs = ev.get("tool_calls") or []
            row["tool_calls"] = [
                {"name": t.get("name"), "args": t.get("args")} for t in tcs[:2]
            ]
        elif ev.get("type") == "tool_result":
            row["tool_name"] = ev.get("tool_name")
            res = ev.get("result")
            row["result"] = json.dumps(res, default=str)[:240]
        out.append(row)
    return out


def _last_tool(flow: list[dict[str, Any]]) -> str:
    for ev in reversed(flow):
        if ev.get("type") == "ai_message":
            tcs = ev.get("tool_calls") or []
            if tcs:
                return str(tcs[0].get("name") or "")
        if ev.get("type") == "tool_result" and ev.get("tool_name"):
            return str(ev.get("tool_name"))
    return ""


def _advise(env: dict[str, Any], flow: list[dict[str, Any]], *, adapter: str, preset: str, episode: str) -> dict[str, Any]:
    status, adv = _http_json(
        "POST",
        f"{env['base']}/gwm/advise",
        {
            "conversation_flow": flow,
            "preset": preset,
            "adapter": adapter,
            "episode_id": episode,
        },
        timeout=300.0,
    )
    assert status == 200, adv
    assert isinstance(adv, dict), adv
    return adv


def _chat_plain(
    env: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    timeout: float = 600.0,
    max_tokens: int = _CHAT_MAX_TOKENS,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    status, body = _http_json(
        "POST",
        f"{env['base']}/chat/completions",
        {
            "model": env["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    assert status == 200, f"unguided chat failed: {status} {body}"
    assert isinstance(body, dict), body
    choices = body.get("choices") or []
    text = (choices[0].get("message") or {}).get("content") or "" if choices else ""
    return {"mode": "unguided", "latency_ms": ms, "text": text}


def _record(sample: dict[str, Any]) -> None:
    _IO_SAMPLES.append(sample)
    why = sample.get("why")
    print(f"\n[{sample.get('mode')}] {why}")
    adv = sample.get("admin_advise") or {}
    print(
        f"  advise state={adv.get('state')} injected={adv.get('injected')} "
        f"trap={adv.get('is_trap')} score={adv.get('success_score')}"
    )
    if adv.get("block"):
        print("  guidance:", str(adv["block"])[:500])
    ug = sample.get("unguided") or {}
    print(f"  unguided ({ug.get('latency_ms')}ms): {(ug.get('text') or '')[:400]}")
    for entry in sample.get("gwm") or []:
        print(
            f"  gwm k={entry.get('k')} ({entry.get('latency_ms')}ms): "
            f"{(entry.get('text') or '')[:400]}"
        )
    print("  success:", sample.get("success"))


def _flush_io(env: dict[str, Any]) -> None:
    if not _IO_SAMPLES:
        return
    IO_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() else "-" for ch in env["model"]).strip("-")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "model": env["model"],
        "base": env["base"],
        "graphs": env["graph_names"],
        "samples": _IO_SAMPLES,
        "graph_harvested_success": {
            "source": "graphs/toucan_qwen36_10p/reports/EXAMPLES.md",
            "note": (
                "Few-shot from a successful toucan trajectory (Seattle weather "
                "then WA alerts) used as graph grounding — contrast with the "
                "live stall prefixes below."
            ),
            "spot_check_state": "multi-turn:16",
            "successful_next_tools": [
                "weather-calculator-get_weather",
                "weather-get_alerts",
            ],
        },
    }
    json_path = IO_DIR / f"{slug}_modes_{stamp}.json"
    md_path = IO_DIR / f"{slug}_modes_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        f"# GWM live I/O — {env['model']}",
        "",
        "Interesting cases are trap/retry prefixes where GWM injects guidance "
        "and the policy next-step differs from the unguided retry.",
        "",
        "## Graph-harvested success (toucan few-shot)",
        "",
        "Outdoor festival in Seattle: successful trajectories call "
        "`weather-calculator-get_weather({city: Seattle})` then "
        "`weather-get_alerts({state: WA})` instead of stalling on one weather tool.",
        "",
    ]
    for sample in _IO_SAMPLES:
        lines += [
            f"## {sample.get('mode')}",
            "",
            str(sample.get("why") or ""),
            "",
            "### Input (prefix tail)",
            "",
            "```json",
            json.dumps(sample.get("input_preview") or [], indent=2)[:3000],
            "```",
            "",
            "### Admin advise",
            "",
            f"- state: `{sample.get('admin_advise', {}).get('state')}`",
            f"- injected: `{sample.get('admin_advise', {}).get('injected')}`",
            f"- trap: `{sample.get('admin_advise', {}).get('is_trap')}`",
            f"- score: `{sample.get('admin_advise', {}).get('success_score')}`",
            "",
            "```",
            str((sample.get("admin_advise") or {}).get("block") or "(no guidance)")[:2000],
            "```",
            "",
            "### Unguided (no GWM xargs)",
            "",
            "```",
            str((sample.get("unguided") or {}).get("text") or "")[:1500],
            "```",
            "",
        ]
        for entry in sample.get("gwm") or []:
            lines += [
                f"### GWM k={entry.get('k')} mode={entry.get('mode')} "
                f"({entry.get('latency_ms')} ms)",
                "",
                "```",
                str(entry.get("text") or "")[:1500],
                "```",
                "",
            ]
        lines += [
            "### Success flags",
            "",
            "```json",
            json.dumps(sample.get("success") or {}, indent=2),
            "```",
            "",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[io] wrote {json_path}")
    print(f"[io] wrote {md_path}")


@pytest.fixture(scope="module")
def live_env():
    base = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8771/v1").rstrip("/")
    model = os.getenv("VLLM_GWM_LIVE_MODEL", "Qwen/Qwen3.6-27B")
    wait_ready(base, timeout_s=float(os.getenv("VLLM_GWM_READY_TIMEOUT", "900")))
    status, info = _http_json("GET", f"{base}/gwm/info")
    assert status == 200, info
    graphs = info.get("graphs") or []
    names = [g["name"] if isinstance(g, dict) else g for g in graphs]
    env = {"base": base, "model": model, "info": info, "graph_names": names}
    yield env
    _flush_io(env)


def _chat_gwm(
    env: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    adapter: str,
    preset: str,
    k: int,
    episode: str,
    timeout: float = 600.0,
    max_tokens: int = 64,
) -> dict[str, Any]:
    mode = "advise" if k <= 1 else "select"
    t0 = time.perf_counter()
    status, body = _http_json(
        "POST",
        f"{env['base']}/chat/completions",
        {
            "model": env["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0 if k <= 1 else 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {
                "gwm.adapter": adapter,
                "gwm.preset": preset,
                "gwm.mode": mode,
                "gwm.k": k,
                "gwm.episode_id": episode,
            },
        },
        timeout=timeout,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    assert status == 200, f"k={k} chat failed: {status} {body}"
    assert isinstance(body, dict), body
    choices = body.get("choices") or []
    assert len(choices) == 1, f"GWM must return a single winning choice, got {len(choices)}"
    text = (choices[0].get("message") or {}).get("content") or ""
    return {
        "k": k,
        "mode": mode,
        "latency_ms": ms,
        "text": text,
        "usage": body.get("usage"),
        "finish_reason": choices[0].get("finish_reason"),
    }


def test_mode1_frozen_graph_live_advise_k1_k2_k10(live_env):
    """Frozen tiny fixture + live Qwen3.6 advise (K=1) and select (K=2, K=10)."""
    assert "tiny" in live_env["graph_names"], live_env["graph_names"]
    flow = _load_flow("crm_retry_flow.json")
    status, adv = _http_json(
        "POST",
        f"{live_env['base']}/gwm/advise",
        {
            "conversation_flow": flow,
            "preset": "tiny",
            "adapter": "tiny",
            "episode_id": f"live-frozen-admin-{uuid.uuid4().hex[:6]}",
        },
        timeout=300.0,
    )
    assert status == 200, adv
    assert not adv.get("error"), adv
    assert adv.get("state"), adv
    print(
        f"\n[frozen/admin-advise] state={adv.get('state')} "
        f"injected={adv.get('injected')} score={adv.get('success_score')}"
    )
    if adv.get("block"):
        print(str(adv["block"])[:400])

    messages = flow_to_chat(flow)
    messages.append({"role": "user", "content": "What is your single next step?"})
    reports = []
    for k in K_VALUES:
        entry = _chat_gwm(
            live_env,
            messages,
            adapter="tiny",
            preset="tiny",
            k=k,
            episode=f"live-frozen-k{k}-{uuid.uuid4().hex[:6]}",
        )
        assert entry["text"].strip(), entry
        reports.append(entry)
        print(
            f"[frozen/k={k} mode={entry['mode']}] "
            f"latency_ms={entry['latency_ms']} chars={len(entry['text'])}"
        )
        print(entry["text"][:300])
    assert [r["k"] for r in reports] == list(K_VALUES)


def test_mode2_build_then_live_infer(live_env):
    """Build a graph from copied toucan trajectories, activate it, live-infer."""
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    from vllm_gwm.build import pipeline
    from vllm_gwm.runtime.adapter import GraphAdapter

    src = GWM_DATA / "rollouts/toucan_qwen36_10p/rollouts.jsonl.gz"
    assert src.is_file(), (
        f"missing {src}; run bash scripts/prep_gwm_data.sh "
        "(needs gwm_artifacts/toucan/qwen36/10p)"
    )
    built_dir = Path(
        os.getenv("GWM_LIVE_BUILD_DIR", str(GWM_DATA / "builds/toucan_live"))
    )
    reuse = os.getenv("GWM_LIVE_REUSE_BUILD", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if reuse and (built_dir / "MANIFEST.json").is_file():
        info = GraphAdapter.validate(built_dir, name="toucan")
        print(f"[build] reusing {info.path}")
    else:
        built_dir.mkdir(parents=True, exist_ok=True)
        ingested = pipeline.ingest_rollouts(src, built_dir, default_domain="toucan")
        assert ingested["train"] >= MIN_TRAIN_ROLLOUTS, ingested
        print(f"[build] ingested train={ingested['train']} test={ingested['test']}")
        res = pipeline.build_graph(
            built_dir,
            embed_device=os.getenv("VLLM_GWM_BUILD_EMBED_DEVICE", "cpu"),
            embed_batch=64,
            watched_tools="respond",
            show_progress_bar=True,
        )
        assert list(res["stages"]) == list(pipeline.STAGES)
        GraphAdapter.validate(built_dir, name="toucan")
        trans = json.loads((built_dir / "reports/transitions.json").read_text())
        n_states = sum(len(dd.get("state_fail") or []) for dd in trans.values())
        print(f"[build] domains={list(trans)} n_states={n_states}")
        assert n_states >= 1

    status, body = _http_json(
        "POST",
        f"{live_env['base']}/gwm/rollback",
        {"adapter": "toucan", "version_path": str(built_dir.resolve())},
        timeout=60.0,
    )
    assert status == 200, body

    flow = _load_flow("toucan_prefix_flow.json")
    messages = flow_to_chat(flow)
    if not any(m["role"] == "user" for m in messages):
        messages = [{"role": "user", "content": "What is 2+2? Reply with a JSON tool call."}]
    entry = _chat_gwm(
        live_env,
        messages,
        adapter="toucan",
        preset="toucan",
        k=1,
        episode=f"live-build-k1-{uuid.uuid4().hex[:6]}",
    )
    assert entry["text"].strip(), entry
    print(f"[build-infer/k=1] latency_ms={entry['latency_ms']}")
    print(entry["text"][:400])


def test_mode3_self_evolve(live_env):
    """Force EvolveManager to rebuild from seeded trajectories, then live-advise."""
    status, before = _http_json("GET", f"{live_env['base']}/gwm/builds", timeout=30.0)
    assert status == 200, before
    n_before = len((before or {}).get("jobs") or [])

    status, trig = _http_json(
        "POST",
        f"{live_env['base']}/gwm/evolve",
        {"adapter": "evolved", "force": True},
        timeout=30.0,
    )
    assert status == 200, trig
    assert trig.get("ok") is True, trig
    job_id = (trig.get("job") or {}).get("id")
    assert job_id, trig
    print(f"[evolve] triggered job={job_id}")

    deadline = time.time() + float(os.getenv("GWM_LIVE_EVOLVE_TIMEOUT", "900"))
    job = trig.get("job") or {}
    while time.time() < deadline:
        status, body = _http_json("GET", f"{live_env['base']}/gwm/builds", timeout=30.0)
        assert status == 200, body
        jobs = {j.get("id"): j for j in (body.get("jobs") or [])}
        job = jobs.get(job_id) or job
        if job.get("status") in ("completed", "failed"):
            break
        time.sleep(5.0)
    assert job.get("status") == "completed", job
    assert n_before + 1 <= len((body.get("jobs") or [])), body
    print(f"[evolve] completed path={job.get('version_path')}")

    flow = _load_flow("toucan_prefix_flow.json")
    messages = flow_to_chat(flow) or [
        {"role": "user", "content": "What is 2+2? Reply with a JSON tool call."}
    ]
    entry = _chat_gwm(
        live_env,
        messages,
        adapter="evolved",
        preset="toucan",
        k=1,
        episode=f"live-evolve-k1-{uuid.uuid4().hex[:6]}",
    )
    assert entry["text"].strip(), entry
    print(f"[evolve-infer/k=1] latency_ms={entry['latency_ms']}")
    print(entry["text"][:400])
