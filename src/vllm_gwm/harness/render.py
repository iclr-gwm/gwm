"""Torch-free conversation_flow renderer (mirrors nla benchmark_ingest).

A1/A2(sentence-transformer) must run in the base interpreter without torch, so we
reimplement the small flatten logic from ``nla_qwenvl.benchmark_ingest`` here
(``_render_ai`` / ``_to_qwen_messages``) rather than importing it. The rendered
form matches the NLA path closely (``[tool_call] name({sorted json})`` lines,
tool results folded into a following user turn). The optional NLA-activation
encoder still imports the real ``benchmark_ingest`` for exact cell parity.
"""
from __future__ import annotations

import json
from typing import Any

from vllm_gwm.harness import scrub  # vendored — generic noise scrubber (see SYNC.md)


def render_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Anthropic-style content blocks: keep text parts only.
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text", "")))
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n".join(parts).strip()
    return ""


def render_ai(event: dict) -> str:
    """Flatten an ai_message to ``text + [tool_call] name({args})`` lines."""
    parts: list[str] = []
    text = render_content(event.get("content"))
    if text:
        parts.append(text)
    for tc in event.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = tc.get("name") or fn.get("name") or "tool"
        args = tc.get("args")
        if args is None:
            args = tc.get("input")
        if args is None and "arguments" in fn:
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                args = raw
        sb = scrub.get_scrubber()
        if sb is not None and isinstance(args, (dict, list)):
            args = sb.scrub(args, tool=name)
        try:
            args_s = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            args_s = str(args)
        parts.append(f"[tool_call] {name}({args_s})")
    return "\n".join(parts).strip()


def render_tool_result(event: dict, max_chars: int) -> str:
    """Best-effort flatten of a tool_result event to text (truncated)."""
    name = event.get("tool_name") or "tool"
    res = event.get("result")
    text = ""
    if isinstance(res, dict):
        inner = res.get("result")
        if isinstance(inner, dict):
            content = inner.get("content")
            if isinstance(content, list):
                chunks = [c.get("text", "") for c in content if isinstance(c, dict)]
                text = "\n".join(c for c in chunks if c)
        if not text:
            text = json.dumps(res, ensure_ascii=False)[:max_chars]
        if res.get("error"):
            text = f"ERROR: {res.get('error')}\n{text}"
    else:
        text = str(res)
    sb = scrub.get_scrubber()
    if sb is not None and text:
        try:                                  # structured payload: drop noise keys
            text = json.dumps(sb.scrub(json.loads(text)), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):       # free text: value redaction only
            text = sb.scrub_text(text)
    if len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return f"[{name}] {text}"


def tool_result_outcome(event: dict) -> str:
    """Classify a tool_result as 'ok' | 'error'."""
    res = event.get("result")
    if not isinstance(res, dict):
        return "ok"
    if res.get("error"):
        return "error"
    inner = res.get("result")
    if isinstance(inner, dict) and inner.get("isError"):
        return "error"
    if res.get("success") is False:
        return "error"
    return "ok"


def to_messages(conversation_flow: list, max_tool_result_chars: int = 2000) -> list[dict]:
    """Flatten a conversation_flow to chat messages (tool-call aware)."""
    raw: list[dict] = []
    pending: list[str] = []

    def _flush():
        if pending:
            raw.append({"role": "user", "content": "Tool result(s):\n" + "\n".join(pending)})
            pending.clear()

    for e in conversation_flow or []:
        if not isinstance(e, dict):
            continue
        et = e.get("type")
        if et == "system_message":
            _flush()
            txt = render_content(e.get("content"))
            if txt:
                raw.append({"role": "system", "content": txt})
        elif et == "user_message":
            _flush()
            txt = render_content(e.get("content"))
            if txt:
                raw.append({"role": "user", "content": txt})
        elif et == "ai_message":
            _flush()
            txt = render_ai(e)
            if txt:
                raw.append({"role": "assistant", "content": txt})
        elif et == "tool_result":
            pending.append(render_tool_result(e, max_tool_result_chars))
    _flush()

    merged: list[dict] = []
    for m in raw:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] = merged[-1]["content"] + "\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


def render_text(conversation_flow: list, max_tool_result_chars: int = 2000,
                max_flow_chars: int = 60000) -> str:
    """Single text blob for embedding, with the same oldest-events-dropped budget
    loop the NLA embedder uses (head = system+user kept; tail carries outcome)."""
    flow = list(conversation_flow or [])
    messages = to_messages(flow, max_tool_result_chars)
    head = 2
    while sum(len(m["content"]) for m in messages) > max_flow_chars and len(flow) > head + 1:
        flow = flow[:head] + flow[head + 1:]
        messages = to_messages(flow, max_tool_result_chars)
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
