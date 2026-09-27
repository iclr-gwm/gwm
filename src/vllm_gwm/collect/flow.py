# SPDX-License-Identifier: MIT
"""Convert OpenAI chat messages to benchmark conversation_flow events."""

from __future__ import annotations

import json
import re
from typing import Any

# Toucan Green wraps the bare question with this delimiter (see
# benchmarks/assets/toucan/green/toucan_orchestrator.py::build_agent_prompt).
# Classifying the full prompt blob (policy + tool catalog) always abstains.
_TOUCAN_QUESTION_RE = re.compile(
    r"Task / user question:\n(.*?)\n\nAvailable tools:",
    re.S,
)


def messages_to_flow(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "") or ""
            tool_calls = getattr(msg, "tool_calls", None) or []
        if role == "system":
            out.append({"type": "system_message", "content": str(content)})
        elif role == "user":
            out.append({"type": "user_message", "content": str(content)})
        elif role == "assistant":
            tc = []
            for call in tool_calls:
                if isinstance(call, dict):
                    fn = call.get("function") or {}
                    tc.append(
                        {
                            "name": fn.get("name") or call.get("name") or "",
                            "args": _parse_args(fn.get("arguments") or call.get("args") or {}),
                        }
                    )
                else:
                    fn = getattr(call, "function", None)
                    tc.append(
                        {
                            "name": getattr(fn, "name", "") if fn else "",
                            "args": _parse_args(getattr(fn, "arguments", "{}") if fn else {}),
                        }
                    )
            out.append(
                {
                    "type": "ai_message",
                    "content": str(content),
                    "tool_calls": tc,
                }
            )
        elif role == "tool":
            out.append(
                {
                    "type": "tool_result",
                    "tool_name": str(getattr(msg, "name", "") or msg.get("name", "")),
                    "result": {"success": True, "result": content},
                }
            )
    return out


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    return {}


def parse_json_tool_action(content: str) -> dict[str, Any] | None:
    """Parse Toucan-style JSON ``{"name", "arguments"}`` from assistant text."""
    text = (content or "").strip()
    if not text:
        return None
    candidates = [text]
    if "{" in text:
        candidates.append(text[text.find("{") :])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("name"):
            args = data.get("arguments")
            if not isinstance(args, dict):
                args = data.get("args") if isinstance(data.get("args"), dict) else {}
            return {"name": str(data["name"]), "args": args}
    return None


def _toucan_ai_event(event: dict[str, Any]) -> dict[str, Any]:
    calls = list(event.get("tool_calls") or [])
    if not calls:
        parsed = parse_json_tool_action(str(event.get("content") or ""))
        if parsed:
            calls = [parsed]
    return {
        "type": "ai_message",
        "content": str(event.get("content") or ""),
        "tool_calls": calls,
    }


def harvest_toucan_flow(flow: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape a chat transcript into the harvest schema used to mine Toucan graphs.

    Keep the bare user question plus JSON tool actions. Drop system prompts and
    Green re-prompt blobs that include the tool catalog (those always classify
    as UNKNOWN against within-domain centroids).
    """
    meta = [
        ev
        for ev in flow
        if isinstance(ev, dict) and ev.get("type") == "task_metadata"
    ]
    question = ""
    actions: list[dict[str, Any]] = []
    for ev in flow:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        if kind == "user_message":
            content = str(ev.get("content") or "")
            match = _TOUCAN_QUESTION_RE.search(content)
            if match:
                question = match.group(1).strip()
            elif not question:
                question = content.strip()
        elif kind == "ai_message":
            actions.append(_toucan_ai_event(ev))
    out: list[dict[str, Any]] = list(meta)
    out.append({"type": "user_message", "content": question})
    out.extend(actions)
    return out


def shape_flow(flow: list[dict[str, Any]], flow_style: str | None) -> list[dict[str, Any]]:
    style = (flow_style or "").strip().lower()
    if style == "toucan":
        return harvest_toucan_flow(flow)
    return list(flow or [])
