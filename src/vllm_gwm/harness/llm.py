"""Minimal OpenAI-compatible chat client for the harness mediator.

stdlib-only (urllib) so it runs inside the CPU workflow_mining venv with no new
dependencies. Two implementations behind one call signature:

* :class:`ChatClient` — POST ``{base_url}/chat/completions`` (vLLM / any
  OpenAI-compatible endpoint, e.g. the resident Qwen3.6 serve on :8020).
* :class:`StubChat` — deterministic canned replies for offline smoke tests and
  plumbing validation (no GPU, no network). Selected with ``--llm-stub``.

Both return ``(text, usage_dict)``.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any


class ChatClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 700, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # the mediator must not burn budget on thinking traces
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode())
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return text, (out.get("usage") or {})


class StubChat:
    """Deterministic offline stand-in: advises on trap/retry context, abstains
    otherwise; scores candidates by a trivial fixed rule. Lets the whole
    server/replay plumbing be exercised and asserted without a served model."""

    def __init__(self, *_a: Any, **_k: Any) -> None:
        self.model = "stub"

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        user = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        if '"scores"' in user or "score each candidate" in user.lower():
            import re
            idx = [int(m) for m in re.findall(r"^candidate (\d+):", user, flags=re.M)]
            n = (max(idx) + 1) if idx else 1
            reply = json.dumps({"scores": [round(1.0 - 0.1 * i, 2) for i in range(n)]})
        elif '"score"' in user or "rate this rollout" in user.lower():
            reply = json.dumps({"score": 0.5, "reason": "stub"})
        elif "RETRY-LOOP" in user or "TRAP" in user:
            reply = json.dumps({"advice": "Stop repeating the failing call; check the last "
                                          "error and change approach (stub advice)."})
        else:
            reply = json.dumps({"advice": ""})
        return reply, {"prompt_tokens": 0, "completion_tokens": 0}
