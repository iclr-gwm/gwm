# SPDX-License-Identifier: Apache-2.0
"""Same-model chat client with bypass for harness/judge calls."""

from __future__ import annotations

import contextvars
import json
from typing import Any, Protocol

from vllm_gwm.harness.llm import ChatClient, StubChat

_BYPASS = contextvars.ContextVar("gwm_bypass", default=False)
_DEPTH = contextvars.ContextVar("gwm_depth", default=0)


def bypass_context(*, enabled: bool = True, depth_delta: int = 0):
    """Context manager marking internal harness calls that must skip GWM."""

    class _Ctx:
        def __enter__(self):
            self._tok_b = _BYPASS.set(enabled)
            self._tok_d = _DEPTH.set(_DEPTH.get() + depth_delta)
            return self

        def __exit__(self, *_):
            _BYPASS.reset(self._tok_b)
            _DEPTH.reset(self._tok_d)

    return _Ctx()


def is_bypass_active() -> bool:
    return _BYPASS.get() or _DEPTH.get() > 0


def is_http_bypass(headers: Any | None) -> bool:
    """True when the in-process contextvar is set or the loopback header is present."""
    if is_bypass_active():
        return True
    if headers is None:
        return False
    getter = getattr(headers, "get", None)
    raw = getter("x-vllm-gwm-bypass") if callable(getter) else None
    if raw is None and isinstance(headers, dict):
        raw = headers.get("x-vllm-gwm-bypass") or headers.get("X-VLLM-GWM-Bypass")
    value = str(raw or "").strip().lower()
    return value in ("1", "true", "yes", "on")


class EngineChatBackend(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]: ...


class EngineChatClient:
    """ChatClient that delegates to an in-process vLLM backend with bypass."""

    def __init__(
        self,
        backend: EngineChatBackend,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        disable_thinking: bool = True,
        stub: bool = False,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        if stub:
            self._stub = StubChat()
        else:
            self._stub = None

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        if self._stub is not None:
            return self._stub.chat(messages)
        with bypass_context(enabled=True, depth_delta=1):
            return self.backend.chat(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )


class HttpChatClient(ChatClient):
    """HTTP OpenAI client; adds bypass header for external vLLM loops."""

    def chat(self, messages: list[dict[str, str]]):
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        headers = {
            "Content-Type": "application/json",
            "X-VLLM-GWM-Bypass": "1",
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode())
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return text, (out.get("usage") or {})
