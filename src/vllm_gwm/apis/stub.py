# SPDX-License-Identifier: Apache-2.0
"""Deterministic offline stand-in for API clients."""

from __future__ import annotations

import json
import re
from typing import Any

from vllm_gwm.apis.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    UsageInfo,
)


class StubAPI:
    """Offline API backend mirroring mock_vllm canned replies."""

    model = "stub"

    def __init__(self, *_a: Any, **_k: Any) -> None:
        pass

    @property
    def supports_n(self) -> bool:
        return True

    @property
    def supports_temperature(self) -> bool:
        return True

    def _reply(self, messages: list[dict[str, str]]) -> str:
        sys_ = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
        user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        if '"scores"' in user or "score each candidate" in user.lower():
            idx = [int(m) for m in re.findall(r"^candidate (\d+):", user, flags=re.M)]
            n = (max(idx) + 1) if idx else 1
            return json.dumps({"scores": [round(1.0 - 0.1 * i, 2) for i in range(n)]})
        if '"score"' in user or "rate this rollout" in user.lower():
            return json.dumps({"score": 0.5, "reason": "stub"})
        if "RETRY-LOOP" in user or "TRAP" in user:
            return json.dumps(
                {
                    "advice": "Stop repeating the failing call; check the last "
                    "error and change approach (stub advice)."
                }
            )
        if "World-Model harness agent" in sys_:
            return json.dumps({"advice": ""})
        return "stub policy response"

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        text = self._reply(messages)
        return text, {"prompt_tokens": 0, "completion_tokens": 0}

    async def create_chat_completion(self, request: Any, raw_request: Any | None = None) -> Any:
        messages = getattr(request, "messages", None) or request.get("messages", [])
        n = int(getattr(request, "n", None) or request.get("n") or 1)
        choices = []
        for i in range(max(1, n)):
            text = self._reply(messages)
            choices.append(
                ChatCompletionChoice(
                    index=i,
                    message=ChatCompletionMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            )
        model = getattr(request, "model", None) or request.get("model") or self.model
        return ChatCompletionResponse(
            model=str(model),
            choices=choices,
            usage=UsageInfo(),
        )
