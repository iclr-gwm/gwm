# SPDX-License-Identifier: Apache-2.0
"""Adapter wrapping vLLM OpenAIServingChat as a PolicyBackend."""

from __future__ import annotations

from typing import Any


class VllmPolicyBackend:
    """Thin async wrapper around in-process vLLM chat serving."""

    def __init__(self, inner: Any):
        self._inner = inner

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "model")

    @property
    def supports_n(self) -> bool:
        return True

    @property
    def supports_temperature(self) -> bool:
        return True

    async def create_chat_completion(
        self, request: Any, raw_request: Any | None = None
    ) -> Any:
        return await self._inner.create_chat_completion(request, raw_request)
