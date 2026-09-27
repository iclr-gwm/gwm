# SPDX-License-Identifier: MIT
"""Protocols for remote chat / policy backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatBackend(Protocol):
    """Sync chat client used by the harness Mediator."""

    model: str

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]: ...


@runtime_checkable
class PolicyBackend(Protocol):
    """Async policy generation used by GwmServingChat."""

    model: str

    @property
    def supports_n(self) -> bool: ...

    @property
    def supports_temperature(self) -> bool: ...

    async def create_chat_completion(
        self, request: Any, raw_request: Any | None = None
    ) -> Any: ...
