# SPDX-License-Identifier: MIT
"""Lightweight OpenAI chat types for API mode (no vLLM import)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = 0.7
    max_tokens: int | None = None
    n: int | None = 1
    stream: bool | None = False
    vllm_xargs: dict[str, Any] | None = None


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str | None = ""


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: UsageInfo | None = None


class ErrorResponse(BaseModel):
    message: str
    type: str = "BadRequest"
    code: int = 400

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"error": super().model_dump(*args, **kwargs)}
