# SPDX-License-Identifier: MIT
"""Generic OpenAI-compatible Chat Completions client."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from vllm_gwm.apis.throttle import QuotaGate, ThrottleConfig, is_quota_error
from vllm_gwm.apis.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    UsageInfo,
)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


# Request fields forwarded verbatim to the upstream API. These are commonly
# supported by OpenAI-compatible backends; dropping them silently changes the caller's
# semantics (a client asking for JSON mode simply did not get it).
PASSTHROUGH_FIELDS = ("response_format", "stop", "seed", "user", "tools", "tool_choice")


def throttle_config_from_env() -> ThrottleConfig:
    return ThrottleConfig(
        max_retries=_env_int("VLLM_GWM_API_MAX_RETRIES", 6),
        retry_step_sec=_env_float("VLLM_GWM_API_RETRY_STEP_SEC", 10.0),
        quota_slow_percent=_env_float("VLLM_GWM_API_QUOTA_SLOW_PERCENT", 80.0),
        timeout=_env_float("VLLM_GWM_API_TIMEOUT", 180.0),
    )


class OpenAIAPI:
    """POST ``{api_base}/chat/completions`` with Bearer auth."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str | None = None,
        auth: str = "bearer",
        max_tokens: int = 700,
        temperature: float = 0.0,
        supports_n: bool = True,
        supports_temperature: bool = True,
        use_max_completion_tokens: bool = False,
        extra_body: dict[str, Any] | None = None,
        throttle: ThrottleConfig | None = None,
        timeout: float | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "EMPTY").strip()
        self.auth = auth.strip().lower()
        self.default_max_tokens = _env_int("VLLM_GWM_API_MAX_TOKENS", max_tokens)
        self.default_temperature = temperature
        self._supports_n = supports_n
        self._supports_temperature = supports_temperature
        self.use_max_completion_tokens = use_max_completion_tokens
        self.extra_body = dict(extra_body or {})
        cfg = throttle or throttle_config_from_env()
        if timeout is not None:
            cfg = ThrottleConfig(
                max_retries=cfg.max_retries,
                retry_step_sec=cfg.retry_step_sec,
                quota_slow_percent=cfg.quota_slow_percent,
                timeout=timeout,
            )
        self._throttle = cfg
        host = self.api_base.split("://", 1)[-1].split("/", 1)[0]
        self._gate = QuotaGate((host, self.api_key), cfg)

    @property
    def supports_n(self) -> bool:
        return self._supports_n

    @property
    def supports_temperature(self) -> bool:
        return self._supports_temperature

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth == "api-key":
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        stream: bool = False,
        passthrough: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        for key, value in (passthrough or {}).items():
            if value is not None:
                body[key] = value
        if stream:
            body["stream"] = True
        tok = max_tokens if max_tokens is not None else self.default_max_tokens
        if self.use_max_completion_tokens:
            body["max_completion_tokens"] = tok
        elif tok:
            body["max_tokens"] = tok
        if self._supports_temperature and temperature is not None:
            body["temperature"] = temperature
        elif self._supports_temperature and self.default_temperature is not None:
            body["temperature"] = self.default_temperature
        if self._supports_n and n is not None and n > 1:
            body["n"] = n
        body.update(self.extra_body)
        return body

    def _url(self) -> str:
        if self.api_base.endswith("/chat/completions"):
            return self.api_base
        return f"{self.api_base}/chat/completions"

    def _post_json(self, body: dict[str, Any]) -> tuple[int, dict[str, Any], Any]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self._url(), data=data, headers=self._headers(), method="POST"
        )
        attempt = 0
        while True:
            attempt += 1
            self._gate.acquire()
            try:
                with urllib.request.urlopen(req, timeout=self._throttle.timeout) as resp:
                    raw = resp.read().decode()
                    headers = resp.headers
                    out = json.loads(raw) if raw else {}
                    self._gate.note_success(headers)
                    return resp.status, out, headers
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if is_quota_error(exc.code, detail) and attempt < self._throttle.max_retries:
                    delay = self._gate.note_quota_failure(attempt, exc.headers)
                    time.sleep(delay)
                    continue
                try:
                    parsed = json.loads(detail) if detail else {}
                except json.JSONDecodeError:
                    parsed = {"error": detail}
                return exc.code, parsed, exc.headers
            except urllib.error.URLError as exc:
                return 503, {"error": str(exc.reason)}, None
            finally:
                self._gate.release()

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        body = self._build_body(messages)
        status, out, _ = self._post_json(body)
        if status >= 400:
            err = out.get("error", out)
            msg = err if isinstance(err, str) else json.dumps(err)
            raise RuntimeError(f"OpenAI API HTTP {status}: {msg[:500]}")
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return text, out.get("usage") or {}

    def _response_from_openai(self, out: dict[str, Any], model: str) -> ChatCompletionResponse:
        choices = []
        for i, ch in enumerate(out.get("choices") or []):
            msg = ch.get("message") or {}
            choices.append(
                ChatCompletionChoice(
                    index=int(ch.get("index", i)),
                    message=ChatCompletionMessage(
                        role=str(msg.get("role") or "assistant"),
                        content=msg.get("content"),
                    ),
                    finish_reason=ch.get("finish_reason"),
                )
            )
        usage_raw = out.get("usage") or {}
        usage = UsageInfo(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            total_tokens=int(usage_raw.get("total_tokens") or 0),
        )
        return ChatCompletionResponse(
            id=str(out.get("id") or f"chatcmpl-api"),
            created=int(out.get("created") or __import__("time").time()),
            model=str(out.get("model") or model),
            choices=choices,
            usage=usage,
        )

    async def create_chat_completion(
        self, request: Any, raw_request: Any | None = None
    ) -> ChatCompletionResponse | ErrorResponse:
        if isinstance(request, ChatCompletionRequest):
            req = request
        elif hasattr(request, "model_dump"):
            req = ChatCompletionRequest.model_validate(request.model_dump())
        elif isinstance(request, dict):
            req = ChatCompletionRequest.model_validate(request)
        else:
            req = ChatCompletionRequest.model_validate(
                {
                    "model": getattr(request, "model", self.model),
                    "messages": getattr(request, "messages", []),
                    "temperature": getattr(request, "temperature", None),
                    "max_tokens": getattr(request, "max_tokens", None),
                    "n": getattr(request, "n", 1),
                    "stream": getattr(request, "stream", False),
                    "vllm_xargs": getattr(request, "vllm_xargs", None),
                }
            )

        # This server is bound to one API/preset: the client-supplied model name
        # is advisory only. Honouring it would forward an arbitrary name to a
        # per-model endpoint, and it lets clients (LiteLLM) apply
        # provider-specific param rules for a model we are not actually calling.
        req.model = self.model
        extra = {
            f: getattr(req, f, None)
            for f in PASSTHROUGH_FIELDS
            if getattr(req, f, None) is not None
        }
        n = int(req.n or 1)
        if not self._supports_n and n > 1:
            choices: list[ChatCompletionChoice] = []
            total_usage = UsageInfo()
            for i in range(n):
                body = self._build_body(
                    req.messages,
                    model=req.model or self.model,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    n=1,
                    passthrough=extra,
                )
                status, out, _ = await asyncio.to_thread(self._post_json, body)
                if status >= 400:
                    err = out.get("error", out)
                    msg = err if isinstance(err, str) else json.dumps(err)
                    return ErrorResponse(message=msg[:500], type="APIError", code=status)
                one = self._response_from_openai(out, req.model or self.model)
                for ch in one.choices:
                    ch.index = len(choices)
                    choices.append(ch)
                if one.usage:
                    total_usage.prompt_tokens += one.usage.prompt_tokens
                    total_usage.completion_tokens += one.usage.completion_tokens
                    total_usage.total_tokens += one.usage.total_tokens
            return ChatCompletionResponse(
                model=req.model or self.model,
                choices=choices,
                usage=total_usage,
            )

        body = self._build_body(
            req.messages,
            model=req.model or self.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            n=n if self._supports_n else 1,
            passthrough=extra,
        )
        status, out, _ = await asyncio.to_thread(self._post_json, body)
        if status >= 400:
            err = out.get("error", out)
            msg = err if isinstance(err, str) else json.dumps(err)
            return ErrorResponse(message=msg[:500], type="APIError", code=status)
        return self._response_from_openai(out, req.model or self.model)
