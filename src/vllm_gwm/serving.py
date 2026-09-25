# SPDX-License-Identifier: Apache-2.0
"""GWM-aware chat serving proxy."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from typing import Any

from fastapi import Request

from vllm_gwm.build.evolve import EvolveManager
from vllm_gwm.collect.flow import messages_to_flow
from vllm_gwm.collect.store import RolloutStore
from vllm_gwm.protocol import GwmRequestOptions
from vllm_gwm.runtime.engine_chat import is_http_bypass
from vllm_gwm.runtime.service import AdvisorService


def _is_error_response(result: Any) -> bool:
    if result is None:
        return False
    if hasattr(result, "code") and hasattr(result, "message"):
        code = getattr(result, "code", None)
        return code is not None and int(code) >= 400
    return False


def _response_choices(result: Any) -> list[Any]:
    return list(getattr(result, "choices", None) or [])


def _choice_content(choice: Any) -> str:
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message") or {}
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return str(getattr(msg, "content", None) or "")


def _with_guidance(messages: Any, guidance: str) -> list:
    """Attach GWM advice without appending a trailing system turn.

    Several chat templates (Qwen3.x among them) reject a system message that is
    not the first turn and fail the whole request with
    ``System message must be at the beginning``, which silently disables the
    advice arm for those policies. Merge the guidance into the leading system
    message instead, and only insert a new one when the conversation has none.
    """
    out = [dict(m) if isinstance(m, dict) else m for m in messages]
    first = out[0] if out else None
    if isinstance(first, dict) and str(first.get("role")) == "system":
        existing = str(first.get("content") or "")
        first["content"] = f"{existing}\n\n{guidance}" if existing else guidance
        return out
    return [{"role": "system", "content": guidance}, *out]


def _deep_copy_request(request: Any) -> Any:
    if hasattr(request, "model_copy"):
        return request.model_copy(deep=True)
    return copy.deepcopy(request)


class GwmServingChat:
    """Applies GWM advice/selection gates over a policy backend."""

    def __init__(
        self,
        inner: Any,
        advisor: AdvisorService,
        store: RolloutStore | None,
        *,
        max_attempts: int = 3,
        fail_open: bool = True,
        evolve: EvolveManager | None = None,
    ):
        self._inner = inner
        self._advisor = advisor
        self._store = store
        self._max_attempts = max_attempts
        self._fail_open = fail_open
        self._evolve = evolve

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def _supports_n(self) -> bool:
        return bool(getattr(self._inner, "supports_n", True))

    @property
    def _supports_temperature(self) -> bool:
        return bool(getattr(self._inner, "supports_temperature", True))

    async def create_chat_completion(
        self, request: Any, raw_request: Request | None = None
    ):
        if is_http_bypass(None if raw_request is None else raw_request.headers):
            return await self._inner.create_chat_completion(request, raw_request)

        xargs = getattr(request, "vllm_xargs", None)
        opts = GwmRequestOptions.from_xargs(xargs)
        if not opts.enabled:
            return await self._inner.create_chat_completion(request, raw_request)

        if getattr(request, "stream", False):
            return await self._gwm_stream(request, raw_request, opts)
        if opts.is_select or opts.k > 1:
            return await self._gwm_select(request, raw_request, opts)
        return await self._gwm_advise(request, raw_request, opts)

    async def _gwm_advise(
        self,
        request: Any,
        raw_request: Request | None,
        opts: GwmRequestOptions,
    ):
        base_flow = messages_to_flow(request.messages)
        guidance = ""
        best: Any = None
        best_score = -1.0
        for _attempt in range(self._max_attempts):
            req = _deep_copy_request(request)
            req.n = 1
            req.vllm_xargs = None
            if guidance:
                req.messages = _with_guidance(request.messages, guidance)
            result = await self._inner.create_chat_completion(req, raw_request)
            if _is_error_response(result):
                if self._fail_open and best is not None:
                    return best
                return result
            choices = _response_choices(result)
            if not choices:
                if self._fail_open and best is not None:
                    return best
                return result
            flow = base_flow + messages_to_flow(
                [{"role": "assistant", "content": _choice_content(choices[0])}]
            )
            approval = await asyncio.to_thread(
                self._advisor.approve_candidate,
                flow,
                preset=opts.preset or None,
                domain=opts.domain or None,
                graph=opts.graph or opts.adapter or None,
                episode_id=opts.episode_id or None,
            )
            score = float(approval.get("success_score") or 0.0)
            if score >= best_score:
                best_score = score
                best = result
            if approval.get("approved"):
                self._maybe_collect(flow, opts, success=True)
                return result
            guidance = approval.get("advice") or ""
            if not guidance:
                break
        if best is not None:
            self._maybe_collect(base_flow, opts, success=False)
            return best
        try:
            from vllm_gwm.apis.types import ErrorResponse

            return ErrorResponse(
                message="GWM could not approve a candidate",
                type="BadRequest",
                code=400,
            )
        except ImportError:
            from vllm.entrypoints.serve.engine.protocol import ErrorResponse as VllmError

            return VllmError(
                message="GWM could not approve a candidate",
                type="BadRequest",
                code=400,
            )

    async def _gwm_select(
        self,
        request: Any,
        raw_request: Request | None,
        opts: GwmRequestOptions,
    ):
        base_flow = messages_to_flow(request.messages)
        k = opts.k
        preset_anchor = bool(getattr(self._advisor.cfg, "greedy_anchor", True))
        try:
            preset_anchor = bool(
                getattr(
                    self._advisor._preset(opts.preset or None),
                    "greedy_anchor",
                    True,
                )
            )
        except Exception:
            pass
        _env = os.getenv("VLLM_GWM_GREEDY_ANCHOR", "").strip().lower()
        greedy_anchor = (_env in ("1", "true", "yes", "on")) if _env else preset_anchor
        greedy_anchor = greedy_anchor and self._supports_temperature

        if bool(getattr(self._advisor.cfg, "adaptive_k", False)) and k > 1:
            cls = await asyncio.to_thread(
                self._advisor.classify,
                base_flow,
                preset=opts.preset or None,
                domain=opts.domain or None,
                graph=opts.graph or opts.adapter or None,
            )
            if not cls.get("abstain") and not cls.get("is_trap"):
                greedy_req = _deep_copy_request(request)
                greedy_req.n = 1
                if greedy_anchor:
                    greedy_req.temperature = 0.0
                greedy_req.vllm_xargs = None
                res = await self._inner.create_chat_completion(greedy_req, raw_request)
                if _is_error_response(res) or not _response_choices(res):
                    return res
                self._maybe_collect(
                    base_flow
                    + messages_to_flow(
                        [{"role": "assistant", "content": _choice_content(res.choices[0])}]
                    ),
                    opts,
                    success=True,
                )
                return res

        if greedy_anchor and k > 1:
            greedy_req = _deep_copy_request(request)
            greedy_req.n = 1
            greedy_req.temperature = 0.0
            greedy_req.vllm_xargs = None
            sample_req = _deep_copy_request(request)
            sample_req.n = k - 1
            sample_req.vllm_xargs = None
            greedy_res = await self._inner.create_chat_completion(greedy_req, raw_request)
            sample_res = await self._inner.create_chat_completion(sample_req, raw_request)
            for r in (greedy_res, sample_res):
                if _is_error_response(r):
                    return r
                if not _response_choices(r):
                    return r
            result = greedy_res
            result.choices = list(greedy_res.choices) + list(sample_res.choices)
            for i, ch in enumerate(result.choices):
                ch.index = i
        else:
            req = _deep_copy_request(request)
            # Always ask for k candidates. Backends without native ``n``
            # emulate it by looping; forcing n=1 here
            # silently collapsed best-of-K to a single candidate, so every
            # select event degraded to select_single.
            req.n = k
            req.vllm_xargs = None
            result = await self._inner.create_chat_completion(req, raw_request)
            if _is_error_response(result):
                return result
            if not _response_choices(result):
                return result

        flows = []
        for choice in _response_choices(result):
            flows.append(
                base_flow
                + messages_to_flow(
                    [{"role": "assistant", "content": _choice_content(choice)}]
                )
            )
        graph = opts.graph or opts.adapter or None
        if getattr(self._advisor.cfg, "extra_graphs", ""):
            graph = await asyncio.to_thread(
                self._advisor.best_graph,
                base_flow,
                preset=opts.preset or None,
                domain=opts.domain or None,
                primary=graph,
            )
        sel = await asyncio.to_thread(
            self._advisor.select,
            flows,
            preset=opts.preset or None,
            domain=opts.domain or None,
            graph=graph,
            episode_id=opts.episode_id or None,
        )
        scores = [float(r.get("success_score") or 0.0) for r in sel.get("results", [])]
        if not scores:
            return result
        winner = max(range(len(scores)), key=lambda i: scores[i])
        out = _deep_copy_request(result)
        out.choices = [result.choices[winner]]
        self._maybe_collect(flows[winner], opts, success=True)
        if getattr(self._advisor.cfg, "accrete_self", False) or getattr(
            self._advisor.cfg, "mint_states", False
        ):
            await asyncio.to_thread(
                self._advisor.accrete,
                base_flow,
                _choice_content(result.choices[winner]),
                preset=opts.preset or None,
                domain=opts.domain or None,
                graph=graph,
                episode_id=opts.episode_id or None,
            )
        return out

    async def _gwm_stream(
        self,
        request: Any,
        raw_request: Request | None,
        opts: GwmRequestOptions,
    ):
        non_stream = _deep_copy_request(request)
        non_stream.stream = False
        result = await self.create_chat_completion(non_stream, raw_request)
        if _is_error_response(result):
            return result
        choices = _response_choices(result)
        if not choices:
            return result

        async def _replay():
            chunk = {
                "id": getattr(result, "id", "chatcmpl-gwm"),
                "object": "chat.completion.chunk",
                "created": getattr(result, "created", 0),
                "model": getattr(result, "model", ""),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": _choice_content(choices[0]),
                        },
                        "finish_reason": getattr(choices[0], "finish_reason", "stop"),
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return _replay()

    def _maybe_collect(
        self, flow: list[dict[str, Any]], opts: GwmRequestOptions, *, success: bool
    ) -> None:
        if self._store is None:
            return
        if not opts.collect and self._advisor.cfg.collect_mode != "all":
            return
        self._store.append(
            flow,
            episode_id=opts.episode_id,
            success=opts.outcome_success if opts.outcome_success is not None else success,
            source=opts.outcome_source or "client",
            finalize=opts.finalize,
        )
        if opts.finalize and self._evolve is not None:
            adapter = opts.graph or opts.adapter or ""
            self._evolve.maybe_trigger(adapter)
