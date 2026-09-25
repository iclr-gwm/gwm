# SPDX-License-Identifier: Apache-2.0
"""Plugin batch route for GWM-enabled chat completions."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request
from starlette.responses import JSONResponse
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_gwm.protocol import GwmRequestOptions


async def handle_gwm_batch(request: Request) -> JSONResponse:
    body = await request.json()
    entries = body.get("requests") or []
    if not entries:
        return JSONResponse({"error": "requests required"}, status_code=400)
    handler = request.app.state.openai_serving_chat
    if handler is None:
        return JSONResponse({"error": "chat handler unavailable"}, status_code=503)

    sem = asyncio.Semaphore(
        getattr(request.app.state.gwm_cfg, "max_concurrent_gwm", 4)
    )
    results: list[Any] = [None] * len(entries)

    async def _one(idx: int, item: dict[str, Any]):
        async with sem:
            req = ChatCompletionRequest.model_validate(item)
            opts = GwmRequestOptions.from_xargs(req.vllm_xargs)
            if opts.enabled and opts.k > 1 and req.vllm_xargs:
                gwm = dict(req.vllm_xargs.get("gwm") or {})
                gwm["mode"] = "select"
                req.vllm_xargs = {"gwm": gwm}
            out = await handler.create_chat_completion(req, request)
            if hasattr(out, "model_dump"):
                results[idx] = out.model_dump()
            elif hasattr(out, "__aiter__"):
                results[idx] = {"error": "streaming not supported in batch"}
            else:
                results[idx] = out

    await asyncio.gather(*[_one(i, e) for i, e in enumerate(entries)])
    total_k = sum(
        max(1, GwmRequestOptions.from_xargs(e.get("vllm_xargs")).k)
        for e in entries
        if GwmRequestOptions.from_xargs(e.get("vllm_xargs")).enabled
    )
    return JSONResponse({"results": results, "policy_candidates": total_k})
