# SPDX-License-Identifier: Apache-2.0
"""Standalone FastAPI server for GWM API mode (no vLLM engine)."""

from __future__ import annotations

import argparse
import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_gwm.apis.factory import build_api
from vllm_gwm.apis.types import ChatCompletionRequest, ErrorResponse
from vllm_gwm.config import GwmConfig
from vllm_gwm.plugin import GwmEndpointPlugin
from vllm_gwm.runtime.bootstrap import init_gwm_core, parse_gwm_modules
from vllm_gwm.serving import GwmServingChat


def _error_json(result: ErrorResponse) -> JSONResponse:
    return JSONResponse(result.model_dump(), status_code=int(result.code))


def create_api_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="vllm-gwm-api")
    cfg = GwmConfig.from_env()
    if getattr(args, "gwm_show_graph", False):
        cfg.show_graph = True

    policy = build_api(
        api_class=getattr(args, "api_class", "openai"),
        preset=getattr(args, "api_preset", None),
        api_base=getattr(args, "api_base", None),
        model=getattr(args, "model", None),
        api_provider=getattr(args, "api_provider", None),
        gemini_region=getattr(args, "gemini_region", None),
    )
    judge = build_api(
        api_class=getattr(args, "judge_api_class", None)
        or getattr(args, "api_class", "openai"),
        preset=getattr(args, "judge_api_preset", None)
        or getattr(args, "api_preset", None),
        api_base=getattr(args, "judge_api_base", None),
        model=getattr(args, "judge_model", None) or policy.model,
        api_provider=getattr(args, "judge_api_provider", None)
        or getattr(args, "api_provider", None),
        gemini_region=getattr(args, "gemini_region", None),
    )

    registered = parse_gwm_modules(args)
    init_gwm_core(
        app.state,
        cfg=cfg,
        registered=registered,
        judge_llm=judge,
        default_model=policy.model,
    )
    store = app.state.gwm_store
    advisor = app.state.gwm_advisor
    handler = GwmServingChat(
        policy,
        advisor,
        store,
        max_attempts=cfg.max_attempts,
        fail_open=cfg.fail_open,
        evolve=app.state.gwm_evolve,
    )
    app.state.openai_serving_chat = handler
    app.state.gwm_policy_api = policy

    plugin = GwmEndpointPlugin()
    plugin.attach_router(app)

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "api", "model": policy.model}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        req = ChatCompletionRequest.model_validate(body)
        result = await handler.create_chat_completion(req, request)
        if isinstance(result, ErrorResponse):
            return _error_json(result)
        if body.get("stream") and hasattr(result, "__aiter__"):
            return StreamingResponse(result, media_type="text/event-stream")
        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        else:
            payload = result
        if body.get("stream"):
            async def _stream():
                chunk = {
                    "id": payload.get("id"),
                    "object": "chat.completion.chunk",
                    "created": payload.get("created"),
                    "model": payload.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": (
                                    (payload.get("choices") or [{}])[0]
                                    .get("message", {})
                                    .get("content")
                                    or ""
                                ),
                            },
                            "finish_reason": (payload.get("choices") or [{}])[0].get(
                                "finish_reason"
                            ),
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")
        return JSONResponse(payload)

    return app


def run_server(args: argparse.Namespace) -> None:
    import uvicorn

    app = create_api_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
