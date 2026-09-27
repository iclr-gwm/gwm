# SPDX-License-Identifier: MIT
"""GWM endpoint plugin for vLLM."""

from __future__ import annotations

import asyncio
import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from starlette.datastructures import State

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

from vllm_gwm.build.evolve import EvolveManager
from vllm_gwm.collect.store import RolloutStore
from vllm_gwm.config import GwmConfig
from vllm_gwm.runtime.bootstrap import init_gwm_core, parse_gwm_modules
from vllm_gwm.runtime.engine_chat import HttpChatClient
from vllm_gwm.runtime.registry import GraphRegistry
from vllm_gwm.runtime.service import AdvisorService
from vllm_gwm.serving import GwmServingChat
from vllm_gwm.viz.routes import attach_graph_routes
from vllm_gwm.viz.show_graph import GraphViewer


class GwmEndpointPlugin:
    name = "gwm"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter(prefix="/v1/gwm", tags=["gwm"])

        @router.get("/info")
        async def info(request: Request):
            svc = _require_advisor(request)
            return svc.info()

        @router.get("/graphs")
        async def graphs(request: Request):
            svc = _require_advisor(request)
            return svc.list_graphs()

        @router.get("/stats")
        async def stats(request: Request):
            registry: GraphRegistry = request.app.state.gwm_registry
            return {"mediators": registry.stats()}

        @router.post("/advise")
        async def advise(request: Request):
            svc = _require_advisor(request)
            body = await request.json()
            # Must not run inline on the API event loop: the mediator's judge
            # calls back into this same server over HTTP, and a blocked loop
            # can never serve that request -- the whole API server deadlocks.
            # serving.py offloads the identical calls for this reason.
            return await asyncio.to_thread(
                svc.advise,
                body.get("conversation_flow") or [],
                preset=body.get("preset"),
                domain=body.get("domain"),
                graph=body.get("graph") or body.get("adapter"),
                episode_id=body.get("episode_id"),
            )

        @router.post("/select")
        async def select(request: Request):
            svc = _require_advisor(request)
            body = await request.json()
            # Offloaded for the same reason as /advise above.
            return await asyncio.to_thread(
                svc.select,
                body.get("conversation_flows") or [],
                mode=body.get("mode") or "auto",
                preset=body.get("preset"),
                domain=body.get("domain"),
                graph=body.get("graph") or body.get("adapter"),
            )

        @router.post("/feedback")
        async def feedback(request: Request):
            store = _require_store(request)
            body = await request.json()
            store.record_feedback(
                episode_id=str(body.get("episode_id") or ""),
                success=bool(body.get("success")),
                source="explicit",
            )
            evolve = getattr(request.app.state, "gwm_evolve", None)
            adapter = str(body.get("adapter") or body.get("graph") or "")
            if evolve is not None and adapter:
                evolve.maybe_trigger(adapter)
            return {"ok": True}

        @router.get("/builds")
        async def builds(request: Request):
            evolve = getattr(request.app.state, "gwm_evolve", None)
            if evolve is None:
                return {"jobs": []}
            return evolve.status()

        @router.post("/evolve")
        async def evolve(request: Request):
            mgr = getattr(request.app.state, "gwm_evolve", None)
            if mgr is None:
                raise HTTPException(status_code=503, detail="GWM evolve not initialized")
            body = await request.json()
            adapter = str(body.get("adapter") or body.get("graph") or "")
            force = bool(body.get("force", True))
            job = mgr.trigger(adapter, force=force) if force else mgr.maybe_trigger(adapter)
            if job is None:
                return {"ok": False, "job": None, "reason": "not triggered"}
            return {"ok": True, "job": job.__dict__}

        @router.post("/rollback")
        async def rollback(request: Request):
            registry: GraphRegistry = request.app.state.gwm_registry
            body = await request.json()
            name = str(body.get("adapter") or "")
            version = str(body.get("version_path") or "")
            if not name or not version:
                raise HTTPException(
                    status_code=400, detail="adapter and version_path required"
                )
            registry.activate_version(name, Path(version))
            return {"ok": True, "adapter": name, "version_path": version}

        app.include_router(router)
        attach_graph_routes(app)

        @app.post("/v1/chat/completions/gwm-batch")
        async def gwm_batch(request: Request):
            from vllm_gwm.batch import handle_gwm_batch

            return await handle_gwm_batch(request)

    async def init_state(
        self, engine_client: Any | None, state: State, args: Namespace
    ) -> None:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        cfg = GwmConfig.from_env()
        if bool(getattr(args, "gwm_show_graph", False)):
            cfg.show_graph = True
        registered = parse_gwm_modules(args)
        if engine_client is None:
            init_gwm_core(state, cfg=cfg, registered=registered)
            return
        model_name = args.served_model_name[0] if args.served_model_name else args.model
        base_url = f"http://127.0.0.1:{getattr(args, 'port', 8000)}/v1"
        judge_url = os.getenv("VLLM_GWM_JUDGE_BASE_URL", "").strip()
        judge_model = os.getenv("VLLM_GWM_JUDGE_MODEL", "").strip()
        judge_preset = os.getenv("VLLM_GWM_JUDGE_API_PRESET", "").strip()
        if judge_preset:
            from vllm_gwm.apis.factory import build_api

            llm = build_api(preset=judge_preset, model=judge_model or model_name)
        elif judge_url:
            llm = HttpChatClient(
                judge_url,
                judge_model or model_name,
                temperature=0.0,
                timeout=float(os.getenv("VLLM_GWM_JUDGE_TIMEOUT", "180") or 180),
            )
        else:
            llm = HttpChatClient(
                base_url,
                judge_model or model_name,
                temperature=0.0,
                timeout=float(os.getenv("VLLM_GWM_JUDGE_TIMEOUT", "180") or 180),
            )
        init_gwm_core(
            state,
            cfg=cfg,
            registered=registered,
            judge_llm=llm,
            default_model=model_name,
        )
        inner = getattr(state, "openai_serving_chat", None)
        if isinstance(inner, OpenAIServingChat):
            state.openai_serving_chat = GwmServingChat(
                inner,
                state.gwm_advisor,
                state.gwm_store,
                max_attempts=cfg.max_attempts,
                fail_open=cfg.fail_open,
                evolve=state.gwm_evolve,
            )


def _require_advisor(request: Request) -> AdvisorService:
    svc = getattr(request.app.state, "gwm_advisor", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="GWM advisor not initialized")
    return svc


def _require_store(request: Request) -> RolloutStore:
    store = getattr(request.app.state, "gwm_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="GWM store not initialized")
    return store
