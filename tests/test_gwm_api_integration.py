# SPDX-License-Identifier: Apache-2.0
"""Offline integration tests: API backend + full GWM harness stack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from vllm_gwm.apis.stub import StubAPI
from vllm_gwm.apis.types import ChatCompletionRequest, ErrorResponse
from vllm_gwm.config import GwmConfig
from vllm_gwm.plugin import GwmEndpointPlugin
from vllm_gwm.runtime.bootstrap import init_gwm_core
from vllm_gwm.serving import GwmServingChat

from conftest import FIX, TINY

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _error_json(result: ErrorResponse) -> JSONResponse:
    return JSONResponse(result.model_dump(), status_code=int(result.code))


def _build_api_app(tmp_path: Path, tiny_preset) -> FastAPI:
    app = FastAPI(title="vllm-gwm-api-test")
    cfg = GwmConfig(preset_default="tiny", audit_log=False, data_root=tmp_path)
    policy = StubAPI()
    judge = StubAPI()
    init_gwm_core(
        app.state,
        cfg=cfg,
        registered={"tiny": TINY},
        judge_llm=judge,
        default_model="stub",
    )
    app.state.gwm_advisor._presets["tiny"] = tiny_preset
    handler = GwmServingChat(
        policy,
        app.state.gwm_advisor,
        app.state.gwm_store,
        max_attempts=cfg.max_attempts,
        fail_open=cfg.fail_open,
    )
    app.state.openai_serving_chat = handler
    GwmEndpointPlugin().attach_router(app)

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "api", "model": "stub"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        req = ChatCompletionRequest.model_validate(body)
        result = await handler.create_chat_completion(req, request)
        if isinstance(result, ErrorResponse):
            return _error_json(result)
        if body.get("stream") and hasattr(result, "__aiter__"):
            return StreamingResponse(result, media_type="text/event-stream")
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        if body.get("stream"):
            async def _stream():
                content = (
                    (payload.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content")
                    or ""
                )
                chunk = {
                    "id": payload.get("id"),
                    "object": "chat.completion.chunk",
                    "model": payload.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": content},
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


@pytest.fixture()
def api_client(tmp_path, patch_scorer, tiny_preset):
    return TestClient(_build_api_app(tmp_path, tiny_preset))


@pytest.fixture()
def crm_flow():
    return json.loads((FIX / "crm_retry_flow.json").read_text())


def _gwm_xargs(**kwargs):
    return {"gwm": {"adapter": "tiny", "preset": "tiny", **kwargs}}


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["backend"] == "api"


def test_gwm_info_and_graphs(api_client):
    info = api_client.get("/v1/gwm/info")
    assert info.status_code == 200
    body = info.json()
    assert body["server"] == "vllm_gwm"
    graph_names = [g["name"] if isinstance(g, dict) else g for g in body["graphs"]]
    assert "tiny" in graph_names

    graphs = api_client.get("/v1/gwm/graphs")
    assert graphs.status_code == 200
    gbody = graphs.json()
    registered = [g["name"] if isinstance(g, dict) else g for g in gbody.get("graphs") or []]
    assert "tiny" in registered


def test_gwm_evolve_force_endpoint(api_client, monkeypatch):
    from vllm_gwm.build.evolve import BuildJob, EvolveManager

    captured = {}

    def fake_trigger(self, adapter, *, force=True):
        captured["adapter"] = adapter
        captured["force"] = force
        return BuildJob(
            id="joblive", status="running", adapter=adapter, started_at="now"
        )

    monkeypatch.setattr(EvolveManager, "trigger", fake_trigger)
    resp = api_client.post("/v1/gwm/evolve", json={"adapter": "tiny", "force": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["job"]["id"] == "joblive"
    assert captured == {"adapter": "tiny", "force": True}


def test_gwm_stats(api_client):
    resp = api_client.get("/v1/gwm/stats")
    assert resp.status_code == 200
    assert "mediators" in resp.json()


def test_gwm_advise_injects_on_retry_trap(api_client, crm_flow):
    resp = api_client.post(
        "/v1/gwm/advise",
        json={"conversation_flow": crm_flow, "preset": "tiny", "graph": "tiny", "episode_id": "ep-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["injected"] is True
    assert body["block"].startswith("[World-Model guidance]\n")
    harness = body.get("harness") or {}
    assert harness.get("llm_calls", 0) >= 1


def test_gwm_classify_trap_state(api_client, crm_flow):
    resp = api_client.post(
        "/v1/gwm/advise",
        json={"conversation_flow": crm_flow, "preset": "tiny", "graph": "tiny"},
    )
    state = resp.json().get("state")
    advisor = api_client.app.state.gwm_advisor
    cls = advisor.classify(crm_flow, preset="tiny", graph="tiny")
    assert cls.get("state") == state
    assert cls.get("is_trap") is True


def test_gwm_select_joint(api_client, crm_flow):
    tools = [
        ("describe", {"table": "Account"}),
        ("execute", {"query": "SELECT 1"}),
        ("describe", {"table": "Case"}),
        ("respond", {"answer": "007"}),
    ]
    flows = [
        crm_flow
        + [{"type": "ai_message", "content": "", "tool_calls": [{"name": t, "args": a}]}]
        for t, a in tools
    ]
    resp = api_client.post(
        "/v1/gwm/select",
        json={"conversation_flows": flows, "preset": "tiny", "graph": "tiny", "mode": "joint"},
    )
    assert resp.status_code == 200
    results = resp.json().get("results") or []
    assert len(results) == 4
    scores = [r.get("success_score") for r in results]
    assert len(scores) == 4
    assert all(s is not None for s in scores)
    assert max(scores) >= 0.5


def test_gwm_select_single(api_client, crm_flow):
    resp = api_client.post(
        "/v1/gwm/select",
        json={"conversation_flows": [crm_flow], "preset": "tiny", "graph": "tiny", "mode": "single"},
    )
    assert resp.status_code == 200
    results = resp.json().get("results") or []
    assert len(results) == 1
    assert "success_score" in results[0]


def test_gwm_feedback(api_client):
    resp = api_client.post(
        "/v1/gwm/feedback",
        json={"episode_id": "ep-feedback", "success": True},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_chat_plain_no_gwm(api_client):
    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "stub",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["message"]["content"]
    assert "stub policy response" in text


def test_chat_gwm_advise_path(api_client):
    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "stub",
            "messages": [{"role": "user", "content": "What is your next step?"}],
            "vllm_xargs": _gwm_xargs(mode="advise", k=1, episode_id="chat-advise"),
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"]


def test_chat_gwm_select_k4(api_client, crm_flow):
    messages = [{"role": "user", "content": "What is your next step?"}]
    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "stub",
            "messages": messages,
            "n": 4,
            "temperature": 0.7,
            "vllm_xargs": _gwm_xargs(mode="select", k=4, episode_id="chat-select"),
        },
    )
    assert resp.status_code == 200
    choices = resp.json().get("choices") or []
    assert len(choices) == 1


def test_chat_bypass_header(api_client):
    resp = api_client.post(
        "/v1/chat/completions",
        headers={"X-VLLM-GWM-Bypass": "true"},
        json={
            "model": "stub",
            "messages": [{"role": "user", "content": "bypass me"}],
            "vllm_xargs": _gwm_xargs(mode="advise", k=1),
        },
    )
    assert resp.status_code == 200
    assert "stub policy response" in resp.json()["choices"][0]["message"]["content"]


def test_chat_gwm_stream(api_client):
    resp = api_client.post(
        "/v1/chat/completions",
        json={
            "model": "stub",
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
            "vllm_xargs": _gwm_xargs(mode="advise", k=1),
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "[DONE]" in resp.text


def test_approve_candidate_via_advisor(api_client, crm_flow):
    advisor = api_client.app.state.gwm_advisor
    approval = advisor.approve_candidate(crm_flow, preset="tiny", graph="tiny")
    assert approval["approved"] is False
    assert approval["advice"]
    assert approval["success_score"] >= 0.0


def test_policy_mock_vllm_with_gwm_rest(patch_scorer, tiny_preset, crm_flow, tmp_path):
    """Policy via MockVLLM HTTP; judge via StubAPI — full harness on /gwm/advise."""
    sys.path.insert(0, str(SCRIPTS))
    from mock_vllm import MockVLLM  # noqa: E402

    from vllm_gwm.apis.openai import OpenAIAPI
    from vllm_gwm.apis.throttle import ThrottleConfig

    with MockVLLM("mock-gwm") as base_url:
        app = FastAPI()
        cfg = GwmConfig(preset_default="tiny", audit_log=False, data_root=tmp_path)
        policy = OpenAIAPI(
            api_base=base_url,
            model="mock-gwm",
            api_key="EMPTY",
            throttle=ThrottleConfig(max_retries=2, retry_step_sec=0.01, timeout=5.0),
        )
        init_gwm_core(
            app.state,
            cfg=cfg,
            registered={"tiny": TINY},
            judge_llm=StubAPI(),
            default_model="mock-gwm",
        )
        app.state.gwm_advisor._presets["tiny"] = tiny_preset
        GwmEndpointPlugin().attach_router(app)

        client = TestClient(app)
        advise = client.post(
            "/v1/gwm/advise",
            json={"conversation_flow": crm_flow, "preset": "tiny", "graph": "tiny"},
        )
        assert advise.status_code == 200
        assert advise.json()["injected"] is True

        text, _ = policy.chat([{"role": "user", "content": "next step?"}])
        assert "mock vLLM policy" in text


try:
    import vllm as _vllm  # noqa: F401

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed (gwm-batch route)")
def test_gwm_batch_route(api_client):
    resp = api_client.post(
        "/v1/chat/completions/gwm-batch",
        json={
            "requests": [
                {
                    "model": "stub",
                    "messages": [{"role": "user", "content": "batch-1"}],
                    "vllm_xargs": _gwm_xargs(mode="advise", k=1),
                },
                {
                    "model": "stub",
                    "messages": [{"role": "user", "content": "batch-2"}],
                    "n": 4,
                    "vllm_xargs": _gwm_xargs(mode="select", k=4),
                },
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body.get("results") or []) == 2
    assert body.get("policy_candidates", 0) >= 5
