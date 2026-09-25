# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_gwm.viz.routes import attach_graph_routes
from vllm_gwm.viz.show_graph import GraphViewer

FIX = Path(__file__).resolve().parent / "fixtures"
TINY = FIX / "tiny_graph"


def _app(*, enabled: bool) -> FastAPI:
    app = FastAPI()
    attach_graph_routes(app)
    app.state.gwm_show_graph = enabled
    app.state.gwm_graph_viewer = (
        GraphViewer.from_registered({"tiny": TINY}) if enabled else None
    )
    return app


def test_graph_endpoint_404_when_disabled():
    client = TestClient(_app(enabled=False))
    resp = client.get("/v1/graph")
    assert resp.status_code == 404
    assert "gwm-show-graph" in resp.json()["detail"]


def test_graph_endpoint_html_when_enabled():
    client = TestClient(_app(enabled=True))
    resp = client.get("/v1/graph")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "World-Model graph" in resp.text
    assert "tiny" in resp.text


def test_graph_catalog_and_payload():
    client = TestClient(_app(enabled=True))
    cat = client.get("/v1/graph/api/catalog")
    assert cat.status_code == 200
    body = cat.json()
    assert body["default"] == "tiny"
    assert body["graphs"][0]["id"] == "tiny"

    payload = client.get("/v1/graph/api/graph/tiny")
    assert payload.status_code == 200
    data = payload.json()
    assert data["id"] == "tiny"
    assert data["domains"][0]["name"] == "toy"

    missing = client.get("/v1/graph/api/graph/nope")
    assert missing.status_code == 404
