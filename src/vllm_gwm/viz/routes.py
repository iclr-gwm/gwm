# SPDX-License-Identifier: Apache-2.0
"""FastAPI routes for the transition-graph viewer at ``/v1/graph``."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from vllm_gwm.viz.show_graph import GraphViewer


def require_graph_viewer(request: Request) -> GraphViewer:
    if not getattr(request.app.state, "gwm_show_graph", False):
        raise HTTPException(
            status_code=404,
            detail=(
                "graph visualization disabled; start with --gwm-show-graph "
                "or VLLM_GWM_SHOW_GRAPH=1"
            ),
        )
    viewer = getattr(request.app.state, "gwm_graph_viewer", None)
    if viewer is None:
        raise HTTPException(
            status_code=503, detail="graph visualization not initialized"
        )
    return viewer


def attach_graph_routes(app: FastAPI) -> None:
    """Register ``GET /v1/graph`` (HTML) and JSON catalog/payload APIs."""

    @app.get("/v1/graph")
    async def graph_index(request: Request):
        viewer = require_graph_viewer(request)
        return HTMLResponse(viewer.index_html())

    @app.get("/v1/graph/api/catalog")
    async def graph_catalog(request: Request):
        viewer = require_graph_viewer(request)
        return viewer.catalog()

    @app.get("/v1/graph/api/graph/{gid}")
    async def graph_payload(request: Request, gid: str):
        viewer = require_graph_viewer(request)
        try:
            return viewer.payload_for(gid)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown or unavailable graph '{gid}'"
            ) from exc
