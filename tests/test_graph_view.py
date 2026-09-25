# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from vllm_gwm.cli import consume_gwm_serve_args, normalize_api_server_args
from vllm_gwm.config import GwmConfig
from vllm_gwm.viz.show_graph import GraphViewer, build_payload, resolve_graph

FIX = Path(__file__).resolve().parent / "fixtures"
TINY = FIX / "tiny_graph"


def test_resolve_tiny_transitions():
    path = resolve_graph(TINY)
    assert path.name == "transitions.json"
    assert path.is_file()


def test_build_payload_tiny_graph():
    payload = build_payload(resolve_graph(TINY))
    assert payload["domains"]
    toy = payload["domains"][0]
    assert toy["name"] == "toy"
    ids = {n["id"] for n in toy["nodes"]}
    assert "toy:0" in ids and "toy:1" in ids
    assert toy["n_edges"] >= 2
    traps = [n for n in toy["nodes"] if n["id"] == "toy:0"][0]
    assert traps["fail_rate"] == 0.9
    assert traps["top_tools"]


def test_viewer_from_registered():
    viewer = GraphViewer.from_registered({"tiny": TINY})
    cat = viewer.catalog()
    assert cat["default"] == "tiny"
    assert cat["api_base"] == "/v1/graph"
    assert cat["graphs"][0]["available"] is True
    data = viewer.payload_for("tiny")
    assert data["id"] == "tiny"
    html = viewer.index_html()
    assert "World-Model graph" in html
    assert "/v1/graph" in html
    assert "toy:0" in html


def test_show_graph_config_env(monkeypatch):
    monkeypatch.setenv("VLLM_GWM_SHOW_GRAPH", "1")
    assert GwmConfig.from_env().show_graph is True
    monkeypatch.setenv("VLLM_GWM_SHOW_GRAPH", "0")
    assert GwmConfig.from_env().show_graph is False


def test_consume_gwm_serve_args_from_remainder():
    vllm_args, modules, show = consume_gwm_serve_args(
        ["MODEL", "--gwm-show-graph", "--gwm-modules", "crm=/tmp/crm", "--port", "8000"],
        [],
        False,
    )
    assert show is True
    assert modules == ["crm=/tmp/crm"]
    assert vllm_args == ["MODEL", "--port", "8000"]


def test_normalize_api_server_args_positional_model():
    assert normalize_api_server_args(["google/gemma-4-31b-it", "--port", "8497"]) == [
        "--model",
        "google/gemma-4-31b-it",
        "--port",
        "8497",
    ]
    assert normalize_api_server_args(["--model", "m"]) == ["--model", "m"]
    assert normalize_api_server_args(["--", "--model", "m"]) == ["--model", "m"]
