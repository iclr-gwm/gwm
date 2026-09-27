# SPDX-License-Identifier: MIT
"""The GWM HTTP routes must not run the mediator on the API event loop.

``svc.advise`` / ``svc.select`` call the judge, which loops back into this same
server over HTTP. Running them inline on the event loop means the loopback can
never be served: the request hangs forever and takes the whole API server with
it. ``serving.py`` already offloads the identical calls with
``asyncio.to_thread``; these tests pin the same requirement for the routes.
"""

import ast
import pathlib

import pytest

_PLUGIN = pathlib.Path(__file__).resolve().parents[1] / "src/vllm_gwm/plugin.py"


def _route_functions() -> dict[str, ast.AsyncFunctionDef]:
    tree = ast.parse(_PLUGIN.read_text())
    out: dict[str, ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            out[node.name] = node
    return out


@pytest.mark.parametrize("route", ["advise", "select"])
def test_route_offloads_mediator_call(route):
    fn = _route_functions()[route]
    src = ast.dump(fn)
    assert "to_thread" in src, (
        f"/{route} must offload the mediator call with asyncio.to_thread; "
        "running it inline deadlocks the API server against its own judge"
    )


@pytest.mark.parametrize("route", ["advise", "select"])
def test_route_does_not_call_service_inline(route):
    """The bare `svc.<route>(...)` call must be awaited, never evaluated inline."""
    fn = _route_functions()[route]
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            call = node.value
            func = call.func
            # A direct `return svc.advise(...)` is the deadlocking shape.
            if isinstance(func, ast.Attribute) and func.attr == route:
                pytest.fail(
                    f"/{route} returns svc.{route}(...) inline on the event loop"
                )
