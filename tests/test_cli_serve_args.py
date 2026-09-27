# SPDX-License-Identifier: MIT
"""serve must forward --host/--port/--model to the vLLM api_server.

Regression: these are declared on the API-mode parser, so parse_known_args
consumed them and the vLLM path never re-added them — `serve --port 8790`
silently bound vLLM's default :8000. On a shared host that is either an
EADDRINUSE crash or, worse, a server quietly listening on the wrong port.
"""

import pytest

from vllm_gwm.cli import consume_api_serve_args


def test_port_is_forwarded_to_vllm_args():
    known, rest = consume_api_serve_args(
        ["MODEL", "--host", "127.0.0.1", "--port", "8790", "--tensor-parallel-size", "1"]
    )
    assert known.port == 8790            # API mode still sees it
    assert "--port" in rest and rest[rest.index("--port") + 1] == "8790"
    assert "--host" in rest and rest[rest.index("--host") + 1] == "127.0.0.1"
    assert "MODEL" in rest and "--tensor-parallel-size" in rest


def test_inline_equals_form_is_forwarded():
    _, rest = consume_api_serve_args(["--port=8790", "--model", "m"])
    assert rest[rest.index("--port") + 1] == "8790"
    assert rest[rest.index("--model") + 1] == "m"


def test_absent_flags_are_not_invented():
    known, rest = consume_api_serve_args(["MODEL", "--tensor-parallel-size", "1"])
    assert known.port == 8000            # namespace default for API mode
    assert "--port" not in rest          # but nothing injected into passthrough
    assert "--host" not in rest


@pytest.mark.parametrize("flag", ["--host", "--port", "--model"])
def test_no_duplicate_when_already_in_passthrough(flag):
    val = "8790" if flag == "--port" else "x"
    _, rest = consume_api_serve_args([flag, val])
    assert rest.count(flag) == 1
