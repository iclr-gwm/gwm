# SPDX-License-Identifier: Apache-2.0
from vllm_gwm.protocol import GwmRequestOptions
from vllm_gwm.runtime.engine_chat import bypass_context, is_bypass_active, is_http_bypass


def test_xargs_parsing():
    opts = GwmRequestOptions.from_xargs(
        {"gwm": {"adapter": "eops", "k": 4, "mode": "select", "episode_id": "e1"}}
    )
    assert opts.enabled is True
    assert opts.adapter == "eops"
    assert opts.k == 4
    assert opts.is_select is True


def test_namespaced_xargs_parsing():
    opts = GwmRequestOptions.from_xargs(
        {
            "gwm.adapter": "toucan",
            "gwm.mode": "select",
            "gwm.k": "4",
            "gwm.preset": "toucan",
        }
    )
    assert opts.enabled is True
    assert opts.adapter == "toucan"
    assert opts.k == 4
    assert opts.preset == "toucan"
    assert opts.is_select is True


def test_bypass_context():
    assert is_bypass_active() is False
    with bypass_context(enabled=True, depth_delta=1):
        assert is_bypass_active() is True
    assert is_bypass_active() is False


def test_http_bypass_header():
    assert is_http_bypass(None) is False
    assert is_http_bypass({}) is False
    assert is_http_bypass({"x-vllm-gwm-bypass": "1"}) is True
    assert is_http_bypass({"X-VLLM-GWM-Bypass": "true"}) is True
