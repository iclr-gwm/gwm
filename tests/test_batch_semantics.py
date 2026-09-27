# SPDX-License-Identifier: MIT
from vllm_gwm.protocol import GwmRequestOptions


def test_batch_candidate_count():
    entries = [{"vllm_xargs": {"gwm": {"adapter": "e", "k": 10}}} for _ in range(10)]
    total = sum(
        max(1, GwmRequestOptions.from_xargs(e.get("vllm_xargs")).k)
        for e in entries
        if GwmRequestOptions.from_xargs(e.get("vllm_xargs")).enabled
    )
    assert total == 100
