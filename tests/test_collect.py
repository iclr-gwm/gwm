# SPDX-License-Identifier: MIT
from vllm_gwm.collect.flow import messages_to_flow
from vllm_gwm.collect.outcome import OutcomeResolver
from vllm_gwm.collect.store import RolloutStore
from vllm_gwm.config import GwmConfig
from vllm_gwm.harness.llm import StubChat
from vllm_gwm.runtime.registry import GraphRegistry
from vllm_gwm.runtime.service import AdvisorService


def test_flow_normalization():
    flow = messages_to_flow(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_calls": []},
        ]
    )
    assert flow[0]["type"] == "system_message"
    assert flow[1]["type"] == "user_message"
    assert flow[2]["type"] == "ai_message"


def test_store_and_outcome(tmp_path):
    store = RolloutStore(tmp_path / "rollouts")
    flow = messages_to_flow([{"role": "user", "content": "task"}])
    store.append(flow, episode_id="ep1", success=True, finalize=True)
    cfg = GwmConfig(judge_enabled=False)
    advisor = AdvisorService(cfg, GraphRegistry({}), StubChat())
    outcome = OutcomeResolver(cfg, advisor)
    resolved = outcome.resolve(flow, explicit_success=True, explicit_source="explicit")
    assert resolved["approved_for_mining"] is False  # missing ai turn
    flow2 = flow + [{"type": "ai_message", "content": "done", "tool_calls": []}]
    resolved2 = outcome.resolve(flow2, explicit_success=True)
    assert resolved2["source"] == "explicit"
    assert resolved2["approved_for_mining"] is True


def test_store_hydrates_finalized_from_disk(tmp_path):
    root = tmp_path / "rollouts"
    first = RolloutStore(root)
    first.append(
        [{"type": "user_message", "content": "a"}],
        episode_id="ep-a",
        success=True,
        finalize=True,
    )
    first.append(
        [{"type": "user_message", "content": "b"}],
        episode_id="ep-b",
        success=False,
        finalize=True,
    )
    first.append(
        [{"type": "user_message", "content": "c"}],
        episode_id="ep-c",
        success=None,
        finalize=False,
    )
    revived = RolloutStore(root)
    assert revived.count_finalized() == 2
