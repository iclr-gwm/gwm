# SPDX-License-Identifier: Apache-2.0
from vllm_gwm.collect.flow import harvest_toucan_flow, messages_to_flow, shape_flow
from vllm_gwm.presets import load_preset, list_presets


def test_toucan_preset_loads():
    assert "toucan" in list_presets()
    preset = load_preset("toucan")
    assert preset.unify_domains is True
    assert preset.flow_style == "toucan"
    cfg = preset.harness_config()
    assert cfg.select_sem_dedup is True
    assert "JSON tool call" in cfg.action_space


def test_harvest_toucan_flow_strips_catalog_and_parses_json_tools():
    prompt = (
        "You are an MCP tool-using agent.\n\n"
        "Task / user question:\nWhat is 2+2?\n\n"
        "Available tools:\n[]\n\n"
        "First assistant action (JSON only):"
    )
    flow = messages_to_flow(
        [
            {"role": "system", "content": "You are a tool-using AI agent."},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": '{"name": "calculator-add", "arguments": {"a": 2, "b": 2}}',
            },
        ]
    )
    shaped = harvest_toucan_flow(flow)
    assert shaped[0]["type"] == "user_message"
    assert shaped[0]["content"] == "What is 2+2?"
    assert shaped[1]["type"] == "ai_message"
    assert shaped[1]["tool_calls"] == [
        {"name": "calculator-add", "args": {"a": 2, "b": 2}}
    ]
    again = shape_flow(shaped, "toucan")
    assert again[0]["content"] == "What is 2+2?"
    assert again[1]["tool_calls"][0]["name"] == "calculator-add"


def test_shape_flow_passthrough():
    flow = [{"type": "user_message", "content": "hi"}]
    assert shape_flow(flow, "") == flow
    assert shape_flow(flow, "eops") == flow
