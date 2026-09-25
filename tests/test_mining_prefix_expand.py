# SPDX-License-Identifier: Apache-2.0
"""Stage 1 (prefix_expand): one sample per agent action, with its observations."""

import gzip
import json

import pytest

from vllm_gwm.build.mining.prefix_expand import expand_steps, split_by_tag, step_records


def rollout(uid="r1", *, split="train", success=True, domain="toy"):
    return {
        "uid": uid,
        "task_id": f"task-{uid}",
        "domain": domain,
        "split": split,
        "overall_success": success,
        "pass_rate": 1.0 if success else 0.0,
        "conversation_flow": [
            {"type": "system_message", "content": "You are an agent."},
            {"type": "user_message", "content": "Find the owner of Acme."},
            {"type": "ai_message", "content": "", "tool_calls": [
                {"name": "execute", "args": {"q": "SELECT owner FROM Acct"}}]},
            {"type": "tool_result", "tool_name": "execute",
             "result": {"success": False, "error": "no such table: Acct"}},
            {"type": "ai_message", "content": "", "tool_calls": [
                {"name": "describe", "args": {"table": "Account"}},
                {"name": "list_fields", "args": {"table": "Account"}}]},
            {"type": "tool_result", "tool_name": "describe",
             "result": {"success": True, "result": {"columns": ["OwnerId"]}}},
            {"type": "tool_result", "tool_name": "list_fields",
             "result": {"success": True, "result": ["OwnerId"]}},
            {"type": "ai_message", "content": "The owner is user 007.",
             "tool_calls": []},
        ],
    }


def test_one_step_per_ai_message():
    steps = step_records(rollout())
    assert [s["step_idx"] for s in steps] == [1, 2, 3]
    assert {s["n_steps"] for s in steps} == {3}
    assert [s["step_uid"] for s in steps] == ["r1__s1", "r1__s2", "r1__s3"]


def test_tool_names_and_observation_outcome():
    steps = step_records(rollout())
    assert steps[0]["tool_names"] == ["execute"]
    assert steps[0]["obs_outcome"] == "error"        # its tool_result errored
    assert steps[1]["tool_names"] == ["describe", "list_fields"]
    assert steps[1]["obs_outcome"] == "ok"
    assert steps[2]["tool_names"] == []
    assert steps[2]["obs_outcome"] == "none"         # final answer, no tool call


def test_prefixes_grow_and_carry_the_rollout_outcome():
    steps = step_records(rollout(success=False))
    texts = [s["text"] for s in steps]
    assert len(texts[0]) < len(texts[1]) < len(texts[2])
    # step i's text ends at step i's observations: only the first action is in
    # the first prefix, all three are in the last.
    assert "describe" not in texts[0]
    assert "describe" in texts[1]
    assert all(s["overall_success"] is False for s in steps)  # label on every prefix


def test_function_style_tool_calls_and_missing_uid():
    r = rollout()
    r["conversation_flow"][2]["tool_calls"] = [{"function": {"name": "sql"}}]
    assert step_records(r)[0]["tool_names"] == ["sql"]
    del r["uid"]
    with pytest.raises(ValueError, match="uid"):
        step_records(r)


def test_expand_steps_writes_gzip_jsonl(tmp_path):
    src = tmp_path / "rollouts.jsonl.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps(rollout(f"r{i}", success=i % 2 == 0)) + "\n")
    out = tmp_path / "steps.jsonl.gz"
    info = expand_steps(src, out)
    assert info["rollouts"] == 3
    assert info["steps"] == 9
    assert info["by_split"] == {"train": 9}
    assert info["by_obs_outcome"] == {"error": 3, "ok": 3, "none": 3}
    with gzip.open(out, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 9
    assert set(rows[0]) >= {"step_uid", "uid", "task_id", "domain", "split", "step_idx",
                            "n_steps", "tool_names", "obs_outcome", "overall_success",
                            "pass_rate", "text"}


def test_split_by_tag_sends_test_rows_aside(tmp_path):
    src = tmp_path / "rollouts_all.jsonl.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write(json.dumps(rollout("a", split="train")) + "\n")
        f.write(json.dumps(rollout("b", split="test")) + "\n")
        f.write(json.dumps(rollout("c", split=None)) + "\n")   # untagged -> non-test
    counts = split_by_tag(src, tmp_path / "tr.jsonl.gz", tmp_path / "te.jsonl.gz")
    assert counts == {"train": 2, "test": 1}
