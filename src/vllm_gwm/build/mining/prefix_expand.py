# SPDX-License-Identifier: Apache-2.0
"""Stage 1 — per-step prefix expansion (``rollouts.jsonl.gz`` -> ``steps.jsonl.gz``).

A trajectory of **N steps** (one per ``ai_message`` / agent action) yields **N
prefix samples**. Sample *i* is the cumulative flow up to and including step *i*
(the agent's i-th action plus the tool results it produced), rendered to text and
ready for embedding. The growing prefixes are what make state discovery and the
state->state transition model meaningful.

Each step record carries:

- ``step_idx`` (1..N), ``n_steps``
- ``tool_names`` called at this step, ``obs_outcome`` (ok/error/none) from the
  tool results that resolved it
- the rollout's ``split`` and final ``overall_success`` / ``pass_rate`` (the
  outcome label is attached to *every* prefix of the rollout)
- ``text`` — the rendered prefix (sentence-transformer input)

Ported from ``benchmarks/wm/lib/prefix_expand.py``; the rendering comes from the
plugin's own :mod:`vllm_gwm.harness.render` (same ``render_text`` /
``tool_result_outcome`` API), so there is only one renderer in the tree.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vllm_gwm.harness import render

MAX_TOOL_RESULT_CHARS = 2000
MAX_FLOW_CHARS = 60000


def read_jsonl(path: str | Path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def step_records(
    rollout: dict[str, Any],
    *,
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
    max_flow_chars: int = MAX_FLOW_CHARS,
) -> list[dict]:
    """Expand one rollout into N prefix samples (N = number of ai_message steps)."""
    uid = rollout.get("uid")
    if not uid:
        raise ValueError("rollout is missing 'uid' (mining keys every step by rollout uid)")
    flow = rollout.get("conversation_flow") or []
    ai_pos = [
        p for p, e in enumerate(flow)
        if isinstance(e, dict) and e.get("type") == "ai_message"
    ]
    n_steps = len(ai_pos)
    out: list[dict] = []
    for i, p in enumerate(ai_pos):
        next_p = ai_pos[i + 1] if i + 1 < len(ai_pos) else len(flow)
        prefix = flow[:next_p]  # include this action + its trailing tool results
        ai = flow[p]
        tool_names = [
            (tc.get("name") or (tc.get("function") or {}).get("name") or "tool")
            for tc in (ai.get("tool_calls") or [])
            if isinstance(tc, dict)
        ]
        outcomes = [
            render.tool_result_outcome(e)
            for e in flow[p + 1: next_p]
            if isinstance(e, dict) and e.get("type") == "tool_result"
        ]
        if not outcomes:
            obs_outcome = "none"  # final answer / no tool call resolved
        elif "error" in outcomes:
            obs_outcome = "error"
        else:
            obs_outcome = "ok"
        out.append({
            "step_uid": f"{uid}__s{i + 1}",
            "uid": uid,
            "task_id": rollout.get("task_id", uid),
            "domain": rollout.get("domain"),
            "split": rollout.get("split"),
            "temp": rollout.get("temp"),
            "step_idx": i + 1,
            "n_steps": n_steps,
            "tool_names": tool_names,
            "obs_outcome": obs_outcome,
            "overall_success": rollout.get("overall_success"),
            "pass_rate": rollout.get("pass_rate"),
            "text": render.render_text(prefix, max_tool_result_chars, max_flow_chars),
        })
    return out


def expand_steps(
    rollouts_path: str | Path,
    steps_path: str | Path,
    *,
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
    max_flow_chars: int = MAX_FLOW_CHARS,
) -> dict[str, Any]:
    """Expand every rollout in ``rollouts_path`` into ``steps_path``."""
    out = Path(steps_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out.suffix == ".gz" else open

    n_roll = n_steps = 0
    by_split: Counter = Counter()
    by_outcome: Counter = Counter()
    with opener(out, "wt", encoding="utf-8") as f:
        for rollout in read_jsonl(rollouts_path):
            n_roll += 1
            for rec in step_records(
                rollout,
                max_tool_result_chars=max_tool_result_chars,
                max_flow_chars=max_flow_chars,
            ):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_steps += 1
                by_split[rec["split"]] += 1
                by_outcome[rec["obs_outcome"]] += 1

    print(f"[prefix_expand] rollouts={n_roll} steps={n_steps} "
          f"(avg {n_steps / max(n_roll, 1):.1f} steps/rollout)")
    print(f"[prefix_expand] by_split={dict(by_split)} by_obs_outcome={dict(by_outcome)}")
    print(f"[prefix_expand] -> {out}")
    return {
        "rollouts": n_roll,
        "steps": n_steps,
        "by_split": dict(by_split),
        "by_obs_outcome": dict(by_outcome),
        "out": str(out),
    }


def split_by_tag(
    rollouts_all: str | Path,
    train_path: str | Path,
    test_path: str | Path,
) -> dict[str, int]:
    """Stage 0b — split a harvested ``rollouts_all`` by its ``split`` tag.

    Rows tagged ``split == "test"`` go to ``test_path``, everything else (train /
    valid / untagged) to ``train_path``.
    """
    n_train = n_test = 0
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(train_path, "wt", encoding="utf-8") as tr, \
            gzip.open(test_path, "wt", encoding="utf-8") as te:
        for rec in read_jsonl(rollouts_all):
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if (rec.get("split") or "train") == "test":
                te.write(line)
                n_test += 1
            else:
                tr.write(line)
                n_train += 1
    print(f"[split] non-test={n_train} test={n_test}")
    return {"train": n_train, "test": n_test}
