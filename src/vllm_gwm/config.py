# SPDX-License-Identifier: MIT
"""Configuration for the vLLM GWM extension."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return default if v == "" else int(v)


@dataclass
class GwmConfig:
    """Runtime configuration loaded from CLI flags and ``VLLM_GWM_*`` env vars."""

    data_root: Path = field(default_factory=lambda: Path(".gwm_data"))
    preset_default: str = "crm"
    preset_dir: Path | None = None
    max_graphs: int = 2
    load_examples: bool = True
    fail_open: bool = True
    max_attempts: int = 3
    max_concurrent_gwm: int = 4
    collect_mode: str = "opt_in"  # off | opt_in | all
    evolve_every_n: int = 0
    judge_enabled: bool = True
    judge_min_confidence: float = 0.6
    harness_overrides: dict[str, Any] = field(default_factory=dict)
    llm_disable_thinking: bool = True
    audit_log: bool = False
    show_graph: bool = False
    greedy_anchor: bool = True   # H1: draw candidate 0 at temperature=0 (greedy)
    #                              and 1..k-1 at the request temperature, so the
    #                              select arm degrades to base on identical/low-
    #                              margin steps (TOUCAN_PERF_HYPOTHESES H1)
    adaptive_k: bool = False     # H24: classify pre-flow; use k=1 (skip judge) on
    #                              confident non-trap states, requested k otherwise
    extra_graphs: str = ""       # H19: comma-separated extra adapter names to
    #                              classify against; the best-matching (lowest
    #                              cos_dist, non-abstain) graph judges the step
    accrete_self: bool = False   # H27s: prepend the judge-selected action as an
    #                              exemplar at the classified state (label-free,
    #                              self-referential; see G.3 on why no outcome
    #                              signal is used)
    mint_states: bool = False    # H28: mint new centroids for recurring
    #                              abstained states instead of leaving them UNKNOWN
    accrete_max_per_state: int = 8
    mint_min_support: int = 3    # a minted state contributes evidence only
    #                              after this many visits
    mint_radius: float = 0.0     # 0 = the graph's own median P90 radius
    accrete_scope: str = "global"  # global (transductive) | episode (inductive)
    # Self-evolution builds (vllm_gwm.build.pipeline). "" = the pipeline default
    # embedder; the device defaults to CPU because the GPU is busy serving.
    build_embed_model: str = ""
    build_embed_device: str = "cpu"
    build_watched_tools: str = ""

    @classmethod
    def from_env(cls) -> GwmConfig:
        root = os.getenv("VLLM_GWM_DATA_ROOT", ".gwm_data").strip() or ".gwm_data"
        preset_dir = os.getenv("VLLM_GWM_PRESET_DIR", "").strip()
        overrides: dict[str, Any] = {}
        for env, key, conv in (
            ("VLLM_GWM_TRIGGER", "trigger", str),
            ("VLLM_GWM_MAX_PER_EPISODE", "max_per_episode", int),
            ("VLLM_GWM_MAX_TOOL_CALLS", "max_tool_calls", int),
            ("VLLM_GWM_MAX_ADVICE_CHARS", "max_advice_chars", int),
            ("VLLM_GWM_TOPN_STATES", "topn_states", int),
            ("VLLM_GWM_EDGE_MIN_N", "edge_min_n", int),
            ("VLLM_GWM_SELECT_VOTES", "select_votes", int),
            ("VLLM_GWM_SELECT_SEM_DEDUP", "select_sem_dedup", _env_bool),
            ("VLLM_GWM_SELECT_MIN_MARGIN", "select_min_margin", float),
            ("VLLM_GWM_SELECT_EXAMPLES", "select_examples", int),
            ("VLLM_GWM_SELECT_NEG_EXAMPLES", "select_neg_examples", int),
            ("VLLM_GWM_SELECT_TAIL", "select_tail_chars", int),
            ("VLLM_GWM_SELECT_TOURNAMENT", "select_tournament", _env_bool),
            ("VLLM_GWM_SELECT_MIN_MARGIN_ABSTAIN", "select_min_margin_abstain", float),
            ("VLLM_GWM_SELECT_GRAPH_TIEBREAK", "select_graph_tiebreak", _env_bool),
            ("VLLM_GWM_SELECT_EXAMPLES_RANK", "select_examples_rank", _env_bool),
            ("VLLM_GWM_TOOL_DOCS_LIVE", "tool_docs_live", _env_bool),
            ("VLLM_GWM_TOOL_DOCS_CHARS", "tool_docs_chars", int),
            ("VLLM_GWM_LOG_CANDIDATES", "log_candidates", _env_bool),
            # F5 nearest-state fallback, exposed so it can be A/B'd at all.
            # Turning it off measured 59.2 vs a 58.7 control over 6 seeds on
            # toucan_100_test — i.e. no effect. Keep the default on.
            ("VLLM_GWM_NEAREST_ON_ABSTAIN", "nearest_on_abstain", _env_bool),
        ):
            v = os.getenv(env, "").strip()
            if v != "":
                overrides[key] = conv(v) if conv is not _env_bool else _env_bool(env, False)
        return cls(
            data_root=Path(root),
            preset_default=os.getenv("VLLM_GWM_PRESET_DEFAULT", "crm"),
            preset_dir=Path(preset_dir) if preset_dir else None,
            max_graphs=_env_int("VLLM_GWM_MAX_GRAPHS", 2),
            load_examples=_env_bool("VLLM_GWM_LOAD_EXAMPLES", True),
            fail_open=_env_bool("VLLM_GWM_FAIL_OPEN", True),
            max_attempts=_env_int("VLLM_GWM_MAX_ATTEMPTS", 3),
            max_concurrent_gwm=_env_int("VLLM_GWM_MAX_CONCURRENT", 4),
            collect_mode=os.getenv("VLLM_GWM_COLLECT", "opt_in"),
            evolve_every_n=_env_int("VLLM_GWM_EVOLVE_EVERY", 0),
            judge_enabled=_env_bool("VLLM_GWM_JUDGE", True),
            judge_min_confidence=float(os.getenv("VLLM_GWM_JUDGE_MIN_CONF", "0.6")),
            harness_overrides=overrides,
            llm_disable_thinking=_env_bool("VLLM_GWM_DISABLE_THINKING", True),
            audit_log=_env_bool("VLLM_GWM_AUDIT_LOG", False),
            show_graph=_env_bool("VLLM_GWM_SHOW_GRAPH", False),
            greedy_anchor=_env_bool("VLLM_GWM_GREEDY_ANCHOR", True),
            adaptive_k=_env_bool("VLLM_GWM_ADAPTIVE_K", False),
            extra_graphs=os.getenv("VLLM_GWM_EXTRA_GRAPHS", "").strip(),
            accrete_self=_env_bool("VLLM_GWM_ACCRETE_SELF", False),
            mint_states=_env_bool("VLLM_GWM_MINT_STATES", False),
            accrete_max_per_state=_env_int("VLLM_GWM_ACCRETE_MAX_PER_STATE", 8),
            mint_min_support=_env_int("VLLM_GWM_MINT_MIN_SUPPORT", 3),
            mint_radius=float(os.getenv("VLLM_GWM_MINT_RADIUS", "0") or 0),
            accrete_scope=os.getenv("VLLM_GWM_ACCRETE_SCOPE", "global").strip(),
            build_embed_model=os.getenv("VLLM_GWM_BUILD_EMBED_MODEL", "").strip(),
            build_embed_device=os.getenv("VLLM_GWM_BUILD_EMBED_DEVICE", "cpu").strip(),
            build_watched_tools=os.getenv("VLLM_GWM_BUILD_WATCHED_TOOLS", "").strip(),
        )
