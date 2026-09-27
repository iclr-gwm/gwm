# SPDX-License-Identifier: MIT
"""Shared GWM app-state bootstrap for plugin and API server."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Any

from starlette.datastructures import State

from vllm_gwm.apis.factory import build_api
from vllm_gwm.build.evolve import EvolveManager
from vllm_gwm.collect.outcome import OutcomeResolver
from vllm_gwm.collect.store import RolloutStore
from vllm_gwm.config import GwmConfig
from vllm_gwm.runtime.registry import GraphRegistry
from vllm_gwm.runtime.service import AdvisorService
from vllm_gwm.viz.show_graph import GraphViewer


def parse_gwm_modules(args: Namespace | None = None) -> dict[str, Path]:
    modules: dict[str, Path] = {}
    items: list[str] = []
    if args is not None:
        items.extend(list(getattr(args, "gwm_modules", None) or []))
    env = os.getenv("VLLM_GWM_MODULES", "").strip()
    if env:
        items.extend(p for p in env.split(",") if p.strip())
    for item in items:
        if "=" not in item:
            continue
        name, path = item.split("=", 1)
        modules[name.strip()] = Path(path.strip()).resolve()
    return modules


def build_judge_client(
    *,
    default_model: str,
    api_preset: str | None = None,
    api_class: str | None = None,
) -> Any:
    judge_preset = (
        os.getenv("VLLM_GWM_JUDGE_API_PRESET", "").strip()
        or api_preset
        or ""
    ) or None
    judge_class = (
        os.getenv("VLLM_GWM_JUDGE_API_CLASS", "").strip()
        or api_class
        or os.getenv("VLLM_GWM_API_CLASS", "openai")
    ).strip()
    judge_model = os.getenv("VLLM_GWM_JUDGE_MODEL", "").strip() or default_model
    judge_base = os.getenv("VLLM_GWM_JUDGE_BASE_URL", "").strip() or None
    judge_key = os.getenv("VLLM_GWM_JUDGE_API_KEY", "").strip() or None
    if judge_preset or judge_class == "internal" or judge_base:
        return build_api(
            api_class=judge_class,
            preset=judge_preset,
            api_base=judge_base,
            model=judge_model,
            api_key=judge_key,
        )
    if judge_base:
        from vllm_gwm.harness.llm import ChatClient

        return ChatClient(judge_base.rstrip("/"), judge_model, temperature=0.0)
    return None


def init_gwm_core(
    state: State,
    *,
    cfg: GwmConfig | None = None,
    registered: dict[str, Path] | None = None,
    judge_llm: Any | None = None,
    default_model: str = "model",
) -> tuple[GraphRegistry, AdvisorService, RolloutStore, EvolveManager]:
    cfg = cfg or GwmConfig.from_env()
    registered = dict(registered or {})
    if not registered:
        env_root = cfg.data_root / "graphs"
        if env_root.is_dir():
            for p in env_root.iterdir():
                if p.is_dir():
                    registered[p.name] = p.resolve()
    registry = GraphRegistry(
        registered, max_loaded=cfg.max_graphs, load_examples=cfg.load_examples
    )
    for name, path in registered.items():
        registry.register(name, path, pin=True)
    llm = judge_llm
    if llm is None:
        built = build_judge_client(default_model=default_model)
        if built is not None:
            llm = built
        else:
            from vllm_gwm.harness.llm import StubChat

            llm = StubChat()
    advisor = AdvisorService(cfg, registry, llm)
    store = RolloutStore(cfg.data_root / "rollouts")
    outcome = OutcomeResolver(cfg, advisor)
    evolve = EvolveManager(cfg, store, registry, outcome)
    state.gwm_cfg = cfg
    state.gwm_registry = registry
    state.gwm_show_graph = cfg.show_graph
    state.gwm_graph_viewer = (
        GraphViewer.from_registered(registry.registered) if cfg.show_graph else None
    )
    state.gwm_advisor = advisor
    state.gwm_store = store
    state.gwm_outcome = outcome
    state.gwm_evolve = evolve
    return registry, advisor, store, evolve
