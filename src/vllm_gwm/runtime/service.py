# SPDX-License-Identifier: MIT
"""Advisor service — GWM harness behind a clean Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm_gwm.collect.flow import shape_flow
from vllm_gwm.config import GwmConfig
from vllm_gwm.presets import Preset, load_preset
from vllm_gwm.runtime.registry import GraphError, GraphRegistry


def _error_result(exc: Exception) -> dict[str, Any]:
    return {
        "state": "UNKNOWN",
        "cos_dist": 1.0,
        "abstain": True,
        "is_trap": False,
        "last_tool": "",
        "injected": False,
        "block": "",
        "success_score": 0.0,
        "probability": 1.0,
        "error": str(exc),
    }


class AdvisorService:
    def __init__(self, cfg: GwmConfig, registry: GraphRegistry, llm: Any):
        self.cfg = cfg
        self.registry = registry
        self.llm = llm
        self._presets: dict[str, Preset] = {}
        self._accretors: dict[str, Any] = {}

    def _preset(self, name: str | None) -> Preset:
        key = (name or self.cfg.preset_default).strip()
        if key not in self._presets:
            self._presets[key] = load_preset(key, self.cfg.preset_dir)
        return self._presets[key]

    def _log_path(self, preset_name: str) -> str | None:
        if not self.cfg.audit_log:
            return None
        d = self.cfg.data_root / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"gwm_{preset_name}.jsonl")

    def _domain(self, scorer: Any, preset: Preset, domain: str | None) -> str:
        if domain:
            return domain
        if preset.default_domain:
            return preset.default_domain
        doms = sorted(getattr(scorer, "trans", {}) or {})
        return doms[0] if doms else "unknown"

    def _mediator_for(self, preset: Preset, graph: str | None):
        path = self.registry.resolve(graph, preset)
        cfg = preset.harness_config(
            log_path=self._log_path(preset.name),
            overrides=self.cfg.harness_overrides,
        )
        return self.registry.get_mediator(path, preset, self.llm, cfg), path

    def _shape(self, flow: list[dict[str, Any]], preset: Preset) -> list[dict[str, Any]]:
        return shape_flow(flow, preset.flow_style)

    def advise(
        self,
        conversation_flow: list[dict[str, Any]],
        *,
        preset: str | None = None,
        domain: str | None = None,
        graph: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            p = self._preset(preset)
            flow = self._shape(list(conversation_flow or []), p)
            if episode_id:
                flow = [{"type": "task_metadata", "episode": str(episode_id)}] + flow
            med, _ = self._mediator_for(p, graph)
            dom = self._domain(med.scorer, p, domain)
            return med.advise(flow, dom)
        except Exception as exc:  # noqa: BLE001
            return _error_result(exc)

    def select(
        self,
        conversation_flows: list[list[dict[str, Any]]],
        *,
        mode: str = "auto",
        preset: str | None = None,
        domain: str | None = None,
        graph: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            p = self._preset(preset)
            flows = [self._shape(list(f), p) for f in (conversation_flows or [])]
            if not flows:
                raise GraphError("select requires at least one conversation flow")
            med, _ = self._mediator_for(p, graph)
            dom = self._domain(med.scorer, p, domain)
            m = (mode or "auto").strip()
            if m == "joint" or (m == "auto" and len(flows) > 1):
                results = med.select_joint(flows, dom, episode_id=episode_id)
            elif m == "single" or (m == "auto" and len(flows) == 1):
                results = [med.select_single(f, dom) for f in flows]
            else:
                raise GraphError(f"unknown select mode {mode!r}")
            return {"results": results}
        except Exception as exc:  # noqa: BLE001
            return {"results": [_error_result(exc)]}

    def _accretor_for(self, path: Any):
        """One GraphAccretor per resolved graph path (mutates that Scorer)."""
        from vllm_gwm.build.accrete import GraphAccretor

        key = str(path)
        acc = self._accretors.get(key)
        if acc is None:
            snap = None
            if self.cfg.audit_log:
                snap = self.cfg.data_root / "accrete" / f"{Path(path).name}.json"
            acc = GraphAccretor(
                accrete_self=self.cfg.accrete_self,
                mint_states=self.cfg.mint_states,
                max_per_state=self.cfg.accrete_max_per_state,
                mint_min_support=self.cfg.mint_min_support,
                mint_radius=self.cfg.mint_radius,
                scope=self.cfg.accrete_scope,
                snapshot_path=snap,
            )
            self._accretors[key] = acc
        return acc

    def accrete(
        self,
        pre_flow: list[dict[str, Any]],
        selected_text: str,
        *,
        preset: str | None = None,
        domain: str | None = None,
        graph: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """H27s/H28: fold this step's selected action into the live graph.

        Runs after the winner is chosen, so it never influences the step that
        produced it — only later steps at the same (or a newly minted) state.
        """
        if not (self.cfg.accrete_self or self.cfg.mint_states):
            return {}
        try:
            p = self._preset(preset)
            med, path = self._mediator_for(p, graph)
            acc = self._accretor_for(path)
            dom = self._domain(med.scorer, p, domain)
            return acc.step(
                med.scorer,
                dom,
                self._shape(list(pre_flow or []), p),
                selected_text or "",
                episode_id=episode_id or "",
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def classify(
        self,
        conversation_flow: list[dict[str, Any]],
        *,
        preset: str | None = None,
        domain: str | None = None,
        graph: str | None = None,
    ) -> dict[str, Any]:
        """H24: classify a pre-action flow without judging (state / abstain /
        trap / cos_dist). Used by serving to pick a per-step k."""
        try:
            p = self._preset(preset)
            flow = self._shape(list(conversation_flow or []), p)
            med, _ = self._mediator_for(p, graph)
            dom = self._domain(med.scorer, p, domain)
            state, dist, _, _ = med._classify(flow, dom)
            return {"state": state, "abstain": state is None,
                    "cos_dist": float(dist),
                    "is_trap": state in med.scorer.trap}
        except Exception as exc:  # noqa: BLE001
            return {"state": None, "abstain": True, "cos_dist": 1.0,
                    "is_trap": False, "error": str(exc)}

    def best_graph(
        self,
        pre_flow: list[dict[str, Any]],
        *,
        preset: str | None = None,
        domain: str | None = None,
        primary: str | None = None,
    ) -> str | None:
        """H19: among the primary adapter and ``cfg.extra_graphs``, return the
        adapter name that classifies ``pre_flow`` best (a known state wins over
        abstain; ties broken by lower cos_dist). Falls back to ``primary``."""
        names = [primary] + [g.strip() for g in
                             (self.cfg.extra_graphs or "").split(",") if g.strip()]
        best, best_dist, best_known = primary, 2.0, False
        for g in names:
            try:
                p = self._preset(preset)
                flow = self._shape(list(pre_flow or []), p)
                med, _ = self._mediator_for(p, g)
                dom = self._domain(med.scorer, p, domain)
                st, dist, _, _ = med._classify(flow, dom)
                known = st is not None
                if (known and not best_known) or (
                        known == best_known and dist < best_dist):
                    best, best_dist, best_known = g, float(dist), known
            except Exception:  # noqa: BLE001
                continue
        return best

    def approve_candidate(
        self,
        conversation_flow: list[dict[str, Any]],
        *,
        preset: str | None = None,
        domain: str | None = None,
        graph: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Return approval metadata for a completed candidate output."""
        result = self.advise(
            conversation_flow,
            preset=preset,
            domain=domain,
            graph=graph,
            episode_id=episode_id,
        )
        injected = bool(result.get("injected"))
        approved = not injected and not result.get("error")
        if injected:
            approved = False
        return {
            "approved": approved,
            "advice": result.get("block") or "",
            "state": result.get("state"),
            "success_score": result.get("success_score", 0.0),
            "harness": result.get("harness") or {},
            "result": result,
        }

    def info(self) -> dict[str, Any]:
        from vllm_gwm import __version__
        from vllm_gwm.presets import list_presets

        return {
            "server": "vllm_gwm",
            "version": __version__,
            "default_preset": self.cfg.preset_default,
            "presets": list_presets(self.cfg.preset_dir),
            "graphs": self.registry.list_graphs(),
            "collect_mode": self.cfg.collect_mode,
            "evolve_every_n": self.cfg.evolve_every_n,
            "accrete": {
                "accrete_self": self.cfg.accrete_self,
                "mint_states": self.cfg.mint_states,
                "scope": self.cfg.accrete_scope,
                "stats": {k: v.stats.as_dict() for k, v in self._accretors.items()},
            },
            "show_graph": self.cfg.show_graph,
            "graph_endpoint": "/v1/graph" if self.cfg.show_graph else None,
        }

    def list_graphs(self) -> dict[str, Any]:
        return {"graphs": self.registry.list_graphs(), "loaded": self.registry.loaded_keys()}
