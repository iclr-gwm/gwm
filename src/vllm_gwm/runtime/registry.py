# SPDX-License-Identifier: MIT
"""Graph registry: resolve, validate, cache transition graphs."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from vllm_gwm.harness import graph_sidecar, render, wm_prompt
from vllm_gwm.harness.harness import HarnessConfig, Mediator
from vllm_gwm.presets import Preset
from vllm_gwm.runtime.adapter import AdapterError, GraphAdapter


class GraphError(ValueError):
    """Raised when a graph reference cannot be resolved."""


class GraphRegistry:
    def __init__(
        self,
        registered: dict[str, Path],
        *,
        max_loaded: int = 2,
        load_examples: bool = True,
    ):
        self.registered = {k: Path(v).resolve() for k, v in registered.items()}
        self.roots = list({p.parent for p in self.registered.values()})
        self.max_loaded = max(1, int(max_loaded))
        self.load_examples = bool(load_examples)
        self._pinned: set[str] = set()
        self._lock = threading.RLock()
        self._scorers: OrderedDict[str, Any] = OrderedDict()
        self._mediators: OrderedDict[str, Mediator] = OrderedDict()
        self._current: dict[str, str] = {}

    def register(self, name: str, path: Path, *, pin: bool = False) -> None:
        adapter = GraphAdapter.validate(path, name=name)
        with self._lock:
            self.registered[name] = adapter.path
            if pin:
                self._pinned.add(str(adapter.path))

    def resolve(self, ref: str | None, preset: Preset, default: str = "") -> Path:
        candidate = (ref or "").strip() or default.strip() or preset.default_graph.strip()
        if not candidate:
            raise GraphError(f"no graph specified for preset {preset.name!r}")
        if candidate in self.registered:
            return self.registered[candidate]
        return GraphAdapter.resolve_path(candidate, list(self.registered.values()))

    def activate_version(self, name: str, version_path: Path) -> None:
        GraphAdapter.validate(version_path, name=name)
        new_path = version_path.resolve()
        prev = self.registered.get(name)
        pointer = (prev.parent if prev else new_path.parent) / "current.json"
        tmp = pointer.with_suffix(".tmp")
        tmp.write_text(json.dumps({"path": str(new_path)}), encoding="utf-8")
        tmp.replace(pointer)
        with self._lock:
            self._current[name] = str(new_path)
            # Repoint the registration, or resolve() keeps handing out the old
            # bundle and the activation has no effect on the serve path.
            self.registered[name] = new_path
            if new_path.parent not in self.roots:
                self.roots.append(new_path.parent)
            # Evict the caches for the path being REPLACED, not the new one.
            if prev is not None:
                self._scorers.pop(str(prev), None)
                for key in list(self._mediators):
                    if key.split("|", 1)[0] == str(prev):
                        del self._mediators[key]

    def get_scorer(self, path: Path, preset: Preset) -> Any:
        key = str(Path(path).resolve())
        with self._lock:
            sc = self._scorers.get(key)
            if sc is not None:
                self._scorers.move_to_end(key)
                return sc
        with self._lock:
            sc = self._scorers.get(key)
            if sc is None:
                GraphAdapter.validate(path)
                ex = (
                    str(path / "reports/examples.json")
                    if self.load_examples and (path / "reports/examples.json").exists()
                    else None
                )
                sc = graph_sidecar.Scorer(
                    str(path / "reports/transitions.json"),
                    str(path / "out/centroids"),
                    examples_path=ex,
                    unify_domains=preset.unify_domains,
                    p90_scale=preset.p90_scale,
                    log_requests=False,
                )
                self._scorers[key] = sc
                self._evict(self._scorers, key)
            else:
                self._scorers.move_to_end(key)
            return sc

    def get_mediator(
        self, path: Path, preset: Preset, llm: Any, cfg: HarnessConfig
    ) -> Mediator:
        key = f"{Path(path).resolve()}|{preset.name}"
        with self._lock:
            med = self._mediators.get(key)
            if med is not None:
                self._mediators.move_to_end(key)
                return med
        scorer = self.get_scorer(path, preset)
        with self._lock:
            med = self._mediators.get(key)
            if med is None:
                med = Mediator(scorer, llm, cfg, render, wm_prompt)
                self._mediators[key] = med
                self._evict(self._mediators, key)
            return med

    def _evict(self, cache: OrderedDict[str, Any], keep: str) -> None:
        while len(cache) > self.max_loaded:
            evicted = False
            for k in list(cache.keys()):
                if k == keep or k.split("|", 1)[0] in self._pinned:
                    continue
                del cache[k]
                evicted = True
                break
            if not evicted:
                break

    def stats(self) -> dict[str, Any]:
        """Cumulative harness usage per loaded mediator (graph-path | preset)."""
        with self._lock:
            meds = dict(self._mediators)
        return {key: med.stats_snapshot() for key, med in meds.items()}

    def list_graphs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        loaded = {k.split("|", 1)[0] for k in self._mediators}
        for name, path in self.registered.items():
            entry: dict[str, Any] = {
                "name": name,
                "path": str(path),
                "loaded": str(path) in loaded,
            }
            try:
                info = GraphAdapter.validate(path, name=name)
                entry["valid"] = True
                entry["manifest"] = info.manifest
            except AdapterError as exc:
                entry["valid"] = False
                entry["error"] = str(exc)
            out.append(entry)
        return out

    def loaded_keys(self) -> list[str]:
        with self._lock:
            return list(self._scorers.keys())
