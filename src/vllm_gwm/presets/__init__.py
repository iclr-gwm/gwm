# SPDX-License-Identifier: Apache-2.0
"""Benchmark presets for the GWM advisor."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm_gwm.harness.harness import HarnessConfig

_HERE = Path(__file__).resolve().parent
_HARNESS_FIELDS = {f.name for f in dataclasses.fields(HarnessConfig)}


class PresetError(ValueError):
    """Raised for a malformed or unknown preset."""


@dataclass
class Preset:
    name: str
    description: str = ""
    default_graph: str = ""
    default_domain: str = ""
    unify_domains: bool = False
    p90_scale: float = 1.0
    scrub_config: str = ""
    flow_style: str = ""
    greedy_anchor: bool = True    # H1: per-preset select greedy anchor (candidate
    #                               0 at temp=0). Default-on = validated winner;
    #                               authoritative unless VLLM_GWM_GREEDY_ANCHOR is
    #                               explicitly set (crm/eops pin it off).
    harness: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def harness_config(
        self,
        *,
        log_path: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> HarnessConfig:
        merged: dict[str, Any] = dict(self.harness)
        if overrides:
            merged.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(merged) - _HARNESS_FIELDS
        if unknown:
            raise PresetError(
                f"preset {self.name!r}: unknown harness keys {sorted(unknown)}"
            )
        cfg = HarnessConfig(**merged)
        if log_path is not None:
            cfg.log_path = log_path
        return cfg


def _load_toml(path: Path) -> Preset:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    p = data.get("preset") or {}
    return Preset(
        name=str(p.get("name") or path.stem),
        description=str(p.get("description", "")),
        default_graph=str(p.get("default_graph", "")),
        default_domain=str(p.get("default_domain", "")),
        unify_domains=bool(p.get("unify_domains", False)),
        # H20: VLLM_GWM_P90_SCALE overrides the preset's abstain threshold scale
        # for sweeping the classifier's UNKNOWN band (validate offline first).
        p90_scale=float(os.getenv("VLLM_GWM_P90_SCALE", "").strip()
                        or p.get("p90_scale", 1.0)),
        scrub_config=str(p.get("scrub_config", "")),
        flow_style=str(p.get("flow_style", "")),
        greedy_anchor=bool(p.get("greedy_anchor", False)),
        harness=dict(data.get("harness") or {}),
        source_path=path,
    )


def _preset_dirs(extra: Path | None = None) -> list[Path]:
    dirs = [_HERE]
    env = os.getenv("VLLM_GWM_PRESET_DIR", "").strip()
    if env:
        dirs.extend(Path(d) for d in env.split(os.pathsep) if d)
    if extra is not None:
        dirs.append(extra)
    return dirs


def list_presets(extra: Path | None = None) -> list[str]:
    names: list[str] = []
    for d in _preset_dirs(extra):
        if d.is_dir():
            for f in sorted(d.glob("*.toml")):
                if f.stem not in names:
                    names.append(f.stem)
    return names


def load_preset(name: str, extra: Path | None = None) -> Preset:
    found: Path | None = None
    for d in _preset_dirs(extra):
        cand = d / f"{name}.toml"
        if cand.is_file():
            found = cand
    if found is None:
        raise PresetError(f"unknown preset {name!r}; available: {list_presets(extra)}")
    return _load_toml(found)
