# SPDX-License-Identifier: MIT
"""On-disk layout for GWM data."""

from __future__ import annotations

from pathlib import Path


def data_root(base: Path | None = None) -> Path:
    root = base or Path(".gwm_data")
    root.mkdir(parents=True, exist_ok=True)
    return root


def graphs_dir(base: Path | None = None) -> Path:
    d = data_root(base) / "graphs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def builds_dir(base: Path | None = None) -> Path:
    d = data_root(base) / "builds"
    d.mkdir(parents=True, exist_ok=True)
    return d
