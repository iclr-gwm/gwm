# SPDX-License-Identifier: Apache-2.0
"""Request protocol helpers for GWM xargs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GwmRequestOptions:
    enabled: bool = False
    adapter: str = ""
    graph: str = ""
    mode: str = "auto"  # advise | select | auto
    k: int = 1
    preset: str = ""
    domain: str = ""
    episode_id: str = ""
    finalize: bool = False
    collect: bool = False
    outcome_success: bool | None = None
    outcome_source: str = ""

    @classmethod
    def from_xargs(cls, xargs: dict[str, Any] | None) -> GwmRequestOptions:
        if not xargs:
            return cls()
        gwm = xargs.get("gwm")
        if not isinstance(gwm, dict):
            gwm = {k.removeprefix("gwm."): v for k, v in xargs.items() if str(k).startswith("gwm.")}
        if not gwm:
            return cls()
        enabled = bool(gwm.get("adapter") or gwm.get("enabled"))
        mode = str(gwm.get("mode") or "auto")
        k = int(gwm.get("k") or 1)
        return cls(
            enabled=enabled,
            adapter=str(gwm.get("adapter") or ""),
            graph=str(gwm.get("graph") or ""),
            mode=mode,
            k=max(1, k),
            preset=str(gwm.get("preset") or ""),
            domain=str(gwm.get("domain") or ""),
            episode_id=str(gwm.get("episode_id") or ""),
            finalize=bool(gwm.get("finalize")),
            collect=bool(gwm.get("collect")),
            outcome_success=(
                bool(gwm["success"]) if "success" in gwm else None
            ),
            outcome_source=str(gwm.get("outcome_source") or "client"),
        )

    @property
    def is_select(self) -> bool:
        if self.mode == "select":
            return True
        if self.mode == "advise":
            return False
        return self.k > 1
