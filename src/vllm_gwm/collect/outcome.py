# SPDX-License-Identifier: Apache-2.0
"""Hybrid rollout outcome labeling."""

from __future__ import annotations

from typing import Any

from vllm_gwm.config import GwmConfig
from vllm_gwm.runtime.service import AdvisorService


class OutcomeResolver:
    def __init__(self, cfg: GwmConfig, advisor: AdvisorService):
        self.cfg = cfg
        self.advisor = advisor

    def resolve(
        self,
        flow: list[dict[str, Any]],
        *,
        explicit_success: bool | None = None,
        explicit_source: str = "",
    ) -> dict[str, Any]:
        if explicit_success is not None:
            return {
                "success": bool(explicit_success),
                "source": explicit_source or "explicit",
                "confidence": 1.0,
                "approved_for_mining": self._quality_ok(flow),
            }
        if not self.cfg.judge_enabled:
            return {
                "success": None,
                "source": "none",
                "confidence": 0.0,
                "approved_for_mining": False,
            }
        judged = self.advisor.select([flow], mode="single")
        result = (judged.get("results") or [{}])[0]
        score = float(result.get("success_score") or 0.0)
        confidence = score
        approved = confidence >= self.cfg.judge_min_confidence and self._quality_ok(flow)
        return {
            "success": score >= 0.5,
            "source": "judge",
            "confidence": confidence,
            "approved_for_mining": approved,
            "harness": result.get("harness") or {},
        }

    @staticmethod
    def _quality_ok(flow: list[dict[str, Any]]) -> bool:
        if not flow:
            return False
        has_user = any(e.get("type") == "user_message" for e in flow)
        has_ai = any(e.get("type") == "ai_message" for e in flow)
        return has_user and has_ai
