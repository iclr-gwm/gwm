# SPDX-License-Identifier: Apache-2.0
"""Build remote API clients for GWM."""

from __future__ import annotations

import os
from typing import Any

from vllm_gwm.apis.openai import OpenAIAPI, throttle_config_from_env


def list_api_presets() -> list[dict[str, Any]]:
    try:
        from vllm_gwm.internal_api import list_presets

        return list_presets()
    except ImportError:
        return []


def build_api(
    *,
    api_class: str = "openai",
    preset: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    auth: str | None = None,
    supports_n: bool | None = None,
    supports_temperature: bool | None = None,
    use_max_completion_tokens: bool | None = None,
    extra_body: dict[str, Any] | None = None,
    gemini_region: str | None = None,
    thinking_budget: int | None = None,
    api_provider: str | None = None,
    **kwargs: Any,
) -> OpenAIAPI:
    """Construct an :class:`OpenAIAPI` or a lazy-imported internal subclass."""
    cls = (api_class or os.getenv("VLLM_GWM_API_CLASS", "openai")).strip().lower()
    preset_name = (preset or os.getenv("VLLM_GWM_API_PRESET", "")).strip() or None
    if preset_name:
        cls = "internal"

    fields: dict[str, Any] = {}
    if preset_name:
        from vllm_gwm.internal_api import resolve_preset

        fields.update(resolve_preset(preset_name))

    if api_provider:
        fields["provider"] = api_provider
    if api_base or os.getenv("VLLM_GWM_API_BASE") or os.getenv("OPENAI_BASE_URL"):
        fields["endpoint"] = (
            api_base
            or os.getenv("VLLM_GWM_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
        )
    if model or os.getenv("VLLM_GWM_POLICY_MODEL") or os.getenv("OPENAI_MODEL"):
        fields["model"] = model or os.getenv("VLLM_GWM_POLICY_MODEL") or os.getenv(
            "OPENAI_MODEL"
        )
    if api_key:
        fields["api_key"] = api_key
    if supports_n is not None:
        fields["supports_n"] = supports_n
    if supports_temperature is not None:
        fields["supports_temperature"] = supports_temperature
    if use_max_completion_tokens is not None:
        fields["use_max_completion_tokens"] = use_max_completion_tokens
    if gemini_region:
        fields["gemini_region"] = gemini_region
    if thinking_budget is not None:
        fields["thinking_budget"] = thinking_budget
    fields.update({k: v for k, v in kwargs.items() if v is not None})

    if cls == "internal":
        from vllm_gwm.internal_api import build_internal

        if "model" not in fields:
            raise ValueError("internal API backend requires --model or --api-preset")
        return build_internal(**fields)

    base = fields.pop("endpoint", None) or api_base or os.getenv(
        "VLLM_GWM_API_BASE", os.getenv("OPENAI_BASE_URL", "")
    )
    if not base:
        raise ValueError("OpenAI API mode requires --api-base or OPENAI_BASE_URL")
    if "model" not in fields:
        raise ValueError("OpenAI API mode requires --model or OPENAI_MODEL")
    return OpenAIAPI(
        api_base=base,
        model=str(fields.pop("model")),
        api_key=api_key or fields.pop("api_key", None),
        auth=auth or os.getenv("VLLM_GWM_API_AUTH", "bearer"),
        supports_n=fields.pop("supports_n", True),
        supports_temperature=fields.pop("supports_temperature", True),
        use_max_completion_tokens=fields.pop("use_max_completion_tokens", False),
        extra_body=extra_body,
        throttle=throttle_config_from_env(),
        **{k: v for k, v in fields.items() if k not in ("provider", "region_capable")},
    )
