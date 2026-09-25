# SPDX-License-Identifier: Apache-2.0
"""Offline GWMLLM wrapper over vLLM LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_gwm.collect.flow import messages_to_flow
from vllm_gwm.config import GwmConfig
from vllm_gwm.protocol import GwmRequestOptions
from vllm_gwm.runtime.engine_chat import EngineChatClient, bypass_context
from vllm_gwm.runtime.registry import GraphRegistry
from vllm_gwm.runtime.service import AdvisorService


@dataclass
class GWMRequest:
    messages: list[dict[str, Any]]
    adapter: str = ""
    mode: str = "auto"
    k: int = 1
    preset: str = ""
    episode_id: str = ""


class OfflineEngineBackend:
    def __init__(self, llm: Any, sampling_params: Any):
        self.llm = llm
        self.sampling_params = sampling_params

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        from vllm.sampling_params import SamplingParams

        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = self.llm.chat(messages=messages, sampling_params=sp)
        text = outputs[0].outputs[0].text if outputs else ""
        return text, {"prompt_tokens": 0, "completion_tokens": len(text.split())}


class GWMLLM:
    def __init__(
        self,
        llm: Any,
        registry: GraphRegistry,
        cfg: GwmConfig | None = None,
    ):
        self.llm = llm
        self.cfg = cfg or GwmConfig.from_env()
        backend = OfflineEngineBackend(llm, None)
        chat = EngineChatClient(backend, model=getattr(llm, "model", "model"))
        self.advisor = AdvisorService(self.cfg, registry, chat)

    def chat(self, request: GWMRequest) -> dict[str, Any]:
        opts = GwmRequestOptions(
            enabled=bool(request.adapter),
            adapter=request.adapter,
            mode=request.mode,
            k=request.k,
            preset=request.preset,
            episode_id=request.episode_id,
        )
        flow = messages_to_flow(request.messages)
        if opts.is_select or opts.k > 1:
            flows = [flow for _ in range(opts.k)]
            with bypass_context(enabled=True):
                for i in range(opts.k):
                    out = self.llm.chat(
                        messages=request.messages,
                        sampling_params=self.llm.get_default_sampling_params(),
                    )
                    text = out[0].outputs[0].text if out else ""
                    flows[i] = flow + messages_to_flow(
                        [{"role": "assistant", "content": text}]
                    )
            sel = self.advisor.select(
                flows, preset=opts.preset or None, graph=opts.adapter or None
            )
            return sel
        approval = self.advisor.approve_candidate(
            flow,
            preset=opts.preset or None,
            graph=opts.adapter or None,
            episode_id=opts.episode_id or None,
        )
        with bypass_context(enabled=True):
            out = self.llm.chat(
                messages=request.messages,
                sampling_params=self.llm.get_default_sampling_params(),
            )
        text = out[0].outputs[0].text if out else ""
        return {"text": text, "approval": approval}
