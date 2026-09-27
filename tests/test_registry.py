# SPDX-License-Identifier: MIT
import json

from vllm_gwm.config import GwmConfig
from vllm_gwm.harness.llm import StubChat
from vllm_gwm.runtime.registry import GraphRegistry
from vllm_gwm.runtime.service import AdvisorService

from conftest import FIX, TINY


def test_registry_resolve_registered(patch_scorer, tiny_preset):
    reg = GraphRegistry({"tiny": TINY}, max_loaded=2)
    p = tiny_preset
    path = reg.resolve("tiny", p)
    assert path == TINY.resolve()


def test_advise_triggers(patch_scorer, tiny_preset):
    cfg = GwmConfig(preset_default="tiny", audit_log=False)
    reg = GraphRegistry({"tiny": TINY})
    svc = AdvisorService(cfg, reg, StubChat())
    svc._presets["tiny"] = tiny_preset
    flow = json.loads((FIX / "crm_retry_flow.json").read_text())
    res = svc.advise(flow, preset="tiny", episode_id="ep-1")
    assert res["injected"] is True
    assert res["block"].startswith("[World-Model guidance]\n")
