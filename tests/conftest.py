# SPDX-License-Identifier: MIT
import hashlib
import json
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).resolve().parent / "fixtures"
TINY = FIX / "tiny_graph"


class FakeScorer:
    """Lightweight graph scorer for offline GWM tests (no MiniLM)."""

    def __init__(self, transitions_path, centroids_dir, examples_path=None, **_kw):
        self.trans = json.loads(Path(transitions_path).read_text())
        self.examples = {}
        self.unify_domains = False
        self.mtr, self.mfc = 2000, 60000
        self.succ, self.base_succ, self.lift, self.trap, self.top_tool = {}, {}, {}, set(), {}
        for dom, dd in self.trans.items():
            self.base_succ[dom] = 1.0 - (dd.get("base_fail_rate", 0.0) or 0.0)
            base = dd.get("base_fail_rate", 0) or 1e-9
            for sf in dd.get("state_fail", []):
                s = str(sf["state"])
                self.succ[s] = 1.0 - float(sf["fail_rate"])
                if sf["n_rollouts"] >= 20 and (sf["fail_rate"] / base) >= 1.3:
                    self.trap.add(s)

    def base_for(self, domain):
        return self.base_succ.get(domain, 0.3)

    def resolve_domain(self, state, domain):
        return domain

    def classify(self, flow, domain):
        cand = [s for s in self.succ if s.split(":")[0] == domain] or list(self.succ)
        has_fail = any(
            isinstance(ev, dict)
            and ev.get("type") == "tool_result"
            and isinstance(ev.get("result"), dict)
            and (ev["result"].get("success") is False or ev["result"].get("error"))
            for ev in flow
        )
        trap_states = [s for s in cand if s in self.trap]
        state = trap_states[0] if has_fail and trap_states else sorted(cand)[-1]
        return state, 0.1

    def action_score(self, flow, domain):
        return (0.3, None)


@pytest.fixture()
def patch_scorer(monkeypatch):
    from vllm_gwm.harness import graph_sidecar

    monkeypatch.setattr(graph_sidecar, "Scorer", FakeScorer)


@pytest.fixture()
def tiny_preset():
    from vllm_gwm.presets import Preset

    return Preset(name="tiny", default_graph=str(TINY), default_domain="toy")
STUB_DIM = 16
STUB_MODEL = "stub/tiny-embedder"
_TOOL_RE = re.compile(r"\[tool_call\] ([A-Za-z0-9_.-]+)")


def _unit(seed: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=dim)
    return v / (np.linalg.norm(v) + 1e-12)


def stub_vector(text: str, dim: int = STUB_DIM) -> np.ndarray:
    """Deterministic pseudo-embedding: a per-tool-signature direction + small noise.

    Structured (not pure noise) so steps that call the same tools land near each
    other, the way a real encoder would — enough for clustering/centroids to be
    meaningful in tests.
    """
    tools = "|".join(sorted(set(_TOOL_RE.findall(text or "")))) or "no-tool"
    base = _unit(f"tool:{tools}", dim)
    noise = 0.15 * _unit(f"text:{text}", dim)
    v = base + noise
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


class StubSentenceTransformer:
    """Offline stand-in for ``sentence_transformers.SentenceTransformer``."""

    max_seq_length = 128

    def __init__(self, model_id, device=None, trust_remote_code=False, **_kw):
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code

    def get_sentence_embedding_dimension(self):
        return STUB_DIM

    def encode(self, texts, **_kw):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, STUB_DIM), dtype=np.float32)
        return np.vstack([stub_vector(t) for t in texts]).astype(np.float32)


@pytest.fixture
def stub_sentence_transformers(monkeypatch):
    """Install a fake ``sentence_transformers`` module (no download, no torch).

    Both the mining embed stage and the serve-time Scorer resolve the class
    through ``sys.modules``, so this covers build and load in one shot.
    """
    mod = types.ModuleType("sentence_transformers")
    mod.__version__ = "stub"
    mod.SentenceTransformer = StubSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return mod


@pytest.fixture
def stub_labeller(monkeypatch):
    """Replace UMAP+HDBSCAN with k-means, so the DAG runs without those extras."""
    from sklearn.cluster import KMeans

    from vllm_gwm.build.mining import discover

    def labels(X, train_mask, *, n_components=2, n_neighbors=3, min_cluster_size=2,
               min_samples=None, seed=17):
        k = max(1, min(2, int(train_mask.sum())))
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X[train_mask])
        return km.predict(X).astype(int)

    monkeypatch.setattr(discover, "umap_hdbscan_labels", labels)
    return labels
