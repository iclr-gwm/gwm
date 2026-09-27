# SPDX-License-Identifier: MIT
"""GraphAccretor: exemplar accretion, ring cap, minting/support, episode scope."""

from collections import defaultdict

import numpy as np
import pytest

from vllm_gwm.build.accrete import GraphAccretor


class FakeScorer:
    """Minimal stand-in exposing the attributes GraphAccretor mutates."""

    def __init__(self):
        self.cents = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.ids = ["d:0", "d:1"]
        self.p90 = {"d:0": 0.2, "d:1": 0.2}
        self.dom_of = {"d:0": "d", "d:1": "d"}
        self.by_dom = defaultdict(list, {"d": [0, 1]})
        self.p90_scale = 1.0
        self.unify_domains = False
        self.examples = {"d": {"states": {"d:0": [{"text": "mined"}]}}}
        self.mtr = self.mfc = self.tail = 100
        self.model = None

    def resolve_domain(self, state, domain):
        return domain


def _acc(**kw):
    return GraphAccretor(**kw)


def _patch_embed(monkeypatch, vec):
    monkeypatch.setattr(
        GraphAccretor, "_embed", staticmethod(lambda s, f: np.array(vec, dtype=np.float32))
    )


def test_accreted_entry_is_prepended_and_mined_examples_survive():
    sc, acc = FakeScorer(), _acc(accrete_self=True, max_per_state=2)
    acc._accrete(sc, "d", "d:0", "first", 0.1)
    entries = sc.examples["d"]["states"]["d:0"]
    assert entries[0]["text"] == "first" and entries[0]["accreted"]
    assert entries[-1]["text"] == "mined"


def test_ring_cap_counts_only_accreted_entries():
    sc, acc = FakeScorer(), _acc(accrete_self=True, max_per_state=2)
    for i in range(5):
        acc._accrete(sc, "d", "d:0", f"a{i}", 0.1)
    entries = sc.examples["d"]["states"]["d:0"]
    assert sum(1 for e in entries if e.get("accreted")) == 2
    assert [e["text"] for e in entries if e.get("accreted")] == ["a4", "a3"]
    assert entries[-1]["text"] == "mined"


def test_classified_step_accretes_at_that_state(monkeypatch):
    sc, acc = FakeScorer(), _acc(accrete_self=True)
    _patch_embed(monkeypatch, [1.0, 0.0])
    rec = acc.step(sc, "d", [], "chosen", episode_id="e1")
    assert rec["state"] == "d:0" and rec["abstain"] is False and rec["accreted"] == 1
    assert sc.examples["d"]["states"]["d:0"][0]["text"] == "chosen"


def test_minting_waits_for_support_then_becomes_classifiable(monkeypatch):
    sc, acc = FakeScorer(), _acc(mint_states=True, mint_min_support=3, mint_radius=0.1)
    _patch_embed(monkeypatch, [0.7071, 0.7071])  # equidistant → abstains
    assert acc.step(sc, "d", [], "x")["state"] is None      # minted, unpublished
    assert acc.stats.minted == 1
    assert 2 not in sc.by_dom["d"]
    assert acc.step(sc, "d", [], "x")["state"] is None      # visit 2
    assert acc.step(sc, "d", [], "x")["state"] == "d:minted0"  # support reached
    assert 2 in sc.by_dom["d"]                              # now visible to classify


def test_minting_does_not_dangle_the_centroid_index(monkeypatch):
    sc, acc = FakeScorer(), _acc(mint_states=True, mint_min_support=1, mint_radius=0.1)
    _patch_embed(monkeypatch, [0.7071, 0.7071])
    acc.step(sc, "d", [], "x")
    for idx in sc.by_dom["d"]:
        assert idx < sc.cents.shape[0]
    assert len(sc.ids) == sc.cents.shape[0]


def test_episode_scope_purges_accretions_between_episodes(monkeypatch):
    sc = FakeScorer()
    acc = _acc(accrete_self=True, scope="episode")
    _patch_embed(monkeypatch, [1.0, 0.0])
    acc.step(sc, "d", [], "from-e1", episode_id="e1")
    assert any(e.get("accreted") for e in sc.examples["d"]["states"]["d:0"])
    acc.step(sc, "d", [], "from-e2", episode_id="e2")
    texts = [e["text"] for e in sc.examples["d"]["states"]["d:0"]]
    assert texts == ["from-e2", "mined"]  # e1's evidence gone, mined intact


def test_global_scope_carries_evidence_across_episodes(monkeypatch):
    sc, acc = FakeScorer(), _acc(accrete_self=True, scope="global")
    _patch_embed(monkeypatch, [1.0, 0.0])
    acc.step(sc, "d", [], "from-e1", episode_id="e1")
    acc.step(sc, "d", [], "from-e2", episode_id="e2")
    assert sum(1 for e in sc.examples["d"]["states"]["d:0"] if e.get("accreted")) == 2


def test_disabled_accretor_is_a_noop(monkeypatch):
    sc, acc = FakeScorer(), _acc()
    _patch_embed(monkeypatch, [1.0, 0.0])
    assert acc.enabled is False
    assert acc.step(sc, "d", [], "x") == {}
    assert sc.examples["d"]["states"]["d:0"] == [{"text": "mined"}]


def test_empty_selection_is_not_accreted():
    sc, acc = FakeScorer(), _acc(accrete_self=True)
    acc._accrete(sc, "d", "d:0", "   ", 0.1)
    assert sc.examples["d"]["states"]["d:0"] == [{"text": "mined"}]


@pytest.mark.parametrize("scope", ["global", "episode"])
def test_stats_are_reported(monkeypatch, scope):
    sc, acc = FakeScorer(), _acc(accrete_self=True, scope=scope)
    _patch_embed(monkeypatch, [1.0, 0.0])
    rec = acc.step(sc, "d", [], "x", episode_id="e1")
    assert rec["steps"] == 1 and rec["classified"] == 1 and rec["accreted"] == 1


def test_support_gate_holds_under_unify_domains(monkeypatch):
    """A candidate must not be classifiable before it earns support.

    unify_domains presets rank against every row in ``cents`` and ignore
    ``by_dom``, so parking an unsupported centroid in the matrix would make it
    live immediately and silently bypass the gate.
    """
    sc = FakeScorer()
    sc.unify_domains = True
    acc = _acc(mint_states=True, mint_min_support=3, mint_radius=0.1)
    _patch_embed(monkeypatch, [0.7071, 0.7071])
    n_rows = sc.cents.shape[0]
    for _ in range(2):
        assert acc.step(sc, "d", [], "x")["state"] is None
        assert sc.cents.shape[0] == n_rows      # nothing parked in the matrix
        assert len(sc.ids) == n_rows
    assert acc.step(sc, "d", [], "x")["state"] == "d:minted0"
    assert sc.cents.shape[0] == n_rows + 1      # promoted only now
    assert acc.stats.minted_supported == 1


def test_repeated_visits_after_promotion_do_not_remint(monkeypatch):
    sc, acc = FakeScorer(), _acc(mint_states=True, mint_min_support=2, mint_radius=0.1)
    _patch_embed(monkeypatch, [0.7071, 0.7071])
    acc.step(sc, "d", [], "x")
    acc.step(sc, "d", [], "x")                  # promoted
    rows = sc.cents.shape[0]
    acc.step(sc, "d", [], "x")                  # now a normal classified step
    assert sc.cents.shape[0] == rows
    assert acc.stats.minted == 1
