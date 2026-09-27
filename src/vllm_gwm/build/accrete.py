# SPDX-License-Identifier: MIT
"""Self-evolving graph: in-place accretion of live evidence onto a loaded graph.

Two mechanisms, both label-free:

* **H27s self-referential accretion** — the action the judge selected at a
  classified state is prepended to that state's exemplar list, so later steps at
  the same state see the policy's own recent behaviour as evidence.
* **H28 state minting** — when a step abstains (no train centroid within the
  state's P90 radius), greedy leader-clustering over previously abstained
  embeddings either attaches it to an existing *candidate* centroid or starts a
  new one. Candidates are held in the accretor, NOT in the ``Scorer``, until
  they reach ``mint_min_support`` visits; only then are they promoted into the
  centroid matrix. Holding them outside the scorer is what makes the support
  gate real: presets with ``unify_domains`` classify against every centroid row
  regardless of ``by_dom``, so a row parked in the matrix is live immediately.

Mutation is applied to the live ``Scorer`` in place rather than by building and
swapping a graph directory: it needs no re-clustering, no embedding recompute
beyond the classification the select path already performs, and it sidesteps the
version-pointer path entirely.

No outcome signal is consumed. On a teacher-forced replay benchmark such as
Toucan the only available correctness signal is the gold action, and accreting
on it would be training on test labels (TOUCAN_PERF_HYPOTHESES G.3).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vllm_gwm.harness import render


@dataclass
class AccreteStats:
    steps: int = 0
    classified: int = 0
    abstained: int = 0
    accreted: int = 0
    minted: int = 0
    minted_hits: int = 0
    minted_supported: int = 0
    purges: int = 0
    per_state: dict[str, int] = field(default_factory=dict)
    # Distance from each abstained state to its nearest already-minted centroid.
    # Without this there is no way to tell "the UNKNOWN band does not recur"
    # from "the attach radius is simply mis-scaled".
    mint_dists: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        skip = ("per_state", "mint_dists")
        d = {k: v for k, v in self.__dict__.items() if k not in skip}
        d["minted_states"] = len(self.per_state)
        if self.mint_dists:
            import numpy as _np

            qs = _np.percentile(self.mint_dists, [5, 25, 50, 75, 95])
            d["mint_dist_pct"] = {
                str(q): round(float(v), 4)
                for q, v in zip((5, 25, 50, 75, 95), qs)
            }
        return d


class GraphAccretor:
    """Per-graph accretion state. One instance per resolved graph path."""

    def __init__(
        self,
        *,
        accrete_self: bool = False,
        mint_states: bool = False,
        max_per_state: int = 8,
        mint_min_support: int = 3,
        mint_radius: float = 0.0,
        scope: str = "global",
        snapshot_path: Path | None = None,
        snapshot_every: int = 50,
    ):
        self.accrete_self = bool(accrete_self)
        self.mint_states = bool(mint_states)
        self.max_per_state = max(1, int(max_per_state))
        self.mint_min_support = max(1, int(mint_min_support))
        self.mint_radius = float(mint_radius)
        self.scope = (scope or "global").strip().lower()
        self.snapshot_path = snapshot_path
        self.snapshot_every = max(0, int(snapshot_every))
        self.stats = AccreteStats()
        self._lock = threading.Lock()
        self._minted: list[int] = []          # promoted rows in scorer.cents
        self._minted_visits: dict[str, int] = {}
        # Support-gated candidates, held OUTSIDE the scorer so that presets with
        # unify_domains (which ignore by_dom) cannot classify against them early.
        self._cand_vecs: list[np.ndarray] = []
        self._cand_visits: list[int] = []
        self._added: list[tuple[str, str]] = []   # (domain, state) with accreted entries
        self._episode = ""

    @property
    def enabled(self) -> bool:
        return self.accrete_self or self.mint_states

    # ---------------------------------------------------------------- embedding

    @staticmethod
    def _embed(scorer: Any, flow: list[dict[str, Any]]) -> np.ndarray:
        """Mirror ``Scorer.classify``'s text/embedding contract exactly."""
        text = render.render_text(flow, scorer.mtr, scorer.mfc)[-scorer.tail:]
        return scorer.model.encode([text], normalize_embeddings=True)[0]

    def _nearest(self, scorer: Any, vec: np.ndarray, cand: list[int]):
        if not cand:
            return None, 1.0
        sims = scorer.cents[cand] @ vec
        j = int(np.argmax(sims))
        return scorer.ids[cand[j]], 1.0 - float(sims[j])

    # ------------------------------------------------------------------ minting

    def _radius_for(self, scorer: Any) -> float:
        if self.mint_radius > 0:
            return self.mint_radius
        # Default: the graph's own median P90 radius, so a minted state is no
        # more permissive than a mined one.
        vals = [float(v) for v in scorer.p90.values() if v]
        return float(np.median(vals)) if vals else 0.35

    def _mint_or_attach(self, scorer: Any, vec: np.ndarray, domain: str) -> str | None:
        """Grow support for the nearest candidate cluster; promote it into the
        graph once it has been seen ``mint_min_support`` times."""
        radius = self._radius_for(scorer)
        with self._lock:
            j, dist = -1, 1.0
            if self._cand_vecs:
                sims = np.vstack(self._cand_vecs) @ vec
                j = int(np.argmax(sims))
                dist = 1.0 - float(sims[j])
                self.stats.mint_dists.append(dist)
            if j < 0 or dist > radius:
                self._cand_vecs.append(vec)
                self._cand_visits.append(1)
                self.stats.minted += 1
                if self.mint_min_support > 1:
                    return None
                j = len(self._cand_vecs) - 1
            else:
                self._cand_visits[j] += 1
                self.stats.minted_hits += 1
            if self._cand_visits[j] < self.mint_min_support:
                return None
            sid = f"{domain}:minted{len(self._minted)}"
            # Order matters: the row must exist in ``cents`` before any index
            # or id can point at it.
            scorer.cents = np.vstack([scorer.cents, self._cand_vecs[j].reshape(1, -1)])
            idx = scorer.cents.shape[0] - 1
            scorer.ids.append(sid)
            scorer.p90[sid] = radius
            scorer.dom_of[sid] = domain
            self._minted.append(idx)
            self._minted_visits[sid] = self._cand_visits[j]
            self.stats.per_state[sid] = self._cand_visits[j]
            self.stats.minted_supported += 1
            self._publish(scorer, domain, idx)
            # Retire the candidate: further visits now classify onto the real row.
            self._cand_vecs.pop(j)
            self._cand_visits.pop(j)
            return sid

    @staticmethod
    def _publish(scorer: Any, domain: str, idx: int) -> None:
        """Expose a minted centroid to ``Scorer.classify``. The row already
        exists in ``cents``, so appending the index cannot dangle."""
        if idx not in scorer.by_dom[domain]:
            scorer.by_dom[domain].append(idx)

    # ---------------------------------------------------------------- accretion

    def _accrete(self, scorer: Any, domain: str, state: str, text: str, dist: float) -> None:
        if not text.strip():
            return
        dom = scorer.resolve_domain(state, domain)
        with self._lock:
            by_dom = scorer.examples.setdefault(dom, {})
            states = by_dom.setdefault("states", {})
            entries = states.setdefault(state, [])
            # Prepend: ``find_similar`` slices from the head, so appended
            # entries behind a long mined list would never be rendered.
            entries.insert(0, {"text": text, "cos": 1.0 - float(dist), "accreted": True})
            # The cap is on ACCRETED entries only — counting mined exemplars
            # against it evicts each new accretion the moment it is added.
            n_acc = sum(1 for e in entries if e.get("accreted"))
            while n_acc > self.max_per_state:
                for i in range(len(entries) - 1, -1, -1):
                    if entries[i].get("accreted"):
                        entries.pop(i)
                        break
                n_acc -= 1
            if (dom, state) not in self._added:
                self._added.append((dom, state))
            self.stats.accreted += 1

    def _purge(self, scorer: Any) -> None:
        """H30a episode scope: drop every accreted entry between episodes."""
        with self._lock:
            for dom, state in self._added:
                entries = (scorer.examples.get(dom) or {}).get("states", {}).get(state)
                if entries:
                    entries[:] = [e for e in entries if not e.get("accreted")]
            self._added.clear()
            self._minted_visits.clear()
            self._cand_vecs.clear()
            self._cand_visits.clear()
            self.stats.purges += 1

    # --------------------------------------------------------------------- step

    def step(
        self,
        scorer: Any,
        domain: str,
        pre_flow: list[dict[str, Any]],
        selected_text: str,
        episode_id: str = "",
    ) -> dict[str, Any]:
        """Observe one select step: classify the pre-action state (minting on
        abstain) and accrete the selected action there."""
        if not self.enabled:
            return {}
        if self.scope == "episode" and episode_id and episode_id != self._episode:
            if self._added or self._minted_visits:
                self._purge(scorer)
            self._episode = episode_id
        vec = self._embed(scorer, pre_flow)
        # Mirror Scorer.classify's candidate set, so `state` here is exactly the
        # state the judge was given for this step (published minted rows included).
        cand = (
            list(range(len(scorer.ids)))
            if scorer.unify_domains
            else list(scorer.by_dom.get(domain, []))
        )
        state, dist = self._nearest(scorer, vec, cand)
        abstain = state is None or dist > scorer.p90.get(state, 1.0) * scorer.p90_scale
        self.stats.steps += 1
        if abstain:
            self.stats.abstained += 1
            state = self._mint_or_attach(scorer, vec, domain) if self.mint_states else None
            dist = 0.0
        else:
            self.stats.classified += 1
        if state is not None and self.accrete_self:
            self._accrete(scorer, domain, state, selected_text, dist)
        self._maybe_snapshot()
        return {"state": state, "abstain": abstain, **self.stats.as_dict()}

    def _maybe_snapshot(self) -> None:
        if not self.snapshot_path or not self.snapshot_every:
            return
        if self.stats.steps % self.snapshot_every:
            return
        try:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_path.write_text(
                json.dumps(
                    {
                        "stats": self.stats.as_dict(),
                        "minted_visits": self._minted_visits,
                        "accreted_states": [f"{d}/{s}" for d, s in self._added],
                        "config": {
                            "accrete_self": self.accrete_self,
                            "mint_states": self.mint_states,
                            "max_per_state": self.max_per_state,
                            "mint_min_support": self.mint_min_support,
                            "scope": self.scope,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
