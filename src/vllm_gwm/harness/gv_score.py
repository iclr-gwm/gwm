#!/usr/bin/env python3
"""Shared math for the ``graph_value`` candidate scorer (single source of truth).

Imported by BOTH ``graph_sidecar.py`` (live, in-loop best-of-N selection) and
``offline_graph_value.py`` (CPU tuning/eval), so offline-tuned weights transfer
byte-identically to the live run.

A candidate action from current state ``s`` proposing toolset ``tk`` is scored as a
COMPOSITE ADVANTAGE over the state value ``V(s)`` (all N candidates at a step share
``s``, so ranking by advantage == ranking, but keeps the number interpretable):

  score = base
        + w_action  * action_adv            # (1) EB-shrunk action success vs V(s)
        + discount * w_lookahead * look_adv  # (2) one-step lookahead advantage
        - trap_pen                           # (3) penalty for routing into fail-lifted states
        - freq_pen                           # (4) de-emphasise frequent non-advancing actions

Design notes tied to the prior null (see REPORT_WM_GRAPH_SELECTION.md):
  (1) Empirical-Bayes shrinkage toward V(s) by the edge sample count ``n`` REPLACES
      the crude ``n>=3`` hard gate that tied most low-n same-state candidates.
  (2) Lookahead sums ONLY over PROGRESSING next states (drops self-loop s'==s, END,
      noise) and normalises by progressing mass — this is the phantom-credit guard the
      earlier descriptive ``graph_value`` injection lacked.
  (3)/(4) directly counter the documented "frequent no-op wins" failure mode.

Stdlib only.
"""
from __future__ import annotations

import os

# Tuned defaults (overridable per-weight via WM_GV_* env or offline search).
DEFAULTS = {
    "K": 8.0,            # EB shrinkage prior strength (pseudo-rollouts)
    "w_action": 1.0,     # weight on shrunken action advantage
    "w_lookahead": 0.5,  # weight on one-step lookahead advantage
    "discount": 0.9,     # gamma discount on lookahead
    "w_trap": 0.3,       # trap / fail-lift penalty
    "alpha": 0.1,        # frequency de-emphasis on non-advancing actions
}

_SENTINEL = ("END", "noise")


def default_weights() -> dict:
    return dict(DEFAULTS)


def weights_from_env(env=None) -> dict:
    """Read WM_GV_K / WM_GV_W_ACTION / WM_GV_W_LOOKAHEAD / WM_GV_DISCOUNT /
    WM_GV_W_TRAP / WM_GV_ALPHA, falling back to DEFAULTS."""
    env = env if env is not None else os.environ
    keymap = {
        "K": "WM_GV_K", "w_action": "WM_GV_W_ACTION", "w_lookahead": "WM_GV_W_LOOKAHEAD",
        "discount": "WM_GV_DISCOUNT", "w_trap": "WM_GV_W_TRAP", "alpha": "WM_GV_ALPHA",
    }
    w = default_weights()
    for k, envk in keymap.items():
        v = env.get(envk)
        if v is not None and str(v).strip() != "":
            w[k] = float(v)
    return w


def graph_value_step(edge: dict, s: str, V: dict, base: float,
                     lift_of: dict, trap: set, w: dict) -> dict:
    """Score one candidate action.

    edge   : tool_edges[s][tk] = {n, p_action, succ, next:{state:P(next|s,tk)}}
    s      : current (pre-candidate) state id
    V      : {state -> value (1 - fail_rate)}; base used for unseen states
    lift_of: {state -> fail-lift} (1.0 == base); trap: set of trap state ids
    w      : weights dict (see DEFAULTS)

    Returns {"score", "action_adv", "look_adv", "trap_pen", "freq_pen", "s_land"}.
    """
    def v(st: str) -> float:
        return float(V.get(st, base))

    Vs = v(s)
    n = float(edge.get("n", 0) or 0)
    succ = float(edge.get("succ", base))
    p_action = float(edge.get("p_action", 0.0) or 0.0)
    nxt = edge.get("next", {}) or {}

    # (1) Empirical-Bayes shrinkage of action success toward V(s).
    K = float(w["K"])
    sh = (n * succ + K * Vs) / (n + K) if (n + K) > 0 else Vs
    action_adv = sh - Vs

    # (2) One-step lookahead advantage over PROGRESSING next states only.
    prog = {st: p for st, p in nxt.items() if st != s and st not in _SENTINEL}
    pm = sum(prog.values())
    if pm > 0:
        lookahead_V = sum(p * v(st) for st, p in prog.items()) / pm
        look_adv = lookahead_V - Vs
        s_land = max(prog, key=prog.get)
    else:
        look_adv = 0.0
        s_land = max(nxt, key=nxt.get) if nxt else s

    # (3) Trap / fail-lift penalty on the modal (progressing) landing state.
    lift = float(lift_of.get(s_land, 1.0))
    trap_pen = float(w["w_trap"]) * max(0.0, lift - 1.0)
    if s_land in trap:
        trap_pen += float(w["w_trap"])

    # (4) Frequency de-emphasis: p_action only hurts, and only when non-advancing.
    freq_pen = (float(w["alpha"]) * p_action
                if (action_adv <= 0.0 and look_adv <= 0.0) else 0.0)

    score = (base
             + float(w["w_action"]) * action_adv
             + float(w["discount"]) * float(w["w_lookahead"]) * look_adv
             - trap_pen - freq_pen)
    return {"score": score, "action_adv": action_adv, "look_adv": look_adv,
            "trap_pen": trap_pen, "freq_pen": freq_pen, "s_land": s_land}
