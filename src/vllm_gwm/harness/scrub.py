"""Generic, config-driven noise scrubber for tool args + tool results.

The workflow-mining pipeline embeds rendered conversation tails (state discovery /
transition mining) and injects rendered tails as few-shot shots to the agent. Both
paths carried raw infrastructure noise — UUIDs, ISO timestamps, ``@odata`` protocol
envelopes, base64 message bodies, null fields — which (a) dilutes the embedding so
clustering keys partly on tenant ids/dates rather than workflow semantics, and
(b) wastes agent context on non-generalizable tokens.

This module scrubs the *structured* args/result objects BEFORE they are flattened
to text, at the single chokepoint in :mod:`render`. It is driven entirely by
``scrub_config.json`` so the same code generalizes to another benchmark by editing
config (denylist + value-redaction needs zero per-tool knowledge; an optional
per-tool allowlist handles the rare case where a backend's signal keys are known).

Set env ``WM_SCRUB=0`` to disable (identity passthrough) for A/B comparison.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_DEFAULT_CFG = _HERE / "scrub_config.json"


class Scrubber:
    def __init__(self, cfg: dict):
        self.drop_keys = set(cfg.get("drop_keys", []))
        self.drop_key_patterns = [re.compile(p) for p in cfg.get("drop_key_patterns", [])]
        self.value_redactors = [
            (re.compile(r["pattern"]), r["repl"]) for r in cfg.get("value_redactors", [])
        ]
        self.drop_null = bool(cfg.get("drop_null", True))
        self.max_string_len = int(cfg.get("max_string_len", 0) or 0)
        # tool -> set of top-level keys to KEEP (everything else dropped). Optional.
        self.keep_keys_by_tool = {
            t: set(ks) for t, ks in (cfg.get("keep_keys_by_tool") or {}).items()
        }

    def _drop_key(self, k: str) -> bool:
        if k in self.drop_keys:
            return True
        return any(p.search(k) for p in self.drop_key_patterns)

    def _redact_str(self, s: str) -> str:
        for pat, repl in self.value_redactors:
            s = pat.sub(repl, s)
        if self.max_string_len and len(s) > self.max_string_len:
            s = s[: self.max_string_len] + "…"
        return s

    def _empty(self, v: Any) -> bool:
        return v is None or v == "" or v == [] or v == {}

    def scrub(self, obj: Any, tool: str | None = None) -> Any:
        """Recursively scrub ``obj``. ``tool`` enables the per-tool allowlist at the
        top level only (nested structure of a kept key is preserved, then denylist
        rules still apply within it)."""
        if isinstance(obj, dict):
            allow = self.keep_keys_by_tool.get(tool) if tool else None
            out = {}
            for k, v in obj.items():
                if isinstance(k, str):
                    if allow is not None and k not in allow:
                        continue
                    if self._drop_key(k):
                        continue
                v2 = self.scrub(v)  # tool allowlist applies at top level only
                if self.drop_null and self._empty(v2):
                    continue
                out[k] = v2
            return out
        if isinstance(obj, list):
            return [self.scrub(x) for x in obj]
        if isinstance(obj, str):
            return self._redact_str(obj)
        return obj

    def scrub_text(self, s: str) -> str:
        """Scrub a free-text (non-JSON) blob: value redaction only, no key drops."""
        return self._redact_str(s or "")


def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_CFG
    if p.exists():
        return json.loads(p.read_text())
    return {}


_SINGLETON: Scrubber | None = None


def get_scrubber() -> Scrubber | None:
    """Process-wide scrubber from ``scrub_config.json``; ``None`` if disabled via
    ``WM_SCRUB=0`` so callers cleanly fall back to identity."""
    global _SINGLETON
    if os.getenv("WM_SCRUB", "1") == "0":
        return None
    if _SINGLETON is None:
        _SINGLETON = Scrubber(load_config(os.getenv("WM_SCRUB_CONFIG")))
    return _SINGLETON
