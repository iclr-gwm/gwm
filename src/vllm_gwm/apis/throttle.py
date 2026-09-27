# SPDX-License-Identifier: MIT
"""Shared quota / rate-limit gate for outbound API calls."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _quota_usage_percent(headers: Any) -> float | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    for key in ("x-quota-usage-percent", "X-Quota-Usage-Percent"):
        raw = getter(key)
        if raw is not None:
            try:
                return float(str(raw).strip())
            except ValueError:
                return None
    return None


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Retry-After") or getter("retry-after")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def is_quota_error(status: int, body: str) -> bool:
    if status == 429:
        return True
    if status == 503 and any(
        tok in body.lower() for tok in ("rate", "quota", "capacity", "limit")
    ):
        return True
    low = body.lower()
    for code in ("rate_limit_exceeded", "insufficient_quota", "quota_exceeded"):
        if code in low:
            return True
    return False


@dataclass
class ThrottleConfig:
    max_retries: int = 6
    retry_step_sec: float = 10.0
    quota_slow_percent: float = 80.0
    timeout: float = 180.0


@dataclass
class _GateState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    waiters: int = 0
    in_flight: int = 0
    cooldown_until: float = 0.0
    last_quota_percent: float | None = None
    last_quota_reset: str | None = None


class QuotaGate:
    """Process-wide throttle keyed by (host, api_key)."""

    _gates: dict[tuple[str, str], _GateState] = {}
    _meta_lock = threading.Lock()

    def __init__(self, key: tuple[str, str], cfg: ThrottleConfig):
        self.key = key
        self.cfg = cfg
        with self._meta_lock:
            if key not in self._gates:
                self._gates[key] = _GateState()
            self._state = self._gates[key]

    def _proactive_delay(self) -> float:
        pct = self._state.last_quota_percent
        if pct is None or pct < self.cfg.quota_slow_percent:
            return 0.0
        steps = max(0.0, (pct - self.cfg.quota_slow_percent) / 10.0)
        return steps * self.cfg.retry_step_sec

    def _waiter_extra(self, waiters: int) -> float:
        return max(0, waiters - 1) * self.cfg.retry_step_sec

    def acquire(self) -> None:
        with self._state.lock:
            self._state.waiters += 1
            waiters = self._state.waiters
            now = time.monotonic()
            delay = max(0.0, self._state.cooldown_until - now)
            delay += self._proactive_delay()
            delay += self._waiter_extra(waiters)
        if delay > 0:
            time.sleep(delay + random.uniform(0, 1.0))
        with self._state.lock:
            self._state.waiters -= 1
            self._state.in_flight += 1

    def release(self) -> None:
        with self._state.lock:
            self._state.in_flight = max(0, self._state.in_flight - 1)

    def note_success(self, headers: Any) -> None:
        pct = _quota_usage_percent(headers)
        reset = None
        getter = getattr(headers, "get", None) if headers is not None else None
        if callable(getter):
            reset = getter("x-quota-reset-date") or getter("X-Quota-Reset-Date")
        with self._state.lock:
            if pct is not None:
                self._state.last_quota_percent = pct
            if reset:
                self._state.last_quota_reset = str(reset)
            self._state.cooldown_until = 0.0

    def note_quota_failure(self, attempt: int, headers: Any) -> float:
        base = attempt * self.cfg.retry_step_sec
        with self._state.lock:
            extra = self._waiter_extra(self._state.waiters + self._state.in_flight)
            delay = base + extra
            retry_after = _retry_after_seconds(headers)
            if retry_after is not None:
                delay = max(delay, retry_after)
            self._state.cooldown_until = max(
                self._state.cooldown_until, time.monotonic() + delay
            )
        return delay
