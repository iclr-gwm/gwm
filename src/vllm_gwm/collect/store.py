# SPDX-License-Identifier: Apache-2.0
"""Append-only rollout collection store."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any


class RolloutStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "rollouts.jsonl"
        self._feedback_path = self.root / "feedback.jsonl"
        self._lock = threading.Lock()
        self._finalized: set[str] = set()
        self._approved_count = 0
        self._hydrate()

    def append(
        self,
        flow: list[dict[str, Any]],
        *,
        episode_id: str = "",
        success: bool | None = None,
        source: str = "client",
        finalize: bool = False,
    ) -> str:
        rid = str(uuid.uuid4())
        record = {
            "id": rid,
            "episode_id": episode_id,
            "flow": flow,
            "success": success,
            "source": source,
            "finalized": finalize,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if finalize and episode_id and episode_id not in self._finalized:
                self._finalized.add(episode_id)
                if success is not None:
                    self._approved_count += 1
        return rid

    def record_feedback(self, *, episode_id: str, success: bool, source: str = "explicit") -> None:
        with self._lock:
            with self._feedback_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"episode_id": episode_id, "success": success, "source": source},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if episode_id and episode_id not in self._finalized:
                self._finalized.add(episode_id)
                self._approved_count += 1

    def _hydrate(self) -> None:
        """Reload finalized episode ids from an existing jsonl (durable across restarts)."""
        for rec in self.load_records():
            if rec.get("finalized") and rec.get("episode_id"):
                self._finalized.add(str(rec["episode_id"]))
                if rec.get("success") is not None:
                    self._approved_count += 1

    def count_finalized(self) -> int:
        with self._lock:
            return len(self._finalized)

    def since_last_build(self) -> int:
        with self._lock:
            return self._approved_count

    def snapshot_path(self) -> Path:
        snap = self.root / "snapshots"
        snap.mkdir(parents=True, exist_ok=True)
        target = snap / f"rollouts_{uuid.uuid4().hex[:8]}.jsonl"
        if self._path.exists():
            target.write_bytes(self._path.read_bytes())
        return target

    def load_records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
