# SPDX-License-Identifier: Apache-2.0
"""Self-evolution manager: trigger builds after N finalized rollouts."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vllm_gwm.build import pipeline
from vllm_gwm.build.layout import builds_dir
from vllm_gwm.collect.outcome import OutcomeResolver
from vllm_gwm.collect.store import RolloutStore
from vllm_gwm.config import GwmConfig
from vllm_gwm.runtime.registry import GraphRegistry


@dataclass
class BuildJob:
    id: str
    status: str
    adapter: str
    started_at: str
    finished_at: str = ""
    error: str = ""
    version_path: str = ""


class EvolveManager:
    def __init__(
        self,
        cfg: GwmConfig,
        store: RolloutStore,
        registry: GraphRegistry,
        outcome: OutcomeResolver,
    ):
        self.cfg = cfg
        self.store = store
        self.registry = registry
        self.outcome = outcome
        self._jobs: list[BuildJob] = []
        self._lock = threading.Lock()
        self._running = False
        self._last_count = 0

    def maybe_trigger(self, adapter: str) -> BuildJob | None:
        return self._enqueue(adapter, force=False)

    def trigger(self, adapter: str, *, force: bool = True) -> BuildJob | None:
        """Enqueue a rebuild. ``force`` skips the every-N threshold (admin / live tests)."""
        return self._enqueue(adapter, force=force)

    def _enqueue(self, adapter: str, *, force: bool) -> BuildJob | None:
        name = (adapter or "").strip()
        if not name:
            name = next(iter(self.registry.registered), "")
        if not name:
            return None
        if not force and self.cfg.evolve_every_n <= 0:
            return None
        count = self.store.count_finalized()
        with self._lock:
            if not force and count - self._last_count < self.cfg.evolve_every_n:
                return None
            if self._running:
                return None
            self._running = True
            self._last_count = count
        job = BuildJob(
            id=uuid.uuid4().hex[:8],
            status="running",
            adapter=name,
            started_at=datetime.now(tz=UTC).isoformat(),
        )
        self._jobs.append(job)
        threading.Thread(target=self._run_build, args=(job,), daemon=True).start()
        return job

    def _run_build(self, job: BuildJob) -> None:
        try:
            snap = self.store.snapshot_path()
            out_dir = builds_dir(self.cfg.data_root) / job.adapter / job.id
            out_dir.mkdir(parents=True, exist_ok=True)
            if os.getenv("VLLM_GWM_EVOLVE_ALLOW_MINIMAL", "").strip().lower() in (
                "1", "true", "yes", "on"
            ):
                # Escape hatch for tests only. _minimal_build emits a 1-state,
                # zero-centroid toy graph; activating it over a working bundle
                # destroys the graph, so it is opt-in and never the fallback.
                self._minimal_build(snap, out_dir)
            else:
                # The mining pipeline lives in-process now (no build script to
                # shell out to). Any failure propagates and the job fails
                # closed — the previous bundle stays active.
                pipeline.ingest_rollouts(snap, out_dir)
                kwargs: dict[str, Any] = {
                    "embed_device": self.cfg.build_embed_device or None,
                    "watched_tools": self.cfg.build_watched_tools,
                }
                if self.cfg.build_embed_model:
                    kwargs["embed_model"] = self.cfg.build_embed_model
                try:
                    pipeline.build_graph(out_dir, **kwargs)
                except Exception:
                    # Sparse live stores can't sustain per-domain clustering;
                    # reuse the embeddings and pool states globally.
                    pipeline.build_graph(
                        out_dir, reuse_steps=True,
                        discover_args={"within_domain": False}, **kwargs)
            self.registry.activate_version(job.adapter, out_dir)
            job.status = "completed"
            job.version_path = str(out_dir)
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.finished_at = datetime.now(tz=UTC).isoformat()
            with self._lock:
                self._running = False

    def _minimal_build(self, snap: Path, out_dir: Path) -> None:
        """Placeholder build when full mining pipeline is unavailable."""
        records = []
        for line in snap.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        (out_dir / "reports").mkdir(parents=True, exist_ok=True)
        transitions = {"unknown": {"base_fail_rate": 0.2, "state_fail": []}}
        (out_dir / "reports/transitions.json").write_text(
            json.dumps(transitions), encoding="utf-8"
        )
        import numpy as np

        cent = out_dir / "out/centroids"
        cent.mkdir(parents=True, exist_ok=True)
        np.save(cent / "centroids_384.npy", np.zeros((1, 384), dtype=np.float32))
        (cent / "centroid_meta.json").write_text(
            json.dumps({"states": ["unknown:0"]}), encoding="utf-8"
        )
        (out_dir / "MANIFEST.json").write_text(
            json.dumps({"adapter": out_dir.name, "version": job_safe_version(out_dir)}),
            encoding="utf-8",
        )

    def status(self) -> dict[str, Any]:
        return {
            "jobs": [job.__dict__ for job in self._jobs],
            "finalized_rollouts": self.store.count_finalized(),
            "evolve_every_n": self.cfg.evolve_every_n,
        }


def job_safe_version(path: Path) -> str:
    return path.name
