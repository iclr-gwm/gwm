# SPDX-License-Identifier: Apache-2.0
"""Graph adapter bundle validation and manifest handling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REQUIRED = (
    Path("reports/transitions.json"),
    Path("out/centroids/centroid_meta.json"),
)

# The centroid matrix is named for its embedding dimension (centroids_384.npy
# for MiniLM, centroids_4096.npy for a 4096-d embedder), so it is matched by
# shape rather than by a fixed name.
_CENTROID_GLOB = "out/centroids/centroids_*.npy"


def find_centroids(path: Path) -> Path | None:
    """The graph's centroid matrix, whatever embedding dimension built it."""
    hits = sorted(Path(path).glob(_CENTROID_GLOB))
    return hits[0] if hits else None


class AdapterError(ValueError):
    """Raised when a graph bundle is invalid or incompatible."""


@dataclass(frozen=True)
class GraphAdapter:
    """A versioned GWM graph bundle directory."""

    name: str
    path: Path
    manifest: dict[str, Any]

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def required_files(cls, path: Path) -> tuple[Path, ...]:
        cent = find_centroids(path)
        rel = cent.relative_to(Path(path).resolve()) if cent else None
        return (*_REQUIRED, rel) if rel else _REQUIRED

    @classmethod
    def validate(cls, path: Path, *, name: str = "") -> GraphAdapter:
        path = path.resolve()
        if not path.is_dir():
            raise AdapterError(f"not a graph directory: {path}")
        missing = [str(rel) for rel in _REQUIRED if not (path / rel).exists()]
        if find_centroids(path) is None:
            missing.append(_CENTROID_GLOB)
        if missing:
            raise AdapterError(f"graph {path} missing required files: {missing}")
        man_path = path / "MANIFEST.json"
        manifest: dict[str, Any] = {}
        if man_path.exists():
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
        else:
            manifest = {
                "adapter": name or path.name,
                "version": "0",
                "files": {
                    str(rel): {"sha256": cls.sha256_file(path / rel)}
                    for rel in cls.required_files(path)
                },
            }
        files = manifest.get("files") or {}
        for rel in cls.required_files(path):
            rel_s = str(rel)
            entry = files.get(rel_s) or {}
            expected = entry.get("sha256") or entry.get("md5")
            if expected and entry.get("sha256"):
                actual = cls.sha256_file(path / rel)
                if actual != expected:
                    raise AdapterError(
                        f"checksum mismatch for {rel_s} in {path}: "
                        f"expected {expected}, got {actual}"
                    )
        return cls(name=name or manifest.get("adapter", path.name), path=path, manifest=manifest)

    @classmethod
    def resolve_path(cls, ref: str, roots: list[Path]) -> Path:
        """Resolve a registered name or absolute path under allowed roots."""
        p = Path(ref)
        if p.is_absolute():
            resolved = p.resolve()
            for root in roots:
                try:
                    resolved.relative_to(root.resolve())
                    return resolved
                except ValueError:
                    continue
            raise AdapterError(f"path {ref} is outside allowed graph roots")
        for root in roots:
            cand = (root / ref).resolve()
            if cand.is_dir():
                try:
                    cand.relative_to(root.resolve())
                except ValueError:
                    continue
                return cand
        raise AdapterError(f"graph {ref!r} not found under configured roots")
