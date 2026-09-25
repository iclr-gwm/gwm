# SPDX-License-Identifier: Apache-2.0
"""Stage 2/3 — embed prefix samples with a sentence-transformer, then concatenate.

Design note — embed the TAIL, not the whole prefix. State discovery wants "where
in the workflow am I", which lives in the *recent* action + observation. The head
(system prompt, original user task) is near-identical across every step of a task,
so embedding the whole prefix collapses a task's steps together. We embed the last
``tail_chars`` by default; switch with ``window={"tail","head","full"}``.

Outputs per embed call:

- ``<out>.npz`` with ``X`` (float32 ``[N, d]``, L2-normalized),
- ``<out>.meta.jsonl`` — one aligned metadata row per embedding,
- ``<out>.config.json`` — the embedding config (model, dim, window, tail_chars,
  normalisation, device, ...). :mod:`vllm_gwm.build.mining.precheck` folds this
  sidecar into the bundle so a bundle can never be paired with a different
  embedder by accident.

Ported from ``benchmarks/wm/lib/embed_steps_st.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from vllm_gwm.build.mining._deps import lazy_import, sentence_transformer_cls
from vllm_gwm.build.mining.prefix_expand import read_jsonl

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TAIL_CHARS = 3000
DEFAULT_BATCH_SIZE = 256

_META_KEYS = (
    "step_uid", "uid", "task_id", "domain", "split", "temp",
    "step_idx", "n_steps", "tool_names", "obs_outcome",
    "overall_success", "pass_rate",
)


def window_text(text: str, window: str, n: int) -> str:
    if window == "full" or len(text) <= n:
        return text
    if window == "head":
        return text[:n]
    return text[-n:]  # tail (default)


def model_slug(model_id: str) -> str:
    """Filesystem-safe short name for an embedder, for artifact directories."""
    tail = str(model_id).rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-")


def file_md5(path: str | Path) -> str | None:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def embed_config(
    model: Any,
    X: np.ndarray,
    n_rows: int,
    *,
    model_id: str,
    window: str,
    tail_chars: int,
    batch_size: int,
    device: str | None,
    trust_remote_code: bool,
    steps_path: str | Path,
) -> dict[str, Any]:
    """The reproducibility record for one embed run."""
    _st = lazy_import("sentence_transformers")

    dim = None
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        fn = getattr(model, attr, None)
        if callable(fn):
            dim = fn()
            break
    if dim is None:
        dim = X.shape[1] if X.ndim == 2 else 0
    return {
        "model": model_id,
        "model_slug": model_slug(model_id),
        "dim": int(dim),
        "window": window,
        "tail_chars": int(tail_chars),
        "normalize_embeddings": True,
        "batch_size": int(batch_size),
        "device": str(device),
        "trust_remote_code": bool(trust_remote_code),
        "max_seq_length": getattr(model, "max_seq_length", None),
        "sentence_transformers_version": getattr(_st, "__version__", None),
        "n_steps": int(n_rows),
        "input": str(steps_path),
        "input_md5": file_md5(steps_path),
    }


def embed_steps(
    steps_path: str | Path,
    out_path: str | Path,
    *,
    model: str = DEFAULT_EMBED_MODEL,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    window: str = "tail",
    tail_chars: int = DEFAULT_TAIL_CHARS,
    trust_remote_code: bool = False,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Embed ``steps_path`` -> ``out_path`` (+ ``.meta.jsonl`` / ``.config.json``).

    An empty step file (e.g. a train/valid-only harvest with no test rollouts)
    writes an empty ``X`` and no config sidecar rather than loading the model:
    stage 3 reshapes empty splits to the train dimension.
    """
    if window not in ("tail", "head", "full"):
        raise ValueError(f"window must be tail|head|full, got {window!r}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(steps_path))
    texts = [window_text(r.get("text", ""), window, tail_chars) for r in rows]
    print(f"[embed_st] {len(rows)} steps; model={model}; window={window}/{tail_chars}")

    meta_path = out.with_suffix(".meta.jsonl")
    with meta_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in _META_KEYS}, ensure_ascii=False) + "\n")

    if not rows:
        X = np.zeros((0,), dtype=np.float32)
        np.savez(out, X=X)
        print(f"[embed_st] empty input; wrote X={X.shape} -> {out} (no config sidecar)")
        return {"n": 0, "dim": 0, "out": str(out), "meta": str(meta_path), "config": None}

    st_kwargs: dict[str, Any] = {"device": device}
    if trust_remote_code:
        st_kwargs["trust_remote_code"] = True
    encoder = sentence_transformer_cls()(model, **st_kwargs)
    dim_fn = getattr(encoder, "get_sentence_embedding_dimension", None)
    print(f"[embed] model={model} dim={dim_fn() if callable(dim_fn) else '?'} "
          f"device={device}", flush=True)
    X = encoder.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=show_progress_bar,
    ).astype(np.float32)

    np.savez(out, X=X)
    cfg = embed_config(
        encoder, X, len(rows),
        model_id=model, window=window, tail_chars=tail_chars,
        batch_size=batch_size, device=device,
        trust_remote_code=trust_remote_code, steps_path=steps_path,
    )
    cfg_path = out.with_suffix(".config.json")
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[embed_st] wrote X={X.shape} -> {out}")
    print(f"[embed_st] wrote meta -> {meta_path}")
    print(f"[embed_st] wrote config -> {cfg_path}")
    return {
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]) if X.ndim == 2 else 0,
        "out": str(out),
        "meta": str(meta_path),
        "config": str(cfg_path),
        "embedding": cfg,
    }


def concat_embeddings(
    inputs: list[str | Path],
    out_path: str | Path,
) -> dict[str, Any]:
    """Stage 3 — concatenate embed shards (train + test) into one matrix + meta.

    Empty shards (``X.ndim != 2``) are reshaped to the dimension of the first
    non-empty shard, so a train/valid-only build concatenates cleanly.
    """
    parts = [Path(p) for p in inputs]
    mats = [np.load(p)["X"] for p in parts]
    dim = next((m.shape[1] for m in mats if m.ndim == 2), 0)
    if not dim:
        raise ValueError(f"no non-empty embedding shard among {[str(p) for p in parts]}")
    mats = [m if m.ndim == 2 else m.reshape(0, dim) for m in mats]
    X = np.concatenate(mats, axis=0)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X)
    meta_out = out.with_suffix(".meta.jsonl")
    with meta_out.open("w", encoding="utf-8") as o:
        for p in parts:
            meta = p.with_suffix(".meta.jsonl")
            if meta.exists():
                o.write(meta.read_text(encoding="utf-8"))
    print(f"[concat] {out.name}: " + " + ".join(str(m.shape[0]) for m in mats)
          + f" = {X.shape[0]}")
    return {"n": int(X.shape[0]), "dim": int(dim), "out": str(out), "meta": str(meta_out)}
