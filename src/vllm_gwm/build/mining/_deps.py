# SPDX-License-Identifier: Apache-2.0
"""Lazy imports for the graph-mining extras.

``sentence_transformers``, ``umap-learn``, ``hdbscan`` and ``scikit-learn`` are
only needed while *building* a graph bundle, never while serving one, so they
live in the ``[build]`` extra and are imported inside the stage functions that
need them (importing :mod:`vllm_gwm.build.mining` must work with core deps
alone).
"""

from __future__ import annotations

import importlib
from types import ModuleType

_EXTRA_HINT = "pip install vllm-gwm[build]"


def lazy_import(name: str, *, extra: str = "build") -> ModuleType:
    """Import ``name`` or raise a ModuleNotFoundError naming the missing extra."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            f"the GWM graph-mining pipeline needs {name!r}, which is not "
            f"installed; install the build extra: {_EXTRA_HINT} "
            f"(missing extra {extra!r})"
        ) from exc


def sentence_transformer_cls():
    """``sentence_transformers.SentenceTransformer`` (resolved at call time).

    Resolved through the module object rather than a ``from`` import so tests
    can substitute a stub embedder on the module attribute.
    """
    return lazy_import("sentence_transformers").SentenceTransformer
