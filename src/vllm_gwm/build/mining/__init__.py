# SPDX-License-Identifier: Apache-2.0
"""Graph-mining stages (ported from the benchmarks ``wm/`` build driver).

One module per stage of the build DAG; every stage is a plain function taking
explicit paths + config and returning a small result dict (no module-level env
globals, no argparse ``main()``). :mod:`vllm_gwm.build.pipeline` chains them.

    1 prefix_expand  rollouts{,_test}.jsonl.gz -> steps{,_test}.jsonl.gz
    2 embed          steps{,_test}             -> emb_st{,_test}.npz (+ meta/config)
    3 concat         emb shards                -> emb_st_all.npz (+ meta)
    4 discover       emb_st_all                -> states_all.jsonl.gz (UMAP+HDBSCAN)
    5 mine           states_all                -> reports/transitions.json
    6 precheck       states_all + emb          -> out/centroids/ + reports/PRECHECK.md
    7 examples       states_all + steps        -> reports/examples.json

Stage 0 (harvest: benchmark trajectories -> rollouts.jsonl.gz) is
benchmark-specific and lives outside the plugin.

Importing this package only needs the core deps; ``sentence_transformers``,
``umap-learn``, ``hdbscan`` and ``scikit-learn`` are imported inside the stages
that need them (``pip install vllm-gwm[build]``).
"""

from vllm_gwm.build.mining.discover import discover_states, umap_hdbscan_labels
from vllm_gwm.build.mining.embed import concat_embeddings, embed_steps
from vllm_gwm.build.mining.examples import harvest_examples
from vllm_gwm.build.mining.mine import mine_workflow_graph
from vllm_gwm.build.mining.precheck import precheck
from vllm_gwm.build.mining.prefix_expand import expand_steps, split_by_tag, step_records

__all__ = [
    "concat_embeddings",
    "discover_states",
    "embed_steps",
    "expand_steps",
    "harvest_examples",
    "mine_workflow_graph",
    "precheck",
    "split_by_tag",
    "step_records",
    "umap_hdbscan_labels",
]
