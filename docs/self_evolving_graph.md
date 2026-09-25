# Self-Evolving Graph Features

Note: self-evolving graph features are not covered in the paper. Treat this
material as an experimental future-work extension.

## Self-Evolving Graph (Accretion + State Minting)

The graph is normally mined offline and read-only at serve time. These flags let
it grow **in place** from the policy's own live rollouts: no re-clustering, no
directory swap, and no outcome/reward channel.

| Env | Default | Effect |
|---|---|---|
| `VLLM_GWM_ACCRETE_SELF` | `0` | Prepend the judge-selected action at the classified state to that state's exemplar list, so later steps at the same state see the policy's own recent behaviour as evidence. Applied *after* the decision, so a step never influences itself. |
| `VLLM_GWM_MINT_STATES` | `0` | On abstain, greedy leader-cluster the embedding against previously minted centroids: attach if within radius, else append a new centroid. Converts recurring UNKNOWN states into classifiable ones. |
| `VLLM_GWM_ACCRETE_MAX_PER_STATE` | `8` | Ring-buffer cap on **accreted** entries per state. Mined exemplars are never evicted. |
| `VLLM_GWM_MINT_MIN_SUPPORT` | `3` | A minted state stays invisible to the classifier until it has been visited this many times, so a one-off state cannot speak. |
| `VLLM_GWM_MINT_RADIUS` | `0` | Attach radius for minted centroids; `0` uses the graph's own median P90, so a minted state is no more permissive than a mined one. |
| `VLLM_GWM_ACCRETE_SCOPE` | `global` | `global` carries evidence across episodes (**transductive**; later tasks benefit from earlier ones, so report it as such). `episode` purges all accreted evidence between episodes (inductive control). |

Counters are exposed on `GET /v1/gwm/info` under `accrete.stats` and, with
`VLLM_GWM_AUDIT_LOG=1`, snapshotted to `<data_root>/accrete/<adapter>.json`.

**No outcome signal is consumed, deliberately.** On a teacher-forced replay
benchmark (Toucan) the only correctness signal available at eval time is the
gold action, and accreting on it would be training on test labels. Accretion is
therefore self-referential: it changes *what evidence the judge sees*, not what
it is told is correct. On a genuinely executing benchmark, outcome-gated
accretion is the natural extension.

## Periodic Rebuild (`EvolveManager`)

`VLLM_GWM_EVOLVE_EVERY=N` triggers a background rebuild after N finalized
rollouts, then `activate_version()` swaps the bundle atomically. The rebuild
runs the real mining DAG in-process (`vllm_gwm.build.pipeline.build_graph`:
prefix_expand -> embed -> concat -> discover -> mine -> precheck -> examples),
so it needs the `[build]` extra (`umap-learn`, `hdbscan`, `scikit-learn`,
`sentence-transformers`). A failing build **fails closed**: the previously
active bundle stays active; the 1-state placeholder
(`VLLM_GWM_EVOLVE_ALLOW_MINIMAL=1`) is opt-in for tests only.

| Env | Default | Effect |
|---|---|---|
| `VLLM_GWM_BUILD_EMBED_MODEL` | pipeline default (MiniLM) | embedder for evolve builds |
| `VLLM_GWM_BUILD_EMBED_DEVICE` | `cpu` | the GPU is busy serving |
| `VLLM_GWM_BUILD_WATCHED_TOOLS` | - | tools the precheck gates track |

The same pipeline backs `vllm-gwm build --input rollouts.jsonl --adapter <name>`
and `scripts/build_graph.sh` (a thin wrapper over
`python -m vllm_gwm.build.pipeline`).

## Embedding Model (State Classifier)

The serve-time classifier embeds the live flow and takes the cosine-nearest
centroid, so **it must use the same embedder that built the graph**. That model
id is recorded in the bundle's `centroid_meta.json` (`model`, `dim`) and is
loaded from there; you do not configure it per-request. The centroid matrix is
named for its dimension (`centroids_384.npy` for MiniLM, `centroids_4096.npy`
for a 4096-d embedder) and is discovered by glob, so any dimension works.

| Env | Default | Effect |
|---|---|---|
| `VLLM_GWM_EMBED_DEVICE` | `cpu` | Device for the embedder. MiniLM is fine on CPU; a multi-billion-parameter embedder is not usable there at per-step latency, so point it at a GPU (`cuda:0`) and lower `--gpu-memory-utilization` to leave room beside the policy weights. |
| `VLLM_GWM_EMBED_TRUST_REMOTE_CODE` | `0` | Pass `trust_remote_code=True` when loading. Required by embedders that ship custom modelling code; opt-in by design. |

A mismatch between the embedder's output dimension and the graph's centroids
raises at load rather than silently classifying in the wrong space.
