# vLLM Graph World Model (GWM) extension

Installable endpoint plugin for vLLM that loads LoRA-like **graph adapter**
bundles, gates generations through **GWM as advice+selector when K>1**,
and collects rollouts.

![GWM transition-graph viewer](docs/graph_viewer.png)

*The built-in `/v1/graph` viewer rendering a mined transition graph — states,
trap edges (red), the deepest START→END path (cyan), and per-domain filtering.
Enable it with `--gwm-show-graph`.*

## Install

This is a standalone package installed alongside a stock `vllm`. It requires no
changes to vLLM core — it plugs in through vLLM's public `vllm.endpoint_plugins`
entry point.

```bash
# from the repository root
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
# optional classifier / mining extras
.venv/bin/pip install -e ".[model,build]"
```

Install `vllm` in the same environment, then start it. vLLM discovers the plugin
through the `vllm.endpoint_plugins` entry point.

## Quick start

```bash
export VLLM_PLUGINS=gwm
vllm serve google/gemma-4-E2B-it \
  --gwm-modules eops=/path/to/graphs/eops/full \
  --gwm-modules crm=/path/to/graphs/crm/full
```

Enable GWM on a request via `vllm_xargs`:

```json
{
  "model": "google/gemma-4-E2B-it",
  "messages": [{"role": "user", "content": "..."}],
  "vllm_xargs": {
    "gwm": {
      "adapter": "eops",
      "mode": "advise",
      "k": 1,
      "episode_id": "ep-1"
    }
  }
}
```

## Advanced Features

Best-of-K select tuning is documented in
[docs/best_of_k_select_tuning.md](docs/best_of_k_select_tuning.md).

Self-evolving graph features, including accretion, state minting, periodic
rebuilds (`EvolveManager`), and the embedding/state-classifier details, are
documented in [docs/self_evolving_graph.md](docs/self_evolving_graph.md). These
features are not covered in the paper and should be treated as future work.

## Graph adapter layout

```
<adapter>/
  MANIFEST.json
  reports/transitions.json
  out/centroids/centroids_<dim>.npy
  out/centroids/centroid_meta.json
  reports/examples.json   # optional
```

## Admin routes

- `GET /v1/gwm/info`
- `GET /v1/gwm/graphs`
- `GET /v1/gwm/stats` — cumulative harness usage per loaded mediator (events, judged/skipped, unknown-state count, LLM calls/tokens, overrides, fallbacks, latency)
- `POST /v1/gwm/advise`
- `POST /v1/gwm/select`
- `POST /v1/gwm/feedback`
- `GET /v1/gwm/builds`
- `POST /v1/gwm/rollback`
- `POST /v1/chat/completions/gwm-batch`
- `GET /v1/graph` — interactive transition-graph viewer (opt-in)

Enable the viewer with `--gwm-show-graph` (or `VLLM_GWM_SHOW_GRAPH=1`). It
renders the adapters registered via `--gwm-modules`:

```bash
vllm-gwm serve --gwm-show-graph --gwm-modules crm=$(pwd)/graphs/crm/full MODEL
# then open http://127.0.0.1:8000/v1/graph
```

## Tests

```bash
pytest -m "not model" -q
```

GPU integration smokes (physical slot 1):

```bash
CUDA_VISIBLE_DEVICES=1 pytest -m gpu -q
```

## Live three-mode evaluation

See [docs/three_mode_live.md](docs/three_mode_live.md) for running the frozen,
build-from-ratio, and self-evolve modes against a live server, with reference
results.

## Provenance

See [SYNC.md](SYNC.md) for vendored harness sources.

## API backend (no local vLLM)

Serve GWM against any OpenAI-compatible API — no GPU engine required:

```bash
pip install -e ".[server,model]"

export OPENAI_API_KEY=...
vllm-gwm serve --backend api \
  --api-base https://api.openai.com/v1 \
  --model gpt-4o \
  --gwm-modules crm=$(pwd)/graphs/crm/full
```

Key flags: `--backend api`, `--api-base`, `--model`, `--judge-api-base`,
`VLLM_GWM_API_MAX_TOKENS`, `VLLM_GWM_API_MAX_RETRIES`,
`VLLM_GWM_API_RETRY_STEP_SEC`, `VLLM_GWM_API_QUOTA_SLOW_PERCENT`.

Notes on OpenAI-compatible backends:

- Backends without native `n` are emulated by `OpenAIAPI` looping.
- If the backend does not honour `temperature`/`top_p`/`seed`, the
  `greedy_anchor` best-of-K arm is effectively inert (`select_min_margin` still
  applies) and best-of-K may see identical candidates on most steps.
- `response_format`, `stop`, `seed`, `user`, `tools` and `tool_choice` are
  forwarded verbatim from the request.
- A low `max_tokens` cap can truncate verbose agent replies mid-JSON; raise
  `VLLM_GWM_API_MAX_TOKENS` for long answers.

## Examples

See [examples.md](examples.md) for runnable scripts, **Toucan live logs** (GPU 1
`:8497`, Gemma4 / Qwen3.6), curl samples, and env reference.

## Implementation

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for how GWM discovers workflows,
the graph adapter format, request lifecycle flowcharts, and the
collect → label → evolve loop.
