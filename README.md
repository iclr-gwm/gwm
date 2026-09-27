# vLLM Graph World Model (GWM) extension

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-d7af51?style=flat)](LICENSE)
[![Project Website: live](https://img.shields.io/badge/%F0%9F%8C%90_Project_Website-live-26775e?style=flat&labelColor=172b26)](https://iclr-gwm.github.io/gwm/)

> [!NOTE]
> **[Explore the GWM project website →](https://iclr-gwm.github.io/gwm/)**
>
> Interactive animations, paper highlights, benchmark results, and the vLLM + GWM walkthrough.

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

**Install with one command** from the cloned repository in a Python 3.11+
environment supported by vLLM:

```bash
python -m pip install -e ".[server,model,build]" vllm
```

## Quick start

**Build an adapter and start the inference endpoint with one command.**
Supply a corpus of labelled conversations as `rollouts.jsonl` and replace
`MODEL_ID` with your serving model:

```bash
vllm-gwm build --input rollouts.jsonl --adapter agent --output graphs/agent && \
VLLM_PLUGINS=gwm vllm-gwm serve --gwm-modules agent=graphs/agent MODEL_ID
```

Build input uses `conversation_flow` events with explicit `overall_success`
labels, or the plugin's collected-rollout format. The
[rollout format example](docs/assets/examples/agent-rollout.example.jsonl)
illustrates one record; use a corpus of successful and failed conversations
for graph discovery.

**Continue a multi-turn conversation in one request.** Save the
[request example](docs/assets/examples/agent-request.json) as `request.json`,
set `MODEL_ID` and an installed `YOUR_PRESET` matching your agent's tools and
action format, then send it to the running server:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @request.json
```

The example includes a prior assistant tool call, its result, and a follow-up
user turn; it requests a text continuation from that history with K=2.
The HTTP settings use flat `gwm.*` keys. Your agent retains its tool-execution
loop: send history, receive one selected continuation, execute any returned
tool action, append the result, and repeat. Use your model's usual chat
template and tool-parser options when serving native tool calls.
See the [implementation guide](IMPLEMENTATION.md) and [examples](examples.md)
for preset configuration and agent integration.

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

## License

[MIT](LICENSE).
