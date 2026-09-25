# vLLM GWM — examples

Runnable scripts and request samples for the Graph World Model extension.
All paths below are relative to the repository root.

Sample outputs below were captured on this host (2026-09-04, Toucan live 2026-09-05/06). Offline advise/select
JSON is from FakeScorer + MockVLLM on the CRM retry fixture. Chat-completion
`content` varies with the served model; GWM still returns a normal OpenAI envelope
with a **single** winning choice.

## One-time setup

```bash
bash scripts/build_env.sh
```

Installs `.venv` with `.[test,model,server]`, registers the `vllm.endpoint_plugins`
entry point, and syncs CRM/EOPS graph bundles (without large `examples.json` files).

Override extras:

```bash
VLLM_GWM_INSTALL_EXTRAS=test bash scripts/build_env.sh
```

Sample (tail):

```text
synced crm/full -> .../graphs/crm/full
synced eops/full -> .../graphs/eops/full
python: Python 3.12.3
vllm_gwm 0.1.0
entrypoint: vllm_gwm.plugin:GwmEndpointPlugin
build_env: OK
```

---

## Offline tests (no served LLM)

| Script | Purpose |
|--------|---------|
| `bash examples/smoke_cpu.sh` | Validate tiny graph + run CPU pytest |
| `bash scripts/run_full_test.sh` | Full matrix: pytest + fake e2e + MiniLM e2e on GPU slot 1 |
| `.venv/bin/python scripts/e2e_benchmark.py` | Advise/select latency report (FakeScorer + MockVLLM) |
| `.venv/bin/python scripts/e2e_benchmark.py --model` | Same, with real MiniLM classifier |

### `bash examples/smoke_cpu.sh`

```text
Smoke: validate tiny graph adapter
valid tiny_graph .../tests/fixtures/tiny_graph
Run CPU tests
...........                                                              [100%]
11 passed in 0.09s
```

### `.venv/bin/python scripts/e2e_benchmark.py`

```text
=== advise/crm ===
latency_ms=17 injected=True state=toy:0
harness: llm_calls=1 tokens=1153 latency_ms=16
advice block:
[World-Model guidance]
Stop repeating the failing call; read the last error and change approach — re-check the identifier/table before retrying (mock vLLM).

=== advise/eops ===
latency_ms=1 injected=True state=toy:0
harness: llm_calls=1 tokens=1252 latency_ms=1
advice block:
[World-Model guidance]
Stop repeating the failing call; read the last error and change approach — re-check the identifier/table before retrying (mock vLLM).

=== select/k=4 (joint) ===
latency_ms=1 scores=[0.8, 1.0, 0.9, 0.7] winner=1

TOTAL e2e_ms=20
```

Reports written under `.gwm_data/`:

- `e2e_report.json` — offline fake scorer
- `e2e_report_model.json` — MiniLM on `CUDA_VISIBLE_DEVICES=1`
- `pytest_cpu.log`

Sample `.gwm_data/e2e_report.json` (abbreviated):

```json
{
  "mode": "fake_scorer",
  "total_ms": 21,
  "cases": [
    {
      "preset": "crm",
      "latency_ms": 16,
      "state": "toy:0",
      "injected": true,
      "success_score": 0.1,
      "harness_llm_calls": 1,
      "harness_tokens": 1153,
      "advice_preview": "[World-Model guidance]\nStop repeating the failing call; ..."
    }
  ],
  "select_k4": {
    "latency_ms": 1,
    "scores": [0.8, 1.0, 0.9, 0.7],
    "winner_index": 1,
    "score_via": ["harness_joint", "harness_joint", "harness_joint", "harness_joint"]
  }
}
```

MiniLM (`--model` on GPU slot 1) is the same shape; first CRM advise includes
classifier cold-load (~5.5 s), later cases ~65–100 ms, `total_ms` ~5661.

GPU slot check:

```bash
CUDA_VISIBLE_DEVICES=1 bash examples/smoke_gpu_slot1.sh
```

```text
gpu smoke ok on NVIDIA H200 NVL
GPU slot 1 smoke passed (cuda:0 inside process)
```

---

## Live vLLM integration

Requires a Python environment with **both** `vllm` and `vllm-gwm` installed.

### Launch server + run live e2e

```bash
# default: Gemma4 on GPU slot 1, port 8765
bash scripts/live_smoke.sh

# first-time vLLM install into extension venv (slow)
bash scripts/live_smoke.sh --install-vllm

# custom model / port
VLLM_GWM_LIVE_MODEL=google/gemma-4-E2B-it \
VLLM_GWM_LIVE_PORT=8765 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/live_smoke.sh
```

Sample (after the server is ready):

```text
== vllm-gwm live_smoke ==
model=google/gemma-4-E2B-it port=8765 gpu=1 attach=0
...
starting vLLM on :8765 (GPU 1) ...
vLLM log: .../.gwm_data/vllm_live.log
running live_e2e.py ...
waiting for http://127.0.0.1:8765/v1 ...
gwm/info ok presets=['crm', 'eops'] graphs=3
gwm/graphs ok loaded=[...]

=== admin/advise/crm ===
latency_ms=... state=... injected=...
[World-Model guidance]
...

=== chat/gwm-advise ===
latency_ms=...
<assistant text from the served model>

=== chat/gwm-select k=4 ===
latency_ms=...
<winning candidate only>

TOTAL live_e2e_ms≈...
live_smoke: OK
report: .gwm_data/live_e2e_report.json
```

Report: `.gwm_data/live_e2e_report.json`

Server log: `.gwm_data/vllm_live.log`

### Attach to an already-running vLLM

```bash
export VLLM_PLUGINS=gwm
export VLLM_GWM_MODULES="tiny=$(pwd)/tests/fixtures/tiny_graph,crm=$(pwd)/graphs/crm/full"
export VLLM_BASE_URL=http://127.0.0.1:8000/v1

bash scripts/live_smoke.sh --attach
# or directly:
.venv/bin/python scripts/live_e2e.py --base-url "$VLLM_BASE_URL" --model google/gemma-4-E2B-it
```

---

## Graph assets

```bash
# sync CRM/EOPS bundles from standalone gwm repo (skips 188MB examples.json)
bash examples/fetch_graphs.sh graphs/

bash examples/smoke_advise.sh   # prints example vllm serve command
```

```text
synced crm/full -> graphs/crm/full
synced eops/full -> graphs/eops/full
Example: vllm serve MODEL --gwm-modules eops=.../graphs/eops/full --gwm-modules crm=.../graphs/crm/full
```

---

## Serve with GWM manually

```bash
export VLLM_PLUGINS=gwm
export VLLM_GWM_MODULES="eops=$(pwd)/graphs/eops/full,crm=$(pwd)/graphs/crm/full,toucan=/path/to/gwm_artifacts/toucan/full"
export VLLM_GWM_DATA_ROOT=$(pwd)/.gwm_data
export VLLM_GWM_PRESET_DEFAULT=toucan   # optional; per-request gwm.preset still wins

vllm serve google/gemma-4-E2B-it --host 127.0.0.1 --port 8765
```

Or via wrapper CLI:

```bash
vllm-gwm serve --gwm-modules eops=$(pwd)/graphs/eops/full google/gemma-4-E2B-it --port 8765
```

Transition-graph viewer (`GET /v1/graph`):

```bash
vllm-gwm serve --gwm-show-graph \
  --gwm-modules eops=$(pwd)/graphs/eops/full \
  --gwm-modules crm=$(pwd)/graphs/crm/full \
  google/gemma-4-E2B-it --port 8765
```

Then open `http://127.0.0.1:8765/v1/graph`. Equivalent env: `VLLM_GWM_SHOW_GRAPH=1`.

```bash
curl -s http://127.0.0.1:8765/v1/graph/api/catalog
curl -s http://127.0.0.1:8765/v1/graph/api/graph/crm
```

---

## Toucan live (GPU slot 1, 2026-09-05/06)

Captured against the GWM plugin on **physical GPU 1**, port **8497**, while the
benchmark harness ran Toucan `llm` + in-vLLM GWM (`select` k=4). No wm-harness
sidecar. Graph remined 2026-09-05 (`train_rollouts=1800`, PRECHECK GO, 42 states).

Attach-only smoke (does **not** start another vLLM):

```bash
export VLLM_BASE_URL=http://127.0.0.1:8497/v1
bash examples/smoke_toucan.sh
```

### Serve (from benchmarks helper, equivalent)

```bash
# GPU 1 only. vllm-gwm serve maps positional MODEL -> --model for api_server.
export CUDA_VISIBLE_DEVICES=1
export VLLM_PLUGINS=gwm
export VLLM_GWM_MODULES="toucan=$HOME/repo/gwm_artifacts/toucan/full"
export VLLM_GWM_PRESET_DEFAULT=toucan
export VLLM_GWM_SHOW_GRAPH=1

python -m vllm_gwm.cli serve --gwm-show-graph \
  google/gemma-4-31b-it \
  --served-model-name gemma4-31b-it \
  --host 127.0.0.1 --port 8497 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --language-model-only --enable-prefix-caching
```

Switch to Qwen on the **same slot** after stopping the previous process:

```bash
python -m vllm_gwm.cli serve --gwm-show-graph \
  Qwen/Qwen3.6-27B \
  --served-model-name qwen3.6-27b \
  --host 127.0.0.1 --port 8497 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

**Preflight (required):** `api_server` does not map `model_tag` → `model`. If the
CLI does not prepend `--model`, vLLM loads default `Qwen/Qwen3-0.6B` while still
advertising `gemma4-31b-it`. Toucan exact-match then collapses (~22/100).

Serve log (abbreviated, 2026-09-05 20:00 PDT):

```text
(APIServer) INFO ... model   google/gemma-4-31b-it
(APIServer) INFO ... - gwm -> vllm_gwm.plugin:GwmEndpointPlugin
(APIServer) INFO ... Loaded endpoint plugin gwm
(APIServer) INFO ... Route: /v1/gwm/info, Methods: GET
(APIServer) INFO ... Route: /v1/gwm/advise, Methods: POST
(APIServer) INFO ... Route: /v1/gwm/select, Methods: POST
(EngineCore) INFO ... Model loading took 57.91 GiB ...   # Gemma; Qwen3.6 is ~50 GiB
```

Driver log after reload for 20K diversified generation:

```text
[2026-09-06T03:00:53Z] ok root=google/gemma-4-31b-it gwm adapter=toucan mode=select k=4
[2026-09-06T03:00:53Z] generate model=gemma4 split=train shard=0 pool=20000
... Loaded 9000 tasks (target=toucan_train) ...
```

```bash
curl -s http://127.0.0.1:8497/v1/models | jq '{id: .data[0].id, root: .data[0].root}'
```

```json
{
  "id": "gemma4-31b-it",
  "root": "google/gemma-4-31b-it"
}
```

### `GET /v1/gwm/info` (live)

```json
{
  "server": "vllm_gwm",
  "version": "0.1.0",
  "default_preset": "toucan",
  "presets": ["crm", "eops", "toucan"],
  "graphs": [
    {
      "name": "toucan",
      "path": "$HOME/repo/gwm_artifacts/toucan/full",
      "loaded": true,
      "valid": true,
      "manifest": {
        "bench": "toucan",
        "ratio_tag": "full",
        "ratio_seed": 42,
        "train_rollouts": 1800,
        "built_utc": "2026-09-05T08:16:43Z"
      }
    }
  ],
  "collect_mode": "opt_in",
  "evolve_every_n": 0,
  "show_graph": true,
  "graph_endpoint": "/v1/graph"
}
```

Viewer catalog (`GET /v1/graph/api/catalog`):

```json
{
  "graphs": [
    {
      "id": "toucan",
      "label": "toucan",
      "available": true,
      "source": "$HOME/repo/gwm_artifacts/toucan/full/reports/transitions.json"
    }
  ],
  "default": "toucan",
  "api_base": "/v1/graph"
}
```

```bash
vllm-gwm inspect $HOME/repo/gwm_artifacts/toucan/full
```

```text
GraphAdapter(name='full', path=PosixPath('.../gwm_artifacts/toucan/full'),
  manifest={'bench': 'toucan', 'ratio_tag': 'full', 'train_rollouts': 1800, ...})
```

### Chat: select k=4 (flat `gwm.*` xargs — required on HTTP)

Toucan purple `llm` executor sends this via LiteLLM `extra_body` (see
`benchmarks/assets/toucan/purple-executors/llm/gwm_xargs.py`). Env:
`TOUCAN_GWM_ADAPTER=toucan TOUCAN_GWM_MODE=select TOUCAN_GWM_K=4`.

```bash
curl -s http://127.0.0.1:8497/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-31b-it",
    "messages": [{"role":"user","content":"Task / user question:\nWhat is the weather in Paris?\n\nAvailable tools: ..."}],
    "temperature": 0.7,
    "max_tokens": 256,
    "vllm_xargs": {
      "gwm.adapter": "toucan",
      "gwm.preset": "toucan",
      "gwm.mode": "select",
      "gwm.k": 4,
      "gwm.episode_id": "toucan-task-001"
    }
  }'
```

Preset `flow_style = "toucan"` strips the catalog blob and classifies a
harvest-shaped flow (bare question + JSON tool actions). Response is a normal
OpenAI envelope with **one** `choices[0]` (the ranked winner).

Advise (k=1): same URL, `"gwm.mode": "advise", "gwm.k": 1`.

### Admin advise on a harvest-shaped flow (live)

First user turn often **abstains** (`unify_domains` + trigger skip) until a tool
action exists. Captured 2026-09-06 against the Gemma+Toucan server above
(`latency_ms=92`):

```bash
curl -s http://127.0.0.1:8497/v1/gwm/advise \
  -H 'Content-Type: application/json' \
  -d '{
    "conversation_flow": [
      {"type":"user_message","content":"What is the weather in Paris?"}
    ],
    "preset": "toucan",
    "adapter": "toucan",
    "episode_id": "docs-demo-1"
  }'
```

```json
{
  "state": "UNKNOWN",
  "cos_dist": 0.4948,
  "abstain": true,
  "is_trap": false,
  "last_tool": "",
  "injected": false,
  "block": "",
  "success_score": 0.4285714285714286,
  "probability": 1.0,
  "harness": {
    "triggered": false,
    "retry_loop": {"detected": false},
    "tool_calls": [],
    "llm_calls": 0,
    "tokens": 0,
    "latency_ms": 92,
    "skip": "trigger"
  }
}
```

### Benchmarks driver (eval + 20K gen)

From the **benchmarks** repo (not this extension). Eval on `toucan_100_test`
after remine; generation uses disjoint 10K shards (Gemma then Qwen on GPU 1).

```bash
# eval: in-vLLM GWM select k=4 (no sidecar)
export OPENAI_API_BASE=http://127.0.0.1:8497/v1
export TOUCAN_AGENT_LLM=openai/gemma4-31b-it
export TOUCAN_GWM_ADAPTER=toucan TOUCAN_GWM_PRESET=toucan
export TOUCAN_GWM_MODE=select TOUCAN_GWM_K=4
export TOUCAN_LLM_TEMPERATURE=0.7
./mas bench run toucan --executor llm --target toucan_100_test

# ~20K trajectories, vLLM GWM only, GPU 1, Gemma shard 0 then Qwen shard 1
bash assets/toucan/scripts/run_20k_vllm_gwm_gen.sh
```

Held-out exact-match (`toucan_100_test`, 1 run, remined graph, GPU 1):

| Arm | Qwen3.6-27B | Gemma4-31B-IT |
|-----|------------:|--------------:|
| greedy (no `gwm.*`) | 59/100 | 58/100 |
| in-vLLM `select` k=4 | 55/100 | 58/100 |
| in-vLLM `select` k=8 | 58/100 | 58/100 |
| in-vLLM `advise` | 57/100 | 58/100 |

---

## HTTP request examples

Pipe curl through `jq` if you want pretty JSON. A first-turn user message with no
tool failure often returns `injected: false` and an empty `block`. The advise
sample below is the **trap / retry** case (CRM retry fixture).

### Admin: advise on a trajectory

```bash
curl -s http://127.0.0.1:8765/v1/gwm/advise \
  -H 'Content-Type: application/json' \
  -d @- <<'EOF'
{
  "conversation_flow": [{"type":"user_message","content":"Find Acme owner"}],
  "preset": "crm",
  "adapter": "crm",
  "episode_id": "demo-1"
}
EOF
```

Sample (`POST /v1/gwm/advise`, trap/retry trajectory):

```json
{
  "state": "toy:0",
  "cos_dist": 0.1,
  "abstain": false,
  "is_trap": true,
  "last_tool": "execute",
  "injected": true,
  "block": "[World-Model guidance]\nStop repeating the failing call; read the last error and change approach — re-check the identifier/table before retrying (mock vLLM).",
  "success_score": 0.1,
  "probability": 1.0,
  "harness": {
    "triggered": true,
    "retry_loop": {
      "detected": true,
      "kind": "consecutive_failures",
      "tool": "execute",
      "count": 3
    },
    "tool_calls": [],
    "llm_calls": 1,
    "tokens": 1153,
    "latency_ms": 16
  }
}
```

### Chat completion with GWM advise (k=1)

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [{"role":"user","content":"What is your next step?"}],
    "max_tokens": 128,
    "vllm_xargs": {
      "gwm": {
        "adapter": "eops",
        "mode": "advise",
        "k": 1,
        "episode_id": "ep-1"
      }
    }
  }'
```

Sample (OpenAI envelope; GWM retries internally up to `VLLM_GWM_MAX_ATTEMPTS`,
then returns the approved or best candidate). MockVLLM policy text:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "google/gemma-4-E2B-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I will stop repeating the failing call and instead re-read the relevant record / inspect the schema, then issue a corrected call."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 28,
    "total_tokens": 60
  }
}
```

### Chat completion with GWM select (k=4)

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [{"role":"user","content":"Plan the next action."}],
    "max_tokens": 128,
    "temperature": 0.7,
    "vllm_xargs": {
      "gwm": {
        "adapter": "eops",
        "mode": "select",
        "k": 4,
        "episode_id": "ep-2"
      }
    }
  }'
```

Sample: the server generates `n=4` candidates, scores them jointly
(`[0.8, 1.0, 0.9, 0.7]` → winner index **1**), and returns **only** that choice
(same envelope as advise, still `choices.length == 1`).

Admin `POST /v1/gwm/select` body uses `conversation_flows` (a list of candidate
trajectories) and returns:

```json
{
  "results": [
    {"state": "toy:0", "success_score": 0.8, "score_via": "harness_joint", "injected": false},
    {"state": "toy:0", "success_score": 1.0, "score_via": "harness_joint", "injected": false},
    {"state": "toy:0", "success_score": 0.9, "score_via": "harness_joint", "injected": false},
    {"state": "toy:0", "success_score": 0.7, "score_via": "harness_joint", "injected": false}
  ]
}
```

### Explicit rollout feedback (hybrid labeling)

```bash
curl -s http://127.0.0.1:8765/v1/gwm/feedback \
  -H 'Content-Type: application/json' \
  -d '{"episode_id":"ep-1","success":true}'
```

```json
{"ok": true}
```

### Build graph from collected rollouts

```bash
vllm-gwm build --input .gwm_data/rollouts/rollouts.jsonl --adapter demo --output graphs/demo/build
```

```text
built graphs/demo/build
```

---

## Environment reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_PLUGINS` | — | Must include `gwm` |
| `VLLM_GWM_MODULES` | — | `name=path,name=path` graph adapters |
| `VLLM_GWM_DATA_ROOT` | `.gwm_data` | rollouts, logs, builds |
| `VLLM_GWM_PRESET_DEFAULT` | `crm` | default preset (`crm`, `eops`, `toucan`) |
| `VLLM_GWM_EVOLVE_EVERY` | `0` | self-evolve after N finalized rollouts |
| `VLLM_GWM_MAX_ATTEMPTS` | `3` | advise retry cap |
| `VLLM_GWM_SHOW_GRAPH` | `0` | serve transition-graph viewer at `GET /v1/graph` |
| `CUDA_VISIBLE_DEVICES` | — | use `1` for physical GPU slot 1 |

---

## CLI subcommands

```bash
vllm-gwm serve --gwm-modules tiny=tests/fixtures/tiny_graph MODEL ...
vllm-gwm serve --gwm-show-graph --gwm-modules tiny=tests/fixtures/tiny_graph MODEL ...
vllm-gwm build --input rollouts.jsonl --adapter NAME
vllm-gwm inspect graphs/crm/full
vllm-gwm import /path/to/bundle --name crm
vllm-gwm rollback --adapter crm --version-path graphs/crm/build/abc123
```

### `vllm-gwm inspect graphs/crm/full`

```text
GraphAdapter(name='full', path=PosixPath('.../graphs/crm/full'), manifest={'bench': 'crm', 'ratio_tag': 'full', 'ratio_seed': 42, 'train_rollouts': 14660, ...})
```

### `vllm-gwm inspect tests/fixtures/tiny_graph`

```text
GraphAdapter(name='tiny_graph', path=PosixPath('.../tests/fixtures/tiny_graph'), manifest={'name': 'tiny_graph', 'domains': ['toy'], 'n_states': 2, 'n_traps': 1, 'synthetic': True, ...})
```

### `vllm-gwm import /path/to/bundle --name crm`

```text
imported /path/to/bundle -> .../.gwm_data/graphs/crm
```
