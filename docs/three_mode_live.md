# Live three-mode GWM evaluation

Runs the GWM plugin end-to-end against a live vLLM server in **three modes**,
across two agent suites (`eops`, `crm`), on three GPUs in parallel (one mode per
GPU).

| Mode | What it exercises |
|------|-------------------|
| **frozen** | A pre-mined graph bundle is loaded read-only; GWM gates generation via advise (K=1) and select (K>1) over held-out tasks. |
| **build**  | Mine a graph from a *ratio* of rollouts (e.g. 20%), activate it live via `/v1/gwm/rollback`, then run GWM over the **remaining** rollouts. |
| **evolve** | Seed the live rollout store, force an `EvolveManager` rebuild (`/v1/gwm/evolve`), wait for it to finalize, then run GWM against the evolved graph. |

## Reproduce

```bash
# 1. Fresh, isolated environment (vLLM from the source the plugin targets + this package)
VLLM_SRC=/path/to/vllm bash scripts/fresh_env.sh          # -> .venv-live

# 2. Rollout corpora laid out as <ROLLOUTS_ROOT>/<suite>/<ratio>/out/rollouts.jsonl.gz
#    (build ratio, e.g. 20p; held-out eval pool, e.g. 100p)

# 3. Run all three modes in parallel on GPUs 0,2,3
HF_HOME=/path/to/hf_home \
MODEL=google/gemma-4-E2B-it MODEL_SLUG=gemma4-e2b \
ROLLOUTS_ROOT=/path/to/rollouts RATIO=20p N_TEST=10 \
GPUS="0 2 3" PORTS="8801 8802 8803" \
bash scripts/three_mode_live.sh

# Repeat with MODEL=Qwen/Qwen3.6-27B MODEL_SLUG=qwen36-27b
```

Per-mode JSON reports land in `.gwm_live/<slug>/{frozen,build,evolve}.json`.

- `frozen` uses the in-repo bundles (`graphs/eops/full`, `graphs/crm/full`).
- Held-out eval = records in the `100p` pool whose `uid` is **not** in the build
  `ratio` set (the "remaining ratio").
- `gold_hit` = fraction of graded candidates whose text references the gold
  next-tool of the trajectory (a coarse, model-agnostic signal, not a full
  task-success metric).
- Sparse ratios can't sustain per-domain clustering; the pipeline transparently
  falls back to global clustering (recorded as `discover_mode`).

## Results

Environment: 3× H200 GPUs (one mode per GPU), MiniLM (`all-MiniLM-L6-v2`)
state embedder on CPU, build ratio 20p, 10 held-out tasks/suite, K∈{1,4}.

### gemma-4-E2B-it

| Mode | Suite | tasks | gold_hit | avg latency (ms) | notes |
|------|-------|:---:|:---:|:---:|-------|
| frozen | eops | 10 | 0.31 | 1399 | advise+select, 1 guidance injection |
| frozen | crm  | 10 | 0.30 | 1028 | |
| build  | eops | 10 | 0.38 | 499 | within-domain, 15 states, activated |
| build  | crm  | 10 | 0.30 | 567 | global fallback, 86 states, activated |
| evolve | eops | 10 | 0.25 | 503 | rebuild completed (seeded 327) |
| evolve | crm  | 10 | 0.25 | 589 | rebuild completed (seeded 332) |

### Qwen3.6-27B

| Mode | Suite | tasks | gold_hit | avg latency (ms) | notes |
|------|-------|:---:|:---:|:---:|-------|
| frozen | eops | 10 | 0.80 | 2849 | advise+select, 1 guidance injection |
| frozen | crm  | 10 | 0.25 | 3356 | |
| build  | eops | 10 | 0.85 | 2338 | within-domain, 15 states, activated |
| build  | crm  | 10 | 0.25 | 3003 | global fallback, 86 states, activated |
| evolve | eops | 10 | 0.85 | 2354 | rebuild completed (seeded 327) |
| evolve | crm  | 10 | 0.30 | 3104 | rebuild completed (seeded 332) |

All 12 cells ran live end-to-end: every `build` activated over HTTP (status 200)
and every `evolve` job finalized. The larger policy (27B) resolves eops
next-actions far more often than the 2B model; crm is harder for both.
