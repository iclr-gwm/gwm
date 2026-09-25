# Vendored harness — provenance & re-sync guide

The GWM harness under `src/vllm_gwm/harness/` is vendored from upstream research
code (a world-model benchmark harness and a standalone GWM package). Keep it
byte-close to those sources so re-sync stays mechanical.

## Vendored files

| Vendored file | Upstream origin | Notes |
|---|---|---|
| `harness/harness.py` | benchmark harness | includes `select_sem_dedup` |
| `harness/llm.py` | benchmark harness | verbatim |
| `harness/graph_sidecar.py` | workflow-mining sidecar | import patches only |
| `harness/wm_prompt.py`, `gv_score.py`, `scrub.py` | benchmark `wm/` tree | verbatim |
| `harness/render.py` | benchmark `wm/lib/render.py` | scrub import patched |
| `harness/prompts/*.md` | benchmark harness prompts | verbatim |
| `viz/show_graph.py`, `viz/show_graph.html` | benchmark graph viewer | payload math kept; HTTP server replaced by FastAPI `/v1/graph` |
| `presets/*.toml` | GWM presets + Toucan bench config | CRM/EOPS mediator config; Toucan `unify_domains` / JSON action space |

## Re-sync procedure

1. Copy upstream files over their counterparts.
2. Re-apply import patches in `graph_sidecar.py` and `render.py`.
3. Run `pytest tests/test_harness_offline.py tests/test_registry.py -m "not model"`.
