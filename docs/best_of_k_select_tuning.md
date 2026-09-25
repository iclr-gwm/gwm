# Best-of-K Select Tuning

These `VLLM_GWM_*` env vars tune the `mode="select"` best-of-K path. All are
**default-off** (behaviour unchanged unless set); a preset may enable a
validated default per benchmark.

| Env | Default | Effect |
|---|---|---|
| `VLLM_GWM_GREEDY_ANCHOR` | `0` | Draw candidate 0 at temperature=0 (greedy) and 1..k-1 at the request temperature, pinning the greedy sample at index 0. On identical/low-margin steps the select arm then degrades to greedy (base) behaviour instead of returning a temperature sample. |
| `VLLM_GWM_SELECT_MIN_MARGIN` | `0.0` | Keep candidate 0 unless the judge's winner beats it by >= this score margin; suppresses low-margin coin-flip overrides. Best paired with the greedy anchor. |
| `VLLM_GWM_SELECT_EXAMPLES` | preset (`0`) | Number of successful past-trajectory exemplars shown to the judge at the classified state. |
| `VLLM_GWM_SELECT_NEG_EXAMPLES` | `0` | Number of contrastive **step-labeled** negative exemplars (actions that errored at this state) shown to the judge as "failed attempts here". |
| `VLLM_GWM_SELECT_VOTES` | `1` | Judge votes over reshuffled candidate orders; majority `best` / mean scores. |
| `VLLM_GWM_SELECT_TOURNAMENT` | `0` | Single-elimination pairwise bracket instead of one K-way judge call (helps at large K where K-way judging saturates). |
| `VLLM_GWM_SELECT_SEM_DEDUP` | preset | Dedup candidates on the tool call (name+args) only, ignoring free text / whitespace. |

## Preset Defaults

A preset TOML may ship a validated config: `greedy_anchor` under `[preset]`,
and any `HarnessConfig` field (e.g. `select_min_margin`, `select_examples`)
under `[harness]`. The global env switch is OR-ed with the preset's
`greedy_anchor`. The bundled **`toucan`** preset ships the validated winner
(`greedy_anchor = true`, `select_min_margin = 0.2`): on Qwen3.6
`toucan_100_test` this converts a -5 best-of-K regression into a reproducible +4
over greedy base (55 -> mean 59). `crm` / `eops` are unchanged (anchor off).
