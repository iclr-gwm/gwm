# API backend

Serve GWM against any OpenAI-compatible `/v1` endpoint — no local GPU required.

## Modules

- `openai.py` — `OpenAIAPI` (`POST {base}/chat/completions`, Bearer auth)
- `throttle.py` — quota gate (`x-quota-usage-percent`, 429 backoff)
- `factory.py` — `build_api()`

## Usage

```bash
export OPENAI_API_KEY=...
vllm-gwm serve --backend api \
  --api-base https://api.openai.com/v1 \
  --model gpt-4o \
  --gwm-modules crm=./graphs/crm/full
```

## Quota

Reads `x-quota-usage-percent` / `x-quota-reset-date` on success. Slows down when
usage ≥ `VLLM_GWM_API_QUOTA_SLOW_PERCENT` (default 80). On 429:
`delay = NUM_TRY * VLLM_GWM_API_RETRY_STEP_SEC + (waiters-1)*step`.
