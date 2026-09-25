#!/usr/bin/env python3
"""A tiny mock OpenAI-compatible (vLLM-style) server for offline testing.

Serves ``/v1/chat/completions`` (plus ``/v1/models`` and ``/health``) with canned,
deterministic replies shaped for both roles the GWM stack uses a model for:

* **GWM harness LLM** — recognises the advise / select_joint / select_single
  prompts and returns the right JSON contract (``{"advice": …}`` /
  ``{"compare","best","scores"}`` / ``{"score","reason"}``);
* **policy LLM** — for a normal agent chat (e.g. e2e "what is your next step?")
  returns a plain-text next step.

This lets the offline tests and ``scripts/e2e_demo.py`` exercise the real
``ChatClient`` HTTP path (urllib POST → parse OpenAI envelope, real token usage)
with no GPU and no served model.

Standalone:  python scripts/mock_vllm.py --port 9000 --model Qwen/Qwen3.6-27B
In tests:    from mock_vllm import MockVLLM;  with MockVLLM() as base_url: ...
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple


def _toks(text: str) -> int:
    return max(1, len(text) // 4)


def canned_reply(messages: List[Dict[str, Any]]) -> str:
    """Deterministic assistant content for a chat, by prompt shape."""
    sys_ = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
    user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")

    # --- GWM harness: whole-rollout single score ---
    if "Rate this rollout" in user:
        return json.dumps({"score": 0.5, "reason": "mock vLLM rollout score"})

    # --- GWM harness: best-of-K joint selection ---
    if re.search(r"^candidate \d+:", user, flags=re.M) or '"compare"' in user or "[Candidates]" in user:
        idx = [int(m) for m in re.findall(r"^candidate (\d+):", user, flags=re.M)]
        n = (max(idx) + 1) if idx else 1
        return json.dumps({"compare": "mock vLLM comparison",
                           "best": 0,
                           "scores": [round(1.0 - 0.1 * i, 2) for i in range(n)]})

    # --- GWM harness: advise ---
    if "World-Model harness agent" in sys_:
        if "RETRY-LOOP detected live" in user or "TRAP: this is a known trap" in user:
            return json.dumps({"advice": "Stop repeating the failing call; read the "
                                         "last error and change approach — re-check the "
                                         "identifier/table before retrying (mock vLLM)."})
        return json.dumps({"advice": ""})

    # --- policy LLM: next step in a normal agent chat ---
    return ("I will stop repeating the failing call and instead re-read the relevant "
            "record / inspect the schema, then issue a corrected call. (mock vLLM policy)")


def openai_envelope(model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    content = canned_reply(messages)
    prompt_tokens = sum(_toks(str(m.get("content", ""))) for m in messages)
    completion_tokens = _toks(content)
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def _make_handler(model: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, obj: Any):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path in ("/health", "/healthz"):
                self._send(200, {"status": "ok"})
            elif path.endswith("/v1/models") or path.endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("chat/completions"):
                self._send(404, {"error": "not found"}); return
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode()) if n else {}
            req_model = body.get("model") or model
            self._send(200, openai_envelope(req_model, body.get("messages") or []))
    return H


def serve(host: str, port: int, model: str) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), _make_handler(model))
    return srv


@contextmanager
def MockVLLM(model: str = "mock-model", host: str = "127.0.0.1"):
    """Context manager: start the mock on an ephemeral port, yield its ``/v1`` base URL."""
    srv = serve(host, 0, model)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    actual_port = srv.server_address[1]
    try:
        yield f"http://{host}:{actual_port}/v1"
    finally:
        srv.shutdown()
        srv.server_close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    args = ap.parse_args()
    srv = serve(args.host, args.port, args.model)
    print(f"[mock-vllm] serving OpenAI-compatible API on "
          f"http://{args.host}:{args.port}/v1  model={args.model}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
