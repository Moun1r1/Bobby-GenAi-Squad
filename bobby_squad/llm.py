"""bobby_squad.llm — a pluggable, provider-agnostic chat backend.

The whole library only needs a callable  (messages: list[dict], max_tokens: int) -> str.  `LLM` is a stdlib
default that talks to any OpenAI-compatible /v1/chat/completions endpoint (configurable via GA_LLM_URL /
GA_LLM_MODEL). Swap it for your own callable to route through a gateway, a hosted API, etc. Any provider-specific
request fields go in GA_EXTRA_BODY as a JSON object and are merged into the request body.
"""
import json
import os
import urllib.request
from typing import List, Optional

_env = lambda *ks, d="": next((os.environ[k] for k in ks if os.environ.get(k)), d)  # noqa: E731
DEFAULT_URL = _env("BOBBY_LLM_URL", "GA_LLM_URL", d="http://localhost:8000/v1/chat/completions")
DEFAULT_MODEL = _env("BOBBY_LLM_MODEL", "GA_LLM_MODEL", d="local-model")   # served model id — set to yours
# provider-specific request fields (operator-supplied), merged verbatim into every request body
DEFAULT_EXTRA_BODY = json.loads(_env("BOBBY_EXTRA_BODY", "GA_EXTRA_BODY", d="{}"))


class LLM:
    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 temperature: float = 0.5, timeout: int = 120, extra_body: Optional[dict] = None):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body = extra_body if extra_body is not None else DEFAULT_EXTRA_BODY
        self.last_usage = {}         # {prompt_tokens, completion_tokens, total_tokens} from the most recent call

    def _post(self, payload: dict) -> dict:
        if self.extra_body:
            payload.update(self.extra_body)
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
            self.last_usage = data.get("usage", {}) or {}
            return data["choices"][0]["message"]
        except Exception:
            self.last_usage = {}
            return {}

    def __call__(self, messages: List[dict], max_tokens: int = 160, temperature=None) -> str:
        msg = self._post({"model": self.model, "messages": messages, "max_tokens": max_tokens,
                          "temperature": self.temperature if temperature is None else temperature})
        return msg.get("content") or ""

    def chat(self, messages: List[dict], tools: Optional[list] = None, max_tokens: int = 300, temperature=None) -> dict:
        """Full chat turn with optional NATIVE function-calling. Returns the assistant message dict (with `content`
        and, if the model called tools, `tool_calls`). Uses the endpoint's real tool-calling, not a text protocol."""
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": self.temperature if temperature is None else temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return self._post(payload)
