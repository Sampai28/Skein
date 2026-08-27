"""Ollama adapter.

Talks to Ollama's ``/api/generate`` with streaming disabled. Streaming is
supported by the server and deliberately not used here: a step in this runtime
is a unit that either produces an output or does not, and a partially streamed
completion has no meaning to a dependent step. Streaming belongs at the API
edge, where the WebSocket already fans out partial *step* results.

The client is created once and reused. Unlike the demo fetch tool, this is on
the hot path for every LLM step, and a new connection pool per call would add a
TCP and HTTP handshake to each one.
"""

from __future__ import annotations

from typing import Any

import httpx

from skein.errors import LlmError
from skein.llm.base import LlmResponse, estimate_tokens


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2:3b",
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s = timeout_s
        # Injectable so tests can supply a transport without a live server.
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)
        self._owns_client = client is None

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        chosen = model or self.default_model
        payload: dict[str, Any] = {
            "model": chosen,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            # Ollama spells this num_predict.
            payload["options"]["num_predict"] = max_tokens

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LlmError(
                f"ollama returned {exc.response.status_code} for model {chosen!r}",
                model=chosen,
                status=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise LlmError(
                f"could not reach ollama at {self.base_url}: {exc}",
                model=chosen,
                base_url=self.base_url,
            ) from exc

        body = response.json()
        text = body.get("response", "")

        # Ollama reports prompt_eval_count / eval_count on a non-streamed
        # response, but not always — it omits them when the prompt is served
        # entirely from cache. Falling back to an estimate keeps the token
        # budget meaningful instead of letting cached prompts count as free.
        prompt_tokens = body.get("prompt_eval_count")
        completion_tokens = body.get("eval_count")
        estimated = prompt_tokens is None or completion_tokens is None

        return LlmResponse(
            text=text,
            model=chosen,
            prompt_tokens=prompt_tokens if prompt_tokens is not None else estimate_tokens(prompt),
            completion_tokens=(
                completion_tokens if completion_tokens is not None else estimate_tokens(text)
            ),
            estimated=estimated,
            raw={key: value for key, value in body.items() if key != "response"},
        )

    async def health(self) -> bool:
        """Used by ``/readyz``. Distinguishes 'the model server is up' from
        'the model is pulled', which are different failures with different
        fixes."""
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
