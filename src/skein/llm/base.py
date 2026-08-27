"""The LLM interface.

Kept deliberately small — one method, one response type. Everything the runtime
needs from a model is "given a prompt, return text and a token count", and a
narrow interface is what makes the model swappable for a stub in tests and for a
recorded response in replay.

Nothing in the runtime imports :class:`~skein.llm.ollama.OllamaClient` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LlmResponse:
    text: str
    model: str
    #: Best-effort. Not every backend reports usage, and a run's token budget
    #: has to work regardless, so a missing count is estimated rather than
    #: treated as zero — see `estimated` below.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LlmClient(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        ...

    async def close(self) -> None:
        ...


def estimate_tokens(text: str) -> int:
    """A rough token count for backends that do not report usage.

    Four characters per token is the usual English approximation. It is wrong
    for code and wrong for non-Latin scripts, and it is used only so the token
    budget has *something* to count — a budget that silently counts zero is
    worse than one that counts approximately.
    """
    return max(1, len(text) // 4)


class StubLlmClient:
    """A deterministic fake, for tests and for running without Ollama.

    Returns a function of the prompt rather than a constant, so a workflow whose
    branch depends on model output can still be exercised, and so two different
    prompts do not collapse to the same trajectory hash.
    """

    def __init__(self, responses: dict[str, str] | None = None, default: str = "") -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        self.calls.append(prompt)
        text = self.responses.get(prompt, self.default or f"stub:{len(prompt)}")
        return LlmResponse(
            text=text,
            model=model or "stub",
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
            estimated=True,
        )

    async def close(self) -> None:
        return None
