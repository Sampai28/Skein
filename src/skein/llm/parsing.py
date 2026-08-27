"""Parsing model output into typed models, with a documented fallback.

Models return text. Workflows want structure. The gap between those two is
where most agent frameworks quietly fail, because a model that has answered
correctly in prose but not in the requested JSON is neither a success nor a
clean failure.

**The fallback ladder**, in order, each step only attempted if the previous
failed:

1. Parse the whole response as JSON and validate it.
2. Extract the first fenced ``json`` block and try that.
3. Extract the outermost balanced ``{...}`` or ``[...]`` and try that.
4. Raise :class:`~skein.errors.LlmParseError`, which is classified retryable.

Step 4 matters as much as the others. A parse failure is *not* silently turned
into a null result or a best-guess coercion — it is a typed error that the retry
layer sees, so the model gets resampled, and if it keeps failing the step fails
honestly rather than passing a plausible-looking empty object downstream.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from skein.errors import LlmParseError

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_candidates(text: str) -> list[str]:
    """Candidate JSON substrings, most-likely first."""
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    candidates.extend(match.strip() for match in _FENCE_RE.findall(text))

    for opener, closer in (("{", "}"), ("[", "]")):
        block = _balanced_span(text, opener, closer)
        if block:
            candidates.append(block)

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _balanced_span(text: str, opener: str, closer: str) -> str | None:
    """The outermost balanced span, or None.

    Counting brackets rather than regex-matching, because a regex cannot match
    nested structures and a model's JSON is usually nested. String contents are
    skipped so a brace inside a quoted value does not throw off the depth.
    """
    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_typed(text: str, model: type[T]) -> T:
    """Parse ``text`` into ``model``, walking the fallback ladder."""
    errors: list[str] = []

    for candidate in extract_json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"json: {exc.msg}")
            continue
        try:
            return model.model_validate(payload)
        except PydanticValidationError as exc:
            errors.append(f"schema: {exc.error_count()} problem(s)")
            continue

    raise LlmParseError(
        f"could not parse model output into {model.__name__}",
        model=model.__name__,
        attempts=errors[:5],
        # Truncated: the raw text can be long, and the whole thing ends up in an
        # error payload that is written to the trace and returned over HTTP.
        text_preview=text[:500],
    )


def parse_typed_or_none(text: str, model: type[T]) -> T | None:
    """Non-raising variant, for callers that have a real default.

    Used only where a missing value is genuinely acceptable. Reaching for this
    to make a parse error go away is how a workflow ends up producing confident
    empty results.
    """
    try:
        return parse_typed(text, model)
    except LlmParseError:
        return None


def schema_hint(model: type[BaseModel]) -> str:
    """A compact schema description to append to a prompt.

    Substantially improves the odds of step 1 in the ladder succeeding, which
    is cheaper than relying on retries to eventually produce valid JSON.
    """
    schema = model.model_json_schema()
    return (
        "Respond with JSON only, matching this schema. No prose, no code fence.\n"
        + json.dumps(schema, indent=None, separators=(",", ":"))
    )
