"""Resolving ``${steps.x.output.y}`` and ``${inputs.z}`` against run state.

Resolution happens immediately before a step executes, never earlier. A step's
inputs cannot be resolved at submission because its dependencies have not run,
and resolving at scheduling time rather than execution time would open a window
where a value is read before the producing step's output is committed.
"""

from __future__ import annotations

from typing import Any

from skein.errors import BindingResolutionError
from skein.model.workflow import BINDING_RE


class _Unset:
    """Sentinel for 'no item in scope', distinct from an item whose value is None."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()


def resolve_value(
    value: Any,
    outputs: dict[str, Any],
    run_inputs: dict[str, Any],
    item: Any = _UNSET,
) -> Any:
    """Recursively resolve bindings inside a value.

    A binding that is the *entire* string resolves to the referenced object with
    its type intact — ``${steps.search.output.documents}`` yields a list, not the
    string representation of one. That distinction is what lets a ``map`` step
    fan out over a real collection.

    ``item`` is supplied only by a map step, once per element. It is passed as a
    sentinel-defaulted argument rather than an optional dict so that
    ``${item}`` bound to ``None`` is distinguishable from ``item`` not being in
    scope at all — the first is a legitimate value, the second is an error.
    """
    if isinstance(value, str):
        return _resolve_string(value, outputs, run_inputs, item)
    if isinstance(value, dict):
        return {
            key: resolve_value(inner, outputs, run_inputs, item)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [resolve_value(inner, outputs, run_inputs, item) for inner in value]
    return value


def _resolve_string(
    value: str, outputs: dict[str, Any], run_inputs: dict[str, Any], item: Any = _UNSET
) -> Any:
    stripped = value.strip()
    match = BINDING_RE.match(stripped)
    if match:
        return _lookup(match.group(1), outputs, run_inputs, item)

    # Not a whole-string binding. Interpolate any embedded ones into text, which
    # is what a prompt template needs.
    if "${" not in value:
        return value

    result = value
    for token in _embedded_tokens(value):
        inner = token[2:-1]
        resolved = _lookup(inner, outputs, run_inputs, item)
        result = result.replace(token, _stringify(resolved))
    return result


def _embedded_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while True:
        start = value.find("${", index)
        if start == -1:
            break
        end = value.find("}", start)
        if end == -1:
            break
        tokens.append(value[start : end + 1])
        index = end + 1
    return tokens


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def _lookup(
    path: str, outputs: dict[str, Any], run_inputs: dict[str, Any], item: Any = _UNSET
) -> Any:
    parts = path.split(".")
    root = parts[0]

    if root == "inputs":
        return _walk(run_inputs, parts[1:], path)

    if root == "item":
        if isinstance(item, _Unset):
            raise BindingResolutionError(
                f"binding {path!r} uses 'item', which is only in scope inside a map step",
                binding=path,
            )
        return _walk(item, parts[1:], path)

    if root == "steps":
        if len(parts) < 3 or parts[2] != "output":
            raise BindingResolutionError(
                f"malformed binding {path!r}; expected steps.<id>.output[...]",
                binding=path,
            )
        step_id = parts[1]
        if step_id not in outputs:
            # The step exists (validation proved that) but did not succeed, so
            # there is no output to read. Naming both facts saves a round trip
            # to the trace to work out which.
            raise BindingResolutionError(
                f"binding {path!r} refers to step {step_id!r}, which produced no output "
                f"(it failed, was skipped, or has not completed)",
                binding=path,
                step_id=step_id,
                available=sorted(outputs),
            )
        return _walk(outputs[step_id], parts[3:], path)

    raise BindingResolutionError(
        f"unknown binding root {root!r} in {path!r}; expected 'steps', 'inputs' or 'item'",
        binding=path,
    )


def _walk(node: Any, parts: list[str], path: str) -> Any:
    for part in parts:
        if isinstance(node, dict):
            if part not in node:
                raise BindingResolutionError(
                    f"binding {path!r}: key {part!r} not present",
                    binding=path,
                    missing_key=part,
                    available=sorted(node) if isinstance(node, dict) else None,
                )
            node = node[part]
            continue
        if isinstance(node, list) and part.isdigit():
            index = int(part)
            if index >= len(node):
                raise BindingResolutionError(
                    f"binding {path!r}: index {index} out of range (length {len(node)})",
                    binding=path,
                )
            node = node[index]
            continue
        raise BindingResolutionError(
            f"binding {path!r}: cannot read {part!r} from {type(node).__name__}",
            binding=path,
        )
    return node


def evaluate_condition(expression: Any) -> bool:
    """Truthiness for a ``branch`` step's ``when``, after resolution.

    Deliberately not an expression language. ``when`` resolves to a value and
    that value's truthiness decides the branch. Adding an evaluator here would
    mean either shipping ``eval`` — arbitrary code from a workflow definition —
    or maintaining a parser, and neither is worth it when the producing step can
    return a boolean.

    The one special case is the string ``"false"``, which JSON round-trips and
    naive truthiness would treat as true.
    """
    if isinstance(expression, str):
        lowered = expression.strip().lower()
        if lowered in {"false", "0", "no", "", "null", "none"}:
            return False
        return True
    return bool(expression)
