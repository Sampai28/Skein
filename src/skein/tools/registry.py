"""The tool registry.

A tool is an async callable with declared Pydantic input and output models. The
schemas are not decoration: inputs are validated before the call so a binding
mistake fails with a precise error instead of a ``KeyError`` inside someone
else's function, and outputs are validated after so a tool that returns the
wrong shape is classified as a *tool failure* — retryable, circuit-breakable —
rather than crashing the run.

Tools must be async. A synchronous tool that blocks for 200 ms blocks the entire
event loop for 200 ms, stalling every other in-flight step in the process; that
is the single most common way an asyncio service quietly loses its concurrency.
:meth:`ToolRegistry.register_sync` exists for genuinely blocking work and pushes
it to a thread, which is the only correct way to hold that shape.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError as PydanticValidationError

from skein.errors import ToolNotFoundError, ToolOutputSchemaError, ValidationError

ToolFn = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    fn: ToolFn
    input_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None
    description: str = ""
    #: Per-tool concurrency ceiling. None means the registry default applies.
    max_concurrency: int | None = None

    async def invoke(self, inputs: dict[str, Any]) -> Any:
        """Validate, call, validate."""
        call_args = inputs
        if self.input_model is not None:
            try:
                model = self.input_model.model_validate(inputs)
            except PydanticValidationError as exc:
                # A bad input is the caller's fault, not the tool's: raised as a
                # validation error so it is neither retried nor counted against
                # the tool's circuit breaker.
                raise ValidationError(
                    f"tool {self.name!r} rejected its inputs: {exc.error_count()} problem(s)",
                    tool=self.name,
                    errors=exc.errors(include_url=False),
                ) from exc
            call_args = model.model_dump()

        result = await self.fn(**call_args)

        if self.output_model is not None:
            try:
                validated = self.output_model.model_validate(result)
            except PydanticValidationError as exc:
                raise ToolOutputSchemaError(
                    f"tool {self.name!r} returned output its schema rejects",
                    tool=self.name,
                    errors=exc.errors(include_url=False),
                ) from exc
            return validated.model_dump()

        return result


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(
        self,
        name: str,
        fn: ToolFn,
        *,
        input_model: type[BaseModel] | None = None,
        output_model: type[BaseModel] | None = None,
        description: str = "",
        max_concurrency: int | None = None,
    ) -> Tool:
        if not inspect.iscoroutinefunction(fn):
            raise ValueError(
                f"tool {name!r} must be an async function; use register_sync() to "
                f"run blocking work in a thread"
            )
        tool = Tool(
            name=name,
            fn=fn,
            input_model=input_model,
            output_model=output_model,
            description=description,
            max_concurrency=max_concurrency,
        )
        self.tools[name] = tool
        return tool

    def register_sync(
        self,
        name: str,
        fn: Callable[..., Any],
        **kwargs: Any,
    ) -> Tool:
        """Adapt a blocking callable by running it in the default executor.

        ``asyncio.to_thread`` hands the work to a thread pool so the event loop
        keeps turning. Note this does not make CPU-bound work parallel — the GIL
        still applies — it only stops it blocking the loop. Genuinely CPU-bound
        tools want a process pool; see docs/design-notes.md.
        """

        @functools.wraps(fn)
        async def runner(**call_kwargs: Any) -> Any:
            return await asyncio.to_thread(fn, **call_kwargs)

        return self.register(name, runner, **kwargs)

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ToolNotFoundError(
                f"no tool registered as {name!r}",
                tool=name,
                available=sorted(self.tools),
            )
        return self.tools[name]

    def has(self, name: str) -> bool:
        return name in self.tools

    def names(self) -> list[str]:
        return sorted(self.tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": (
                    tool.input_model.model_json_schema() if tool.input_model else None
                ),
                "output_schema": (
                    tool.output_model.model_json_schema() if tool.output_model else None
                ),
            }
            for tool in sorted(self.tools.values(), key=lambda item: item.name)
        ]


def default_registry() -> ToolRegistry:
    """A registry with the demo tools installed."""
    from skein.tools.demo import install_demo_tools

    registry = ToolRegistry()
    install_demo_tools(registry)
    return registry
