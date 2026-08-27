"""A small demo tool set: calculator, HTTP fetch, retrieval stub.

Three tools, chosen to cover the three shapes that behave differently under
concurrency rather than to be useful:

* ``calculator`` is CPU-bound and fast — it returns without ever yielding, which
  makes it the right tool for testing that the scheduler's own overhead is not
  the bottleneck.
* ``http_fetch`` is I/O-bound and genuinely awaits, so it is what exercises real
  concurrency and real cancellation.
* ``retrieval`` is I/O-bound with a deterministic in-memory corpus, so tests can
  assert on content without a network.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from skein.errors import ToolError
from skein.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorInput(BaseModel):
    expression: str = Field(max_length=256)


class CalculatorOutput(BaseModel):
    expression: str
    result: float


def _safe_eval(node: ast.AST) -> float:
    """Evaluate an arithmetic AST.

    Walking the AST rather than calling ``eval``. ``eval`` on a string that
    reached us through a workflow definition — which may itself have come from
    a model's output — is arbitrary code execution, and no amount of input
    filtering makes it safe.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ToolError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if op in (operator.truediv, operator.floordiv, operator.mod) and right == 0:
            raise ToolError("division by zero")
        # A large exponent is a denial of service on a single call: 9**9**9
        # occupies a core for a very long time inside an event loop that cannot
        # preempt it.
        if op is operator.pow and abs(right) > 64:
            raise ToolError("exponent too large")
        return float(op(left, right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported unary operator: {type(node.op).__name__}")
        return float(op(_safe_eval(node.operand)))
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


async def calculator(expression: str) -> dict[str, Any]:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {expression!r}") from exc
    return {"expression": expression, "result": _safe_eval(parsed)}


# ---------------------------------------------------------------------------
# http_fetch
# ---------------------------------------------------------------------------

class FetchInput(BaseModel):
    url: str = Field(max_length=2048)
    timeout_s: float = Field(default=10.0, gt=0, le=60)


class FetchOutput(BaseModel):
    url: str
    status: int
    body: str
    truncated: bool


#: Bodies are truncated. A tool output flows into a step output, which flows
#: into the trace and into any dependent step's inputs — an unbounded response
#: would be written to disk and held in memory for the life of the run.
MAX_BODY_CHARS = 8192


async def http_fetch(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    try:
        # A fresh client per call rather than a shared one. A shared client is
        # more efficient and would need lifecycle management tied to the app,
        # which is worth doing in production and is not worth the coupling in a
        # demo tool. Noted rather than hidden.
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise ToolError(f"fetch of {url!r} timed out after {timeout_s}s", url=url) from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"fetch of {url!r} failed: {exc}", url=url) from exc

    body = response.text
    truncated = len(body) > MAX_BODY_CHARS
    return {
        "url": url,
        "status": response.status_code,
        "body": body[:MAX_BODY_CHARS],
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# retrieval stub
# ---------------------------------------------------------------------------

class RetrievalInput(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    k: int = Field(default=3, ge=1, le=20)


class RetrievalDocument(BaseModel):
    id: str
    text: str
    score: float


class RetrievalOutput(BaseModel):
    query: str
    documents: list[RetrievalDocument]


class _Corpus:
    """A fixed in-memory corpus with deterministic scoring.

    Scoring is token overlap, not embeddings. That is the point: the retrieval
    step exists so a workflow has a realistic fan-out source, and a real vector
    store would make every test depend on a model download.
    """

    DOCS: ClassVar[dict[str, str]] = {
        "doc-asyncio": (
            "asyncio runs coroutines on a single-threaded event loop. Concurrency "
            "comes from tasks yielding at await points, not from parallelism."
        ),
        "doc-taskgroup": (
            "TaskGroup provides structured concurrency: the block does not exit "
            "until every child task has finished, and a failing child cancels its "
            "siblings."
        ),
        "doc-cancellation": (
            "Cancellation in asyncio is delivered as a CancelledError raised at "
            "the next await point. It inherits from BaseException so that except "
            "Exception does not swallow it."
        ),
        "doc-backpressure": (
            "Backpressure means refusing or slowing intake when downstream cannot "
            "keep up. A bounded queue applies it; an unbounded queue defers it "
            "until memory runs out."
        ),
        "doc-breaker": (
            "A circuit breaker stops calling a failing dependency so that it can "
            "recover and so that callers fail fast instead of waiting."
        ),
        "doc-retry": (
            "Exponential backoff with jitter spreads retries out. Without jitter, "
            "every client retries at the same instant and re-creates the overload."
        ),
    }

    @classmethod
    def search(cls, query: str, k: int) -> list[dict[str, Any]]:
        terms = {token.lower().strip(".,") for token in query.split() if len(token) > 2}
        scored: list[tuple[float, str, str]] = []
        for doc_id, text in cls.DOCS.items():
            tokens = {token.lower().strip(".,") for token in text.split()}
            overlap = len(terms & tokens)
            if overlap:
                scored.append((overlap / max(1, len(terms)), doc_id, text))
        # Sort by score then id, so ties are broken deterministically and the
        # same query always produces the same trajectory hash.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"id": doc_id, "text": text, "score": round(score, 4)}
            for score, doc_id, text in scored[:k]
        ]


async def retrieval(query: str, k: int = 3) -> dict[str, Any]:
    return {"query": query, "documents": _Corpus.search(query, k)}


# ---------------------------------------------------------------------------

def install_demo_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        "calculator",
        calculator,
        input_model=CalculatorInput,
        output_model=CalculatorOutput,
        description="Evaluate an arithmetic expression.",
    )
    registry.register(
        "http_fetch",
        http_fetch,
        input_model=FetchInput,
        output_model=FetchOutput,
        description="GET a URL and return the (truncated) body.",
        # Lower than the default: outbound HTTP is the tool most likely to be
        # rate-limited by whatever is on the other end.
        max_concurrency=4,
    )
    registry.register(
        "retrieval",
        retrieval,
        input_model=RetrievalInput,
        output_model=RetrievalOutput,
        description="Search a small fixed corpus by token overlap.",
    )
    return registry
