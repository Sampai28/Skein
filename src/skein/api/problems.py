"""RFC 7807 problem details for every error path.

One handler per error family, all producing the same shape. Clients branch on
the ``code`` extension member; ``detail`` is prose and will change.

``application/problem+json`` rather than ``application/json``, because that is
what the RFC specifies and what makes a client library treat the body as an
error document rather than as a successful response that happens to have an
error field in it.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from skein.errors import SkeinError

logger = logging.getLogger("skein.api")

PROBLEM_MEDIA_TYPE = "application/problem+json"


def problem_response(exc: SkeinError, instance: str) -> JSONResponse:
    body = exc.to_problem()
    body["instance"] = instance
    headers: dict[str, str] = {}

    # Retry-After where the error carries one. A 429 or 503 without it leaves
    # the client to guess, and clients guess badly — usually by retrying
    # immediately, which is the behaviour the status code exists to prevent.
    retry_after = exc.details.get("retry_after_s")
    if retry_after is not None:
        headers["Retry-After"] = str(max(1, int(float(retry_after))))

    return JSONResponse(
        status_code=exc.http_status,
        content=body,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(SkeinError)
    async def _skein_error(request: Request, exc: SkeinError) -> JSONResponse:
        if exc.http_status >= 500:
            logger.error("%s on %s: %s", exc.code, request.url.path, exc.message)
        return problem_response(exc, str(request.url.path))

    @app.exception_handler(RequestValidationError)
    async def _request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's default 422 body is a bare list under "detail". Reshaped
        # into the same problem envelope so a client has exactly one error
        # format to parse.
        return JSONResponse(
            status_code=422,
            media_type=PROBLEM_MEDIA_TYPE,
            content={
                "type": "https://skein.dev/problems/request-validation",
                "title": "RequestValidationError",
                "status": 422,
                "detail": "the request body did not match the expected schema",
                "code": "request_validation",
                "instance": str(request.url.path),
                "errors": _sanitise(exc.errors()),
            },
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Full detail to the log, nothing internal to the client. An exception
        # message can carry a file path, a hostname or a fragment of input.
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            media_type=PROBLEM_MEDIA_TYPE,
            content={
                "type": "https://skein.dev/problems/internal-error",
                "title": "InternalError",
                "status": 500,
                "detail": "an unexpected error occurred",
                "code": "internal_error",
                "instance": str(request.url.path),
            },
        )


def _sanitise(errors: list[dict]) -> list[dict]:
    """Drop the ``ctx`` key, which can contain the original exception object and
    is not JSON-serialisable in every pydantic version."""
    cleaned: list[dict] = []
    for error in errors:
        cleaned.append(
            {key: value for key, value in error.items() if key not in {"ctx", "url"}}
        )
    return cleaned
