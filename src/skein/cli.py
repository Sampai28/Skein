"""A small CLI: submit a workflow file, optionally follow it.

Follows by polling ``GET /runs/{id}`` rather than by opening the WebSocket. The
WebSocket carries every trace event, which is what the viewer wants and is more
than a terminal needs; polling the run summary gives the same status transitions
with a fraction of the code and no reconnect logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from skein.model.validation import validate_workflow
from skein.model.workflow import Workflow

DEFAULT_BASE_URL = "http://localhost:8000"


def load_workflow(path: Path) -> Workflow:
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if "workflow" in raw:
        raw = raw["workflow"]
    return Workflow.model_validate(raw)


async def submit(base_url: str, path: Path, inputs: dict[str, Any], follow: bool) -> int:
    workflow = load_workflow(path)

    # Validated client-side too. The server will validate again — it must, since
    # it cannot trust a client — but failing here gives the error without a
    # round trip and without occupying a queue slot.
    validate_workflow(workflow)

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        response = await client.post(
            "/workflows",
            json={"workflow": workflow.model_dump(mode="json"), "inputs": inputs},
        )
        if response.status_code >= 400:
            print(json.dumps(response.json(), indent=2), file=sys.stderr)
            return 1

        run_id = response.json()["run_id"]
        print(f"run_id: {run_id}")

        if not follow:
            return 0

        seen: dict[str, str] = {}
        while True:
            run = (await client.get(f"/runs/{run_id}")).json()
            for step in run["steps"]:
                if seen.get(step["id"]) != step["status"]:
                    seen[step["id"]] = step["status"]
                    duration = f" {step['duration_s']:.3f}s" if step.get("duration_s") else ""
                    print(f"  {step['id']:<24} {step['status']}{duration}")
            if run["status"] in {"succeeded", "failed", "cancelled"}:
                print(f"\n{run['status']}  ({run.get('duration_s')}s)")
                if run.get("error_message"):
                    print(f"error: {run['error_code']}: {run['error_message']}")
                print(f"trajectory: {run.get('trajectory_hash')}")
                return 0 if run["status"] == "succeeded" else 2
            await asyncio.sleep(0.25)


async def cancel(base_url: str, run_id: str) -> int:
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        response = await client.delete(f"/runs/{run_id}")
        print(json.dumps(response.json(), indent=2))
        return 0 if response.status_code < 400 else 1


async def validate_only(path: Path) -> int:
    workflow = load_workflow(path)
    validate_workflow(workflow)
    print(f"{path}: valid ({len(workflow.steps)} steps)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="skein")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)

    submit_parser = sub.add_parser("submit", help="submit a workflow file")
    submit_parser.add_argument("path", type=Path)
    submit_parser.add_argument("--input", action="append", default=[], metavar="KEY=VALUE")
    submit_parser.add_argument("--follow", action="store_true")

    cancel_parser = sub.add_parser("cancel", help="cancel a run")
    cancel_parser.add_argument("run_id")

    validate_parser = sub.add_parser("validate", help="validate a workflow file locally")
    validate_parser.add_argument("path", type=Path)

    args = parser.parse_args()

    try:
        if args.command == "submit":
            inputs: dict[str, Any] = {}
            for pair in args.input:
                key, _, value = pair.partition("=")
                # Try JSON first so --input k=[1,2] and --input k=true work;
                # fall back to the raw string for the common k=text case.
                try:
                    inputs[key] = json.loads(value)
                except json.JSONDecodeError:
                    inputs[key] = value
            return asyncio.run(submit(args.base_url, args.path, inputs, args.follow))
        if args.command == "cancel":
            return asyncio.run(cancel(args.base_url, args.run_id))
        return asyncio.run(validate_only(args.path))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
