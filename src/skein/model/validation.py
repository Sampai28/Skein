"""Graph-level validation, run once at submission.

Everything here answers a question a single field validator cannot: does this
graph have a cycle, does every reference resolve, can every step actually be
reached. Running it at submission and never again is what lets the scheduler be
simple — it can assume the graph is sound and spend its complexity on
concurrency instead.

Each check raises its own named error so the API can return a precise code and
the metrics can count rejections by rule rather than by "invalid".
"""

from __future__ import annotations

from collections import defaultdict, deque

from skein.errors import (
    DagCycleError,
    DuplicateStepIdError,
    InvalidPolicyValueError,
    MissingBindingError,
    StepCountExceededError,
    UnknownStepReferenceError,
    UnreachableStepError,
)
from skein.model.workflow import MAX_STEPS_HARD_CAP, BINDING_RE, Step, StepKind, Workflow


def validate_workflow(workflow: Workflow) -> None:
    """Run every graph check. Raises the first failure as a typed error."""
    step_map = _check_duplicate_ids(workflow)
    _check_step_count(workflow)
    _check_references(workflow, step_map)
    _check_bindings_declared(workflow, step_map)
    _check_cycles(workflow, step_map)
    _check_reachable(workflow, step_map)
    _check_policy_values(workflow, step_map)


def _check_duplicate_ids(workflow: Workflow) -> dict[str, Step]:
    seen: dict[str, Step] = {}
    duplicates: list[str] = []
    for step in workflow.steps:
        if step.id in seen:
            duplicates.append(step.id)
        seen[step.id] = step
    if duplicates:
        raise DuplicateStepIdError(
            f"duplicate step ids: {sorted(set(duplicates))}",
            step_ids=sorted(set(duplicates)),
        )
    return seen


def _check_step_count(workflow: Workflow) -> None:
    count = len(workflow.steps)
    if count > MAX_STEPS_HARD_CAP:
        raise StepCountExceededError(
            f"workflow has {count} steps, hard cap is {MAX_STEPS_HARD_CAP}",
            step_count=count,
            cap=MAX_STEPS_HARD_CAP,
        )
    if count > workflow.budget.max_steps:
        raise StepCountExceededError(
            f"workflow has {count} steps, budget allows {workflow.budget.max_steps}",
            step_count=count,
            cap=workflow.budget.max_steps,
        )


def _check_references(workflow: Workflow, step_map: dict[str, Step]) -> None:
    """Every id named in depends_on, on_true or on_false must exist."""
    for step in workflow.steps:
        for field, referenced in (
            ("depends_on", step.depends_on),
            ("on_true", step.on_true),
            ("on_false", step.on_false),
        ):
            unknown = [ref for ref in referenced if ref not in step_map]
            if unknown:
                raise UnknownStepReferenceError(
                    f"step {step.id!r} {field} references unknown step(s): {sorted(unknown)}",
                    step_id=step.id,
                    field=field,
                    unknown=sorted(unknown),
                )


def _check_bindings_declared(workflow: Workflow, step_map: dict[str, Step]) -> None:
    """A step that reads another step's output must declare the dependency.

    This is the check that catches the most common authoring mistake. Reading
    ``${steps.fetch.output.body}`` without listing ``fetch`` in ``depends_on``
    produces a workflow that is *sometimes* correct — it works whenever fetch
    happens to finish first, and fails under load when it does not. Rejecting it
    at submission converts a race into a validation error.
    """
    for step in workflow.steps:
        declared = set(step.depends_on)
        referenced = step.binding_references()

        unknown = referenced - set(step_map)
        if unknown:
            raise UnknownStepReferenceError(
                f"step {step.id!r} binds to unknown step(s): {sorted(unknown)}",
                step_id=step.id,
                unknown=sorted(unknown),
            )

        undeclared = referenced - declared
        if undeclared:
            raise MissingBindingError(
                f"step {step.id!r} reads output of {sorted(undeclared)} "
                f"but does not declare them in depends_on",
                step_id=step.id,
                undeclared=sorted(undeclared),
            )

        _check_binding_syntax(step)
        _check_item_scope(step)


def _check_item_scope(step: Step) -> None:
    """``${item...}`` is only in scope inside a map step.

    Caught here rather than at execution because the failure would otherwise
    appear only when the step ran — potentially minutes into a workflow, and
    only on the branch that reached it.
    """
    if step.kind is StepKind.MAP:
        return
    for value in step._binding_strings():
        if "${item}" in value or "${item." in value:
            raise MissingBindingError(
                f"step {step.id!r} uses an ${{item}} binding but is a "
                f"{step.kind.value} step; 'item' exists only inside a map",
                step_id=step.id,
                kind=step.kind.value,
            )


def _check_binding_syntax(step: Step) -> None:
    """Reject a ``${...}`` that is not a well-formed binding.

    A typo like ``${step.fetch.output}`` (singular) would otherwise be passed
    through as a literal string and reach the tool as the text ``${step...}``,
    which is far harder to diagnose than a rejection.
    """
    for value in step._binding_strings():
        if "${" not in value:
            continue
        if value.count("${") == 1 and value.strip().startswith("${") and value.strip().endswith("}"):
            if not BINDING_RE.match(value.strip()):
                raise MissingBindingError(
                    f"step {step.id!r} has a malformed binding: {value!r}; "
                    f"expected ${{steps.<id>.output...}} or ${{inputs...}}",
                    step_id=step.id,
                    binding=value,
                )


def _check_cycles(workflow: Workflow, step_map: dict[str, Step]) -> None:
    """Kahn's algorithm. Whatever is left over is a cycle.

    Preferred to a DFS colouring here because the leftover set *is* the cycle
    membership, so the error can name the steps involved instead of only
    reporting that one exists.
    """
    indegree: dict[str, int] = {step_id: 0 for step_id in step_map}
    dependents: dict[str, list[str]] = defaultdict(list)

    for step in workflow.steps:
        for dependency in step.depends_on:
            indegree[step.id] += 1
            dependents[dependency].append(step.id)

    queue = deque(step_id for step_id, degree in indegree.items() if degree == 0)
    settled = 0
    while queue:
        current = queue.popleft()
        settled += 1
        for dependent in dependents[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)

    if settled != len(step_map):
        involved = sorted(step_id for step_id, degree in indegree.items() if degree > 0)
        raise DagCycleError(
            f"workflow contains a dependency cycle among: {involved}",
            step_ids=involved,
        )


def _check_reachable(workflow: Workflow, step_map: dict[str, Step]) -> None:
    """Every step must be reachable from a root.

    With no cycles, a step is reachable iff it has no dependencies (a root) or
    is transitively downstream of one — which, in an acyclic graph, is every
    step. So the real content of this check is branch targets: a step listed in
    neither ``depends_on`` nor any branch's ``on_true``/``on_false``, and which
    is not itself a root, would never be scheduled. That is almost always a
    forgotten edge rather than an intentional dead step.
    """
    roots = {step.id for step in workflow.steps if not step.depends_on}
    if not roots:
        # Only possible with a cycle, which _check_cycles already caught; kept
        # as a guard so a future edit cannot silently produce an unrunnable
        # workflow.
        raise DagCycleError("workflow has no root step (every step has a dependency)")

    reachable: set[str] = set()
    queue = deque(roots)
    dependents: dict[str, list[str]] = defaultdict(list)
    for step in workflow.steps:
        for dependency in step.depends_on:
            dependents[dependency].append(step.id)

    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        queue.extend(dependents[current])

    orphans = sorted(set(step_map) - reachable)
    if orphans:
        raise UnreachableStepError(
            f"steps unreachable from any root: {orphans}",
            step_ids=orphans,
        )


def _check_policy_values(workflow: Workflow, step_map: dict[str, Step]) -> None:
    """Cross-field policy checks the field validators cannot see."""
    for step in workflow.steps:
        timeout = workflow.timeout_for(step)
        if timeout > workflow.budget.max_duration_s:
            raise InvalidPolicyValueError(
                f"step {step.id!r} timeout ({timeout}s) exceeds the workflow "
                f"duration budget ({workflow.budget.max_duration_s}s); the step "
                f"could never finish inside the run",
                step_id=step.id,
                timeout_s=timeout,
                max_duration_s=workflow.budget.max_duration_s,
            )

        retry = workflow.retry_for(step)
        # Worst case a single step can occupy: every attempt burning its full
        # timeout, plus the backoff between them.
        worst_case = timeout * retry.max_attempts + retry.max_backoff_s * (retry.max_attempts - 1)
        if worst_case > workflow.budget.max_duration_s:
            raise InvalidPolicyValueError(
                f"step {step.id!r} could consume {worst_case:.1f}s across "
                f"{retry.max_attempts} attempts, exceeding the workflow duration "
                f"budget of {workflow.budget.max_duration_s}s",
                step_id=step.id,
                worst_case_s=round(worst_case, 3),
                max_duration_s=workflow.budget.max_duration_s,
            )

    branch_targets = {
        target
        for step in workflow.steps
        if step.kind is StepKind.BRANCH
        for target in (*step.on_true, *step.on_false)
    }
    for target in sorted(branch_targets):
        branch_owners = [
            step.id
            for step in workflow.steps
            if step.kind is StepKind.BRANCH and target in (*step.on_true, *step.on_false)
        ]
        if len(branch_owners) > 1:
            raise InvalidPolicyValueError(
                f"step {target!r} is a branch target of more than one branch "
                f"({branch_owners}); its skip state would be ambiguous",
                step_id=target,
                branches=branch_owners,
            )
