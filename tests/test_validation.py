"""Every DAG and budget reject path.

Validation runs at submission, so each of these is a request the runtime should
refuse before it occupies a queue slot or a worker.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from skein.errors import (
    DagCycleError,
    DuplicateStepIdError,
    InvalidPolicyValueError,
    MissingBindingError,
    StepCountExceededError,
    UnknownStepReferenceError,
)
from skein.model.validation import validate_workflow
from skein.model.workflow import Budget, RetryPolicy, Step, StepKind, Workflow
from tests.conftest import make_step, make_workflow


def test_a_linear_workflow_validates():
    workflow = make_workflow(
        [
            make_step("a"),
            make_step("b", depends_on=["a"]),
            make_step("c", depends_on=["b"]),
        ]
    )
    validate_workflow(workflow)


def test_a_diamond_validates():
    workflow = make_workflow(
        [
            make_step("root"),
            make_step("left", depends_on=["root"]),
            make_step("right", depends_on=["root"]),
            make_step("join", depends_on=["left", "right"]),
        ]
    )
    validate_workflow(workflow)


# -- cycles -----------------------------------------------------------------

def test_a_two_step_cycle_is_rejected():
    workflow = Workflow.model_construct(
        name="cyclic",
        version="1",
        steps=[
            make_step("a", depends_on=["b"]),
            make_step("b", depends_on=["a"]),
        ],
        budget=Budget(),
        defaults_timeout_s=30.0,
        defaults_retry=RetryPolicy(),
        max_concurrency=4,
        retry_budget=10,
    )
    with pytest.raises(DagCycleError) as excinfo:
        validate_workflow(workflow)
    # The error names the members, not merely the existence of a cycle.
    assert set(excinfo.value.details["step_ids"]) == {"a", "b"}


def test_a_longer_cycle_is_rejected():
    workflow = Workflow.model_construct(
        name="cyclic",
        version="1",
        steps=[
            make_step("a", depends_on=["c"]),
            make_step("b", depends_on=["a"]),
            make_step("c", depends_on=["b"]),
        ],
        budget=Budget(),
        defaults_timeout_s=30.0,
        defaults_retry=RetryPolicy(),
        max_concurrency=4,
        retry_budget=10,
    )
    with pytest.raises(DagCycleError):
        validate_workflow(workflow)


def test_a_self_dependency_is_rejected_by_the_model():
    # Caught at model construction rather than graph validation: a step that
    # depends on itself is malformed in isolation.
    with pytest.raises(PydanticValidationError):
        Step(id="a", kind=StepKind.TOOL_CALL, tool="noop", depends_on=["a"])


# -- references -------------------------------------------------------------

def test_unknown_dependency_is_rejected():
    workflow = make_workflow([make_step("a"), make_step("b", depends_on=["nope"])])
    with pytest.raises(UnknownStepReferenceError) as excinfo:
        validate_workflow(workflow)
    assert excinfo.value.details["unknown"] == ["nope"]


def test_duplicate_step_ids_are_rejected():
    workflow = make_workflow([make_step("a"), make_step("a")])
    with pytest.raises(DuplicateStepIdError):
        validate_workflow(workflow)


def test_binding_to_an_undeclared_dependency_is_rejected():
    """The race this prevents is the reason validation exists at all."""
    workflow = make_workflow(
        [
            make_step("producer"),
            make_step("consumer", inputs={"x": "${steps.producer.output.value}"}),
        ]
    )
    with pytest.raises(MissingBindingError) as excinfo:
        validate_workflow(workflow)
    assert excinfo.value.details["undeclared"] == ["producer"]


def test_binding_to_a_declared_dependency_is_accepted():
    workflow = make_workflow(
        [
            make_step("producer"),
            make_step(
                "consumer",
                depends_on=["producer"],
                inputs={"x": "${steps.producer.output.value}"},
            ),
        ]
    )
    validate_workflow(workflow)


def test_a_malformed_binding_is_rejected():
    workflow = make_workflow(
        [make_step("a", inputs={"x": "${step.a.output}"})]  # 'step', not 'steps'
    )
    with pytest.raises(MissingBindingError):
        validate_workflow(workflow)


def test_item_binding_outside_a_map_is_rejected():
    workflow = make_workflow([make_step("a", inputs={"x": "${item.score}"})])
    with pytest.raises(MissingBindingError) as excinfo:
        validate_workflow(workflow)
    assert excinfo.value.details["kind"] == "tool_call"


def test_item_binding_inside_a_map_is_accepted():
    workflow = make_workflow(
        [
            make_step("src"),
            Step(
                id="fan",
                kind=StepKind.MAP,
                tool="noop",
                depends_on=["src"],
                over="${steps.src.output.items}",
                inputs={"expression": "${item.score}"},
            ),
        ]
    )
    validate_workflow(workflow)


# -- reachability -----------------------------------------------------------

def test_a_workflow_where_every_step_has_a_dependency_is_rejected():
    workflow = Workflow.model_construct(
        name="rootless",
        version="1",
        steps=[
            make_step("a", depends_on=["b"]),
            make_step("b", depends_on=["a"]),
        ],
        budget=Budget(),
        defaults_timeout_s=30.0,
        defaults_retry=RetryPolicy(),
        max_concurrency=4,
        retry_budget=10,
    )
    with pytest.raises(DagCycleError):
        validate_workflow(workflow)


# -- counts and policies ----------------------------------------------------

def test_step_count_over_budget_is_rejected():
    steps = [make_step(f"s{index}") for index in range(20)]
    with pytest.raises(PydanticValidationError):
        # The model itself refuses, because budget.max_steps is part of the
        # workflow rather than a runtime setting.
        make_workflow(steps, budget=Budget(max_steps=5))


def test_step_count_over_the_hard_cap_is_rejected():
    steps = [make_step(f"s{index}") for index in range(600)]
    workflow = Workflow.model_construct(
        name="huge",
        version="1",
        steps=steps,
        budget=Budget(max_steps=512),
        defaults_timeout_s=30.0,
        defaults_retry=RetryPolicy(),
        max_concurrency=4,
        retry_budget=10,
    )
    with pytest.raises(StepCountExceededError):
        validate_workflow(workflow)


def test_a_step_timeout_longer_than_the_run_budget_is_rejected():
    workflow = make_workflow(
        [make_step("a", timeout_s=120.0)],
        budget=Budget(max_duration_s=60.0),
    )
    with pytest.raises(InvalidPolicyValueError) as excinfo:
        validate_workflow(workflow)
    assert excinfo.value.details["step_id"] == "a"


def test_retries_that_cannot_fit_the_run_budget_are_rejected():
    """A step whose full attempt sequence exceeds the run budget can never
    complete, and should be refused rather than discovered at timeout."""
    workflow = make_workflow(
        [
            make_step(
                "a",
                timeout_s=20.0,
                retry=RetryPolicy(max_attempts=5, initial_backoff_s=1, max_backoff_s=30),
            )
        ],
        budget=Budget(max_duration_s=60.0),
    )
    with pytest.raises(InvalidPolicyValueError) as excinfo:
        validate_workflow(workflow)
    assert "worst_case_s" in excinfo.value.details


def test_out_of_range_retry_values_are_rejected_by_the_model():
    with pytest.raises(PydanticValidationError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(PydanticValidationError):
        RetryPolicy(max_attempts=99)
    with pytest.raises(PydanticValidationError):
        RetryPolicy(initial_backoff_s=-1)


def test_max_backoff_below_initial_is_rejected():
    with pytest.raises(PydanticValidationError):
        RetryPolicy(initial_backoff_s=10, max_backoff_s=1)


def test_a_step_kind_missing_its_required_field_is_rejected():
    with pytest.raises(PydanticValidationError):
        Step(id="a", kind=StepKind.TOOL_CALL)  # no tool
    with pytest.raises(PydanticValidationError):
        Step(id="a", kind=StepKind.LLM_CALL)  # no prompt
    with pytest.raises(PydanticValidationError):
        Step(id="a", kind=StepKind.MAP, tool="t")  # no over
    with pytest.raises(PydanticValidationError):
        Step(id="a", kind=StepKind.REDUCE)  # no reducer
    with pytest.raises(PydanticValidationError):
        Step(id="a", kind=StepKind.BRANCH)  # no when


def test_a_step_targeted_by_two_branches_is_rejected():
    workflow = make_workflow(
        [
            make_step("root"),
            Step(
                id="b1",
                kind=StepKind.BRANCH,
                depends_on=["root"],
                when="${steps.root.output.ok}",
                on_true=["target"],
            ),
            Step(
                id="b2",
                kind=StepKind.BRANCH,
                depends_on=["root"],
                when="${steps.root.output.ok}",
                on_false=["target"],
            ),
            make_step("target", depends_on=["root"]),
        ]
    )
    with pytest.raises(InvalidPolicyValueError):
        validate_workflow(workflow)


def test_invalid_step_ids_are_rejected():
    for bad in ("", "1abc", "has space", "has/slash", "x" * 65):
        with pytest.raises(PydanticValidationError):
            Step(id=bad, kind=StepKind.TOOL_CALL, tool="noop")
