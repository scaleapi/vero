"""Tests for the single source of truth error taxonomy."""

from vero.evaluation.error_taxonomy import (
    ErrorCategory,
    classify_case,
    classify_signal,
    policy,
)


def test_classify_signal_recognizes_infrastructure_categories():
    assert classify_signal("openai.RateLimitError") is ErrorCategory.TRANSIENT_INFRA
    assert classify_signal("APITimeoutError") is ErrorCategory.TRANSIENT_INFRA
    assert classify_signal("ConnectionError") is ErrorCategory.TRANSIENT_INFRA
    assert classify_signal("AuthenticationError") is ErrorCategory.AUTH_FAILURE
    assert classify_signal("PermissionDeniedError") is ErrorCategory.AUTH_FAILURE
    assert (
        classify_signal("insufficient_quota")
        is ErrorCategory.INFERENCE_BUDGET_EXHAUSTED
    )


def test_classify_signal_leaves_task_failures_unclassified():
    # The benign "produced no answer" marker and a candidate's own bug are not
    # infrastructure — the case-level classifier turns both into task failures.
    assert classify_signal("no_rewards_recorded") is None
    assert classify_signal("ValueError") is None
    assert classify_signal("") is None


def test_classify_case_precedence_and_defaults():
    assert classify_case([]) is ErrorCategory.TASK_FAILURE
    assert classify_case(["no_rewards_recorded"]) is ErrorCategory.TASK_FAILURE
    assert classify_case(["KeyError"]) is ErrorCategory.TASK_FAILURE
    assert classify_case(["openai.RateLimitError"]) is ErrorCategory.TRANSIENT_INFRA
    # Auth (terminating) outranks a transient rate limit seen on another attempt.
    assert (
        classify_case(["openai.RateLimitError", "AuthenticationError"])
        is ErrorCategory.AUTH_FAILURE
    )


def test_policy_encodes_the_intended_treatment():
    budget = policy(ErrorCategory.INFERENCE_BUDGET_EXHAUSTED)
    assert budget.terminating and not budget.retryable
    assert not budget.counts_toward_invalidity

    eval_budget = policy(ErrorCategory.EVALUATION_BUDGET_EXHAUSTED)
    assert not eval_budget.terminating and not eval_budget.retryable

    auth = policy(ErrorCategory.AUTH_FAILURE)
    assert auth.terminating and not auth.retryable

    transient = policy(ErrorCategory.TRANSIENT_INFRA)
    assert transient.retryable and transient.counts_toward_invalidity
    assert not transient.is_informative_sample

    task = policy(ErrorCategory.TASK_FAILURE)
    assert task.is_informative_sample
    assert not task.counts_toward_invalidity and not task.terminating


def test_classify_signal_recognizes_harness_environment_loss():
    # Modal sandbox lifecycle failures and a missing held-out tests fixture are
    # infrastructure the candidate did not cause, not an informative zero.
    assert classify_signal("AddTestsDirError") is ErrorCategory.TRANSIENT_INFRA
    assert (
        classify_signal("Failed to add tests directory to environment.")
        is ErrorCategory.TRANSIENT_INFRA
    )
    assert (
        classify_signal("The Sandbox is unavailable. It may have already shut down.")
        is ErrorCategory.TRANSIENT_INFRA
    )
    assert (
        classify_signal("Modal Sandbox with container ID ta-01 not found.")
        is ErrorCategory.TRANSIENT_INFRA
    )
    assert (
        classify_signal("FetchSpec failed: loading container: file does not exist")
        is ErrorCategory.TRANSIENT_INFRA
    )


def test_classify_case_treats_environment_loss_as_infra_not_task_failure():
    # Regression for the swe-atlas-qna flat-zero: the majority of cases died on
    # Modal sandbox / tests-dir provisioning yet were masked as scoreable task
    # failures at 0.0. They must be excluded infrastructure, not informative.
    assert classify_case(["AddTestsDirError"]) is ErrorCategory.TRANSIENT_INFRA
    assert (
        classify_case(
            ["NotFoundError", "Modal Sandbox with container ID ta-01 not found."]
        )
        is ErrorCategory.TRANSIENT_INFRA
    )
    # A candidate's own bug is still an informative task failure, unchanged.
    assert classify_case(["ValueError"]) is ErrorCategory.TASK_FAILURE
