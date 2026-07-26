"""Tests for the single source of truth error taxonomy."""

from vero.evaluation.scoring.error_taxonomy import (
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
