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


def test_classify_signal_recognizes_an_unprovisioned_upstream_model():
    """A configured-but-not-deployed model must not be blamed on the candidate.

    Left unclassified this is the worst silent failure in the taxonomy: every
    call 404s, the agent writes no answer, and each case is recorded as an
    informative task failure -- the harness scoring the candidate down for a
    model that does not exist.
    """
    for signal in (
        "DeploymentNotFound",
        "model_not_found",
        "The API deployment for this resource does not exist",
        "openai.NotFoundError: model_not_found",
    ):
        assert (
            classify_signal(signal) is ErrorCategory.UPSTREAM_MODEL_UNAVAILABLE
        ), signal

    # Deliberately narrow: a bare "does not exist" is a container/file problem,
    # not a missing model, and matching it here would swallow real infra errors.
    assert (
        classify_signal("loading container: file does not exist")
        is not ErrorCategory.UPSTREAM_MODEL_UNAVAILABLE
    )


def test_unprovisioned_model_policy_is_terminating_and_not_a_sample():
    """It is a permanent misconfiguration, so it stops the run and scores nothing."""
    p = policy(ErrorCategory.UPSTREAM_MODEL_UNAVAILABLE)
    assert p.retryable is False  # every remaining case fails identically
    assert p.terminating is True
    assert p.is_informative_sample is False  # never scored as a candidate failure
    # Unlike auth, still counts toward invalidity: if the terminating path is
    # ever bypassed, the aggregate must come out invalid rather than averaging
    # a shrinking set of survivors.
    assert p.counts_toward_invalidity is True
    assert p.diagnostic_code == "upstream_model_unavailable"


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


def test_a_missing_upstream_deployment_is_not_a_task_failure():
    """The 2026-07-25 swe-atlas-qna run's exact terminal exceptions.

    145 of its 469 cases died on a model that is not provisioned. Classified as
    a task failure, each was recorded as an informative score of 0.0: the
    harness blaming the candidate for a model that does not exist.
    """
    azure = (
        "Error code: 404 - {'error': {'type': 'invalid_request_error', "
        "'code': 'DeploymentNotFound', 'message': 'The API deployment for this "
        "resource does not exist. If you created the deployment within the "
        "last 5 minutes, please wait a moment and try again.'}}"
    )
    assert (
        classify_signal(azure) is ErrorCategory.UPSTREAM_MODEL_UNAVAILABLE
    )
    assert (
        classify_signal("model_not_found") is ErrorCategory.UPSTREAM_MODEL_UNAVAILABLE
    )
    assert (
        classify_case(["NotFoundError", azure])
        is ErrorCategory.UPSTREAM_MODEL_UNAVAILABLE
    )

    unavailable = policy(ErrorCategory.UPSTREAM_MODEL_UNAVAILABLE)
    assert not unavailable.is_informative_sample
    assert not unavailable.retryable
    assert unavailable.terminating
    assert unavailable.counts_toward_invalidity

    # It outranks a co-occurring sandbox death: the deterministic,
    # operator-fixable cause is the one worth reporting.
    assert (
        classify_case(
            [
                "NotFoundError",
                "Modal Sandbox with container ID ta-01KY not found.",
                azure,
            ]
        )
        is ErrorCategory.UPSTREAM_MODEL_UNAVAILABLE
    )


def test_missing_deployment_pattern_does_not_swallow_container_load_failures():
    """`FetchSpec failed: loading container: file does not exist` is infra.

    It is 102 of that same run's cases, and it also contains "does not exist",
    so the deployment pattern must not be written loosely enough to claim it.
    """
    fetchspec = "FetchSpec failed: loading container: file does not exist\n"
    assert classify_signal(fetchspec) is ErrorCategory.TRANSIENT_INFRA
    assert classify_case(["RuntimeError", fetchspec]) is ErrorCategory.TRANSIENT_INFRA
    assert (
        classify_case(["AddTestsDirError", "Failed to add tests directory to environment."])
        is ErrorCategory.TRANSIENT_INFRA
    )


# The exact body the upstream proxy returns for a provider-side rate limit,
# captured from runs/officeqa/gpt-5.6-sol-codex-r1 on 2026-07-29. The trailing
# operator hint is the whole problem: it mentions "quota" while the message
# itself says this is NOT a budget condition.
FIREWORKS_RATE_LIMIT = (
    "litellm.RateLimitError: RateLimitError: Fireworks_aiException - "
    '{"error":{"object":"error","type":"invalid_request_error",'
    '"code":"invalid_request_error","message":"rate limit exceeded, please try '
    'again later"}}. Received Model Group=fireworks_ai/deepseek-v4-flash\n'
    "Available Model Group Fallbacks=None [Provider-level rate limit on "
    "fireworks_ai — this is NOT your per-key proxy limit. Check "
    "#<redacted> for shared org quota alerts, or ask in #<redacted> "
    "with your key alias and the error timestamp.]"
)


def test_provider_rate_limit_is_transient_not_budget_exhaustion():
    """The regression that cost an officeqa cell and truncated its search.

    1,935 of these arrived in one run. Classified as budget exhaustion they are
    terminating and unretryable, so a 37-case evaluation whose error rate was
    2.7% was destroyed and the optimizer was told "the run cannot continue and
    must not be retried" -- while both gateway scopes sat under 10% of their
    3,000M token caps and the gateway never emitted its own 402.
    """
    assert classify_signal(FIREWORKS_RATE_LIMIT) is ErrorCategory.TRANSIENT_INFRA
    assert policy(classify_signal(FIREWORKS_RATE_LIMIT)).retryable
    assert not policy(classify_signal(FIREWORKS_RATE_LIMIT)).terminating


def test_budget_matches_provider_codes_and_not_prose_about_quota():
    """Only single-token codes may mean budget exhaustion.

    `insufficient_quota` (OpenAI's billing code) and `budget_exhausted` (this
    gateway's own 402 code) are real signals. The word "quota" inside a sentence
    is not, and neither is "billing" -- which on a benchmark made of office
    documents a candidate can emit simply by echoing an invoice.
    """
    for code in ("insufficient_quota", "budget_exhausted", "scope budget_exhausted"):
        assert classify_signal(code) is ErrorCategory.INFERENCE_BUDGET_EXHAUSTED, code

    for prose in (
        "Check #<redacted> for shared org quota alerts",
        "AssertionError: expected billing total 42 but got 41",
        "ValueError: quota column missing from the invoice sheet",
    ):
        assert (
            classify_signal(prose) is not ErrorCategory.INFERENCE_BUDGET_EXHAUSTED
        ), prose


def test_auth_matches_sdk_types_and_not_filesystem_permission_errors():
    """`PermissionDeniedError` is auth; `Permission denied` is a file.

    The installer case is not hypothetical: uv tool install writes to
    /usr/local/bin by default, which the unprivileged optimizer user cannot
    touch, and that is why the builds set UV_TOOL_BIN_DIR. Read as an auth
    failure it terminates the whole run.
    """
    for signal in ("PermissionDeniedError", "openai.PermissionDeniedError", "Unauthorized"):
        assert classify_signal(signal) is ErrorCategory.AUTH_FAILURE, signal

    for signal in (
        "Failed to install executable /usr/local/bin/mini-swe-agent: Permission denied",
        "OSError: [Errno 13] Permission denied: '/app/out.json'",
    ):
        assert classify_signal(signal) is not ErrorCategory.AUTH_FAILURE, signal
