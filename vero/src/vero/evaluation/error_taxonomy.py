"""Single source of truth for evaluation error categories and their policy.

Historically the pipeline used one overloaded channel — an HTTP 429 became an
``infrastructure_failure`` diagnostic became a ``failure_value`` reward — to
represent four genuinely different conditions: inference-budget exhaustion, a
transient rate limit, infrastructure that dropped cases, and a legitimate task
failure. Because they were indistinguishable, the system retried permanent
conditions, penalized candidates for infrastructure they did not cause, and
recorded "nothing shipped" as a real score of zero.

Every layer that classifies or reacts to an error consults this module instead
of maintaining its own vocabulary:

- the Harbor backend's sub-run classification (``harbor/backend.py``),
- the evaluation engine's retry/refund contract (``evaluation/engine.py``),
- the optimizer agent's client retry policy (``agents/vero.py``), and
- the error-rate / invalidity logic (``evaluation/evaluator.py``).

Budget exhaustion cannot be recovered from the exception type once the in-
container inference client collapses the gateway's distinct 429 body code into
a generic rate-limit error. It is therefore detected out of band from the
gateway's own usage ledger and assigned directly, not inferred from the
exception here. This module only classifies the signals that *do* survive as
exception type names or messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(str, Enum):
    """The canonical categories every layer agrees on."""

    #: The inference gateway's per-scope token/request budget ran out. A
    #: permanent condition for this run: stop loudly, never retry or refund.
    INFERENCE_BUDGET_EXHAUSTED = "inference_budget_exhausted"
    #: The optimizer agent spent its evaluation budget. Expected and benign:
    #: surfaced to the agent as a tool result; never terminates a run.
    EVALUATION_BUDGET_EXHAUSTED = "evaluation_budget_exhausted"
    #: Authentication/authorization failed (e.g. a wrong or cycled key). Does
    #: not heal on retry: terminate loudly.
    AUTH_FAILURE = "auth_failure"
    #: Transient external infrastructure (rate limit, timeout, connection, 5xx,
    #: overloaded). Retryable; if it persists it renders the aggregate invalid.
    TRANSIENT_INFRA = "transient_infra"
    #: The candidate's harness produced no answer, gave up, or crashed on its
    #: own. An informative sample: scored at the failure value, not infra.
    TASK_FAILURE = "task_failure"


@dataclass(frozen=True)
class CategoryPolicy:
    """How the pipeline must treat a category, in one place."""

    #: May a retry plausibly succeed? Drives client, Harbor, and infra retries.
    retryable: bool
    #: Should the whole run stop immediately rather than continue?
    terminating: bool
    #: Does a case in this category count toward the infrastructure-loss
    #: fraction that can render the aggregate score invalid?
    counts_toward_invalidity: bool
    #: Is a case in this category a real, scoreable sample (contributes to the
    #: aggregate at the failure value) rather than excluded infrastructure noise?
    is_informative_sample: bool
    #: Stable diagnostic code emitted for this category.
    diagnostic_code: str


_POLICIES: dict[ErrorCategory, CategoryPolicy] = {
    ErrorCategory.INFERENCE_BUDGET_EXHAUSTED: CategoryPolicy(
        retryable=False,
        terminating=True,
        counts_toward_invalidity=False,
        is_informative_sample=False,
        diagnostic_code="inference_budget_exhausted",
    ),
    ErrorCategory.EVALUATION_BUDGET_EXHAUSTED: CategoryPolicy(
        retryable=False,
        terminating=False,
        counts_toward_invalidity=False,
        is_informative_sample=False,
        diagnostic_code="evaluation_budget_exhausted",
    ),
    ErrorCategory.AUTH_FAILURE: CategoryPolicy(
        retryable=False,
        terminating=True,
        counts_toward_invalidity=False,
        is_informative_sample=False,
        diagnostic_code="auth_failure",
    ),
    ErrorCategory.TRANSIENT_INFRA: CategoryPolicy(
        retryable=True,
        terminating=False,
        counts_toward_invalidity=True,
        is_informative_sample=False,
        diagnostic_code="transient_infrastructure",
    ),
    ErrorCategory.TASK_FAILURE: CategoryPolicy(
        retryable=False,
        terminating=False,
        counts_toward_invalidity=False,
        is_informative_sample=True,
        diagnostic_code="task_failure",
    ),
}


def policy(category: ErrorCategory) -> CategoryPolicy:
    """Return the policy for a category."""
    return _POLICIES[category]


#: The literal exception-type key the Harbor backend records when the candidate
#: produced no answer without raising (see ``harbor/backend.py``).
NO_REWARD_SIGNAL = "no_rewards_recorded"


# Ordered most-specific first. Each regex is matched, case-insensitively,
# against a combined "<exception type name> <message>" string. Budget-like
# credit/quota exhaustion from an upstream provider is treated as an inference
# budget condition; authentication/permission never retries; the remainder are
# transient infrastructure. Anything unmatched is deliberately NOT infra — a
# candidate whose own harness crashes is a task failure, an informative low
# score, not an infrastructure error.
_SIGNAL_PATTERNS: list[tuple[re.Pattern[str], ErrorCategory]] = [
    (
        re.compile(r"quota|insufficient.?credits|billing", re.IGNORECASE),
        ErrorCategory.INFERENCE_BUDGET_EXHAUSTED,
    ),
    (
        re.compile(
            r"authentication|unauthorized|invalid.?api.?key|permission|forbidden",
            re.IGNORECASE,
        ),
        ErrorCategory.AUTH_FAILURE,
    ),
    (
        re.compile(
            r"rate.?limit|too.?many.?requests|time(?:d.?)?out|connection|"
            r"service.?unavailable|internal.?server|overloaded|"
            r"bad.?gateway|gateway.?time(?:d.?)?out",
            re.IGNORECASE,
        ),
        ErrorCategory.TRANSIENT_INFRA,
    ),
]


def classify_signal(signal: str) -> ErrorCategory | None:
    """Classify one exception-type name or message string.

    Returns ``None`` for the benign no-answer signal and for anything
    unrecognized; callers treat both as :attr:`ErrorCategory.TASK_FAILURE` at
    the case level. Recognized infrastructure signals return their category.
    """
    if not signal or signal == NO_REWARD_SIGNAL:
        return None
    for pattern, category in _SIGNAL_PATTERNS:
        if pattern.search(signal):
            return category
    return None


def classify_case(signals: list[str]) -> ErrorCategory:
    """Classify a case from the exception signals its attempts recorded.

    Precedence follows severity: a terminating condition (budget, then auth)
    anywhere wins, then transient infrastructure, otherwise the case is a task
    failure. An empty signal set — the candidate simply produced no answer —
    is a task failure.
    """
    categories = {classify_signal(signal) for signal in signals}
    categories.discard(None)
    for category in (
        ErrorCategory.INFERENCE_BUDGET_EXHAUSTED,
        ErrorCategory.AUTH_FAILURE,
        ErrorCategory.TRANSIENT_INFRA,
    ):
        if category in categories:
            return category
    return ErrorCategory.TASK_FAILURE
