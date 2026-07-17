"""Exceptions raised by canonical evaluation components."""

import asyncio


class EvaluationError(Exception):
    """Base exception for evaluation failures."""


class UnknownBackendError(EvaluationError, KeyError):
    """Raised when a caller selects an unregistered backend."""


class EvaluationDeniedError(EvaluationError, PermissionError):
    """Raised when trusted authorization denies an evaluation."""


class EvaluationBudgetExceeded(EvaluationError):
    """Raised when an evaluation budget cannot reserve a cost."""


class EvaluationRequestError(EvaluationError, ValueError):
    """Raised when a backend rejects caller-controlled request fields."""


class EvaluationExecutionError(EvaluationError):
    """Raised after an evaluation failure has been recorded."""

    def __init__(self, evaluation_id: str, message: str):
        self.evaluation_id = evaluation_id
        super().__init__(f"Evaluation {evaluation_id} failed: {message}")


class EvaluationInfrastructureError(EvaluationExecutionError):
    """Raised after transient/external infrastructure exhausted its retries."""


class EvaluationCancelledError(asyncio.CancelledError):
    """Cancellation propagated after its terminal evaluation record is stored."""

    def __init__(self, evaluation_id: str, message: str = "evaluation was cancelled"):
        self.evaluation_id = evaluation_id
        super().__init__(message)
