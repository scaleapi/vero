"""Exceptions raised by canonical evaluation components."""


class EvaluationError(Exception):
    """Base exception for evaluation failures."""


class UnknownBackendError(EvaluationError, KeyError):
    """Raised when a caller selects an unregistered backend."""


class EvaluationDeniedError(EvaluationError, PermissionError):
    """Raised when trusted authorization denies an evaluation."""


class EvaluationBudgetExceeded(EvaluationError):
    """Raised when an evaluation budget cannot reserve a cost."""


class EvaluationExecutionError(EvaluationError):
    """Raised after an evaluation failure has been recorded."""

    def __init__(self, evaluation_id: str, message: str):
        self.evaluation_id = evaluation_id
        super().__init__(f"Evaluation {evaluation_id} failed: {message}")
