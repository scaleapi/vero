"""Exceptions raised by the program-neutral evaluation engine."""


class EvaluationError(Exception):
    """Base exception for canonical evaluation failures."""


class UnknownBackendError(EvaluationError, KeyError):
    """Raised when a caller selects an unregistered backend ID."""


class EvaluationDeniedError(EvaluationError, PermissionError):
    """Raised when trusted authorization denies an evaluation."""


class EvaluationBudgetExceeded(EvaluationError):
    """Raised when the durable evaluation budget cannot reserve a cost."""


class EvaluationExecutionError(EvaluationError):
    """Raised after a thrown evaluation failure has been recorded."""

    def __init__(self, evaluation_id: str, message: str):
        self.evaluation_id = evaluation_id
        super().__init__(f"Evaluation {evaluation_id} failed: {message}")
