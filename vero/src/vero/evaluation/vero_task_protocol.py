"""Backend-private aliases for the Python VeroTask subprocess protocol.

The definitions remain at their historical import path so task packages do not
break. General evaluation code must not depend on these contracts.
"""

from vero.core.evaluation import BaseEvaluationParameters, TaskParameters

__all__ = ["BaseEvaluationParameters", "TaskParameters"]
