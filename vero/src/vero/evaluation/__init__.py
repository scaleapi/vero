"""Evaluation: the Evaluator (checkout + run) and the EvaluationEngine that
orchestrates it (sample resolution + budget metering). The in-process
ExperimentRunnerTool and the Harbor eval sidecar are both frontends over the engine.
"""

from vero.evaluation.engine import EvalRequest, EvaluationEngine
from vero.evaluation.evaluator import (
    Evaluator,
    isolate_project,
    run_evaluation,
)

__all__ = [
    "Evaluator",
    "isolate_project",
    "run_evaluation",
    "EvaluationEngine",
    "EvalRequest",
]
