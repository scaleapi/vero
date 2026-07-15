"""VeRO's public program-optimization API."""

from vero.candidate import Candidate
from vero.evaluation import (
    EvaluationBackend,
    EvaluationRecord,
    EvaluationReport,
    EvaluationSet,
    ObjectiveSpec,
)
from vero.optimization import CandidateProducer, OptimizationStrategy, Optimizer
from vero.runtime import (
    OptimizationSession,
    create_local_optimization_session,
    create_optimization_session,
)

__all__ = [
    "Candidate",
    "CandidateProducer",
    "EvaluationBackend",
    "EvaluationRecord",
    "EvaluationReport",
    "EvaluationSet",
    "ObjectiveSpec",
    "OptimizationSession",
    "OptimizationStrategy",
    "Optimizer",
    "create_optimization_session",
    "create_local_optimization_session",
]
