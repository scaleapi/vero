"""Strategy-driven optimization of versioned programs."""

from vero.optimization.command import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
)
from vero.optimization.models import (
    CandidateChange,
    CandidateProposal,
    OptimizationContext,
    OptimizationResult,
)
from vero.optimization.optimizer import Optimizer
from vero.optimization.protocols import (
    CandidateProducer,
    OptimizationStrategy,
    SelectionPolicy,
)
from vero.optimization.strategy import ObjectiveSelectionPolicy, SequentialStrategy

__all__ = [
    "CandidateChange",
    "CandidateProducer",
    "CandidateProposal",
    "CommandCandidateProducer",
    "CommandCandidateProducerConfig",
    "ObjectiveSelectionPolicy",
    "OptimizationContext",
    "OptimizationResult",
    "OptimizationStrategy",
    "Optimizer",
    "SelectionPolicy",
    "SequentialStrategy",
]
