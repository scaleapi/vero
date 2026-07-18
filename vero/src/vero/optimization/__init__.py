"""Strategy-driven optimization of versioned programs."""

from vero.optimization.command import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
)
from vero.optimization.models import (
    CandidateChange,
    CandidateProductionContext,
    CandidateProposal,
    GenerationOutcome,
    OptimizationContext,
    OptimizationResult,
)
from vero.optimization.optimizer import Optimizer
from vero.optimization.protocols import (
    CandidateEvaluationGateway,
    CandidateProducer,
    GenerationBackend,
    OptimizationStrategy,
    SelectionPolicy,
)
from vero.optimization.strategy import (
    EvolutionaryStrategy,
    ObjectiveSelectionPolicy,
    SequentialStrategy,
)

__all__ = [
    "CandidateChange",
    "CandidateProductionContext",
    "CandidateEvaluationGateway",
    "CandidateProducer",
    "CandidateProposal",
    "CommandCandidateProducer",
    "CommandCandidateProducerConfig",
    "EvolutionaryStrategy",
    "GenerationBackend",
    "GenerationOutcome",
    "ObjectiveSelectionPolicy",
    "OptimizationContext",
    "OptimizationResult",
    "OptimizationStrategy",
    "Optimizer",
    "SelectionPolicy",
    "SequentialStrategy",
]
