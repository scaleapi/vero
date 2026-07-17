"""Built-in optimization and selection strategies."""

from __future__ import annotations

from collections.abc import Sequence

from vero.evaluation import EvaluationRecord, ObjectiveSpec, select_best_evaluation
from vero.optimization.models import CandidateProposal, OptimizationContext


class SequentialStrategy:
    """Request one candidate from the same producer on every round."""

    def __init__(
        self,
        *,
        producer_id: str = "default",
        instruction: str | None = None,
    ):
        self.producer_id = producer_id
        self.instruction = instruction

    async def propose(self, context: OptimizationContext) -> Sequence[CandidateProposal]:
        parent_id = (
            context.best.id
            if context.best is not None
            else context.baseline.id
        )
        return [
            CandidateProposal(
                producer_id=self.producer_id,
                parent_id=parent_id,
                instruction=self.instruction,
            )
        ]


class ObjectiveSelectionPolicy:
    """Select the best feasible value of the configured objective."""

    def select(
        self,
        records: Sequence[EvaluationRecord],
        objective: ObjectiveSpec,
    ) -> EvaluationRecord | None:
        compatible = [record for record in records if record.objective_spec == objective]
        return select_best_evaluation(compatible)
