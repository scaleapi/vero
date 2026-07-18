"""Built-in optimization and selection strategies."""

from __future__ import annotations

import random
from collections.abc import Sequence

from vero.evaluation import (
    EvaluationRecord,
    EvaluationSummary,
    ObjectiveResult,
    ObjectiveSpec,
    select_best_evaluation,
)
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


def _record_objective(
    view: object,
) -> tuple[str, str | None, ObjectiveResult] | None:
    """Extract (candidate_id, evaluation_set_name, objective) from a projected view.

    Handles full ``EvaluationRecord`` and aggregate ``EvaluationSummary`` views;
    returns ``None`` for target/none views (``EvaluationAcknowledgement``) that
    carry no objective — so the strategy never sees held-out scores.
    """
    if isinstance(view, EvaluationRecord):
        if view.objective is None:
            return None
        return view.request.candidate.id, view.request.evaluation_set.name, view.objective
    if isinstance(view, EvaluationSummary):
        if view.objective is None:
            return None
        return view.candidate_id, view.evaluation_set.name, view.objective
    return None


class EvolutionaryStrategy:
    """A population-based strategy: select parents, emit N mutated offspring.

    Each round it ingests the disclosed evaluations into a fitness archive
    (direction-aware via ``objective``), keeps the fittest ``population_size``
    candidates, and emits ``num_offspring`` proposals whose parents are chosen by
    tournament selection. Before any feasible candidate exists it seeds offspring
    from the current best (or the baseline). This is the native runner's
    distinctive capability over Harbor's single-agent-as-optimizer.

    Only mutation (parent + instruction) is implemented; crossover (combining two
    parents) is a follow-on — it needs the producer/instruction to reference a
    second candidate's contents, exposed through the read-only ``.vero`` tree.

    ``evaluation_set`` should name the selection set when multiple sets are
    visible, so fitness is ranked on one consistent metric; left ``None`` it
    ranks on all disclosed evaluations (correct for single-set setups).
    """

    def __init__(
        self,
        *,
        objective: ObjectiveSpec,
        producer_id: str = "default",
        population_size: int = 8,
        num_offspring: int = 4,
        tournament_size: int = 3,
        instruction: str | None = None,
        evaluation_set: str | None = None,
        seed: int | None = None,
    ):
        if population_size < 1:
            raise ValueError("population_size must be >= 1")
        if num_offspring < 1:
            raise ValueError("num_offspring must be >= 1")
        if tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        self.objective = objective
        self.producer_id = producer_id
        self.population_size = population_size
        self.num_offspring = num_offspring
        self.tournament_size = tournament_size
        self.instruction = instruction
        self.evaluation_set = evaluation_set
        self._rng = random.Random(seed)
        self._fitness: dict[str, float] = {}
        self._generation = 0

    async def propose(self, context: OptimizationContext) -> Sequence[CandidateProposal]:
        self._ingest(context)
        parents = self._select_parents(context)
        proposals = [
            CandidateProposal(
                producer_id=self.producer_id,
                parent_id=parent_id,
                instruction=self.instruction,
                metadata={
                    "strategy": "evolutionary",
                    "generation": self._generation,
                    "parent_id": parent_id,
                },
            )
            for parent_id in parents
        ]
        self._generation += 1
        return proposals

    def _to_fitness(self, objective: ObjectiveResult) -> float:
        """Map an objective result to an internal fitness where higher is better."""
        if not objective.feasible or objective.value is None:
            return float("-inf")
        return (
            objective.value
            if self.objective.direction == "maximize"
            else -objective.value
        )

    def _ingest(self, context: OptimizationContext) -> None:
        for view in context.evaluations:
            parsed = _record_objective(view)
            if parsed is None:
                continue
            candidate_id, set_name, objective = parsed
            if self.evaluation_set is not None and set_name != self.evaluation_set:
                continue
            fitness = self._to_fitness(objective)
            previous = self._fitness.get(candidate_id)
            if previous is None or fitness > previous:
                self._fitness[candidate_id] = fitness

    def _population(self, context: OptimizationContext) -> list[str]:
        scored = [
            (candidate_id, fitness)
            for candidate_id, fitness in self._fitness.items()
            if candidate_id in context.candidates and fitness > float("-inf")
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [candidate_id for candidate_id, _ in scored[: self.population_size]]

    def _select_parents(self, context: OptimizationContext) -> list[str]:
        population = self._population(context)
        if not population:
            seed_parent = (
                context.best.id if context.best is not None else context.baseline.id
            )
            return [seed_parent] * self.num_offspring
        parents: list[str] = []
        for _ in range(self.num_offspring):
            k = min(self.tournament_size, len(population))
            contenders = self._rng.sample(population, k)
            parents.append(max(contenders, key=lambda cid: self._fitness[cid]))
        return parents


class ObjectiveSelectionPolicy:
    """Select the best feasible value of the configured objective."""

    def select(
        self,
        records: Sequence[EvaluationRecord],
        objective: ObjectiveSpec,
    ) -> EvaluationRecord | None:
        compatible = [record for record in records if record.objective_spec == objective]
        return select_best_evaluation(compatible)
