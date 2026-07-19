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


class DarwinGodelStrategy:
    """Open-ended archive evolution in the style of the Darwin Gödel Machine.

    Where ``EvolutionaryStrategy`` runs a tournament over a fittest-``K``
    population, DGM keeps the **entire archive** as candidate parents and samples
    parent ``p`` with probability proportional to ``score(p) / (1 + children(p))``
    — every archived agent keeps a non-zero probability, so lower-performing
    nodes remain reachable as *stepping stones* (many innovations traverse them).

    The intended pairing is a self-application producer: each offspring is the
    parent's own harness modifying itself (analyze its ``.vero`` trace corpus,
    implement one improvement), which is the self-referential loop the original
    Gödel Machine wanted but validated empirically rather than by proof.

    ``base_weight`` is the floor performance given to not-yet-scored or infeasible
    agents so brand-new lineages are still explorable. ``evaluation_set`` names
    the selection set to rank on when several are visible (held-out target scores
    never appear in ``context.evaluations``, so they cannot leak into selection).
    """

    def __init__(
        self,
        *,
        objective: ObjectiveSpec,
        producer_id: str = "self",
        num_offspring: int = 2,
        instruction: str | None = None,
        evaluation_set: str | None = None,
        base_weight: float = 0.2,
        seed: int | None = None,
    ):
        if num_offspring < 1:
            raise ValueError("num_offspring must be >= 1")
        if base_weight <= 0:
            raise ValueError("base_weight must be > 0 so every agent stays reachable")
        self.objective = objective
        self.producer_id = producer_id
        self.num_offspring = num_offspring
        self.instruction = instruction
        self.evaluation_set = evaluation_set
        self.base_weight = base_weight
        self._rng = random.Random(seed)
        self._score: dict[str, float] = {}
        self._children: dict[str, int] = {}
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
                    "strategy": "darwin-godel",
                    "generation": self._generation,
                    "parent_id": parent_id,
                },
            )
            for parent_id in parents
        ]
        for parent_id in parents:
            self._children[parent_id] = self._children.get(parent_id, 0) + 1
        self._generation += 1
        return proposals

    def _ingest(self, context: OptimizationContext) -> None:
        """Record each candidate's best feasible objective as its performance."""
        for view in context.evaluations:
            parsed = _record_objective(view)
            if parsed is None:
                continue
            candidate_id, set_name, objective = parsed
            if self.evaluation_set is not None and set_name != self.evaluation_set:
                continue
            if not objective.feasible or objective.value is None:
                continue
            quality = (
                objective.value
                if self.objective.direction == "maximize"
                else -objective.value
            )
            previous = self._score.get(candidate_id)
            if previous is None or quality > previous:
                self._score[candidate_id] = quality

    def _weight(self, candidate_id: str) -> float:
        """DGM selection weight: performance, damped by how much it's been explored."""
        performance = self._score.get(candidate_id)
        quality = self.base_weight if performance is None else max(self.base_weight, performance)
        return quality / (1 + self._children.get(candidate_id, 0))

    def _select_parents(self, context: OptimizationContext) -> list[str]:
        archive = list(context.candidates)
        if not archive:
            seed_parent = (
                context.best.id if context.best is not None else context.baseline.id
            )
            return [seed_parent] * self.num_offspring
        weights = [self._weight(candidate_id) for candidate_id in archive]
        if sum(weights) <= 0:
            weights = [1.0] * len(archive)
        # sample WITH replacement: a strong lineage may spawn several children a round
        return self._rng.choices(archive, weights=weights, k=self.num_offspring)


class ObjectiveSelectionPolicy:
    """Select the best feasible value of the configured objective."""

    def select(
        self,
        records: Sequence[EvaluationRecord],
        objective: ObjectiveSpec,
    ) -> EvaluationRecord | None:
        compatible = [record for record in records if record.objective_spec == objective]
        return select_best_evaluation(compatible)
