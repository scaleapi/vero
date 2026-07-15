"""Batch-oriented optimizer over versioned program candidates."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import JsonValue

from vero.candidate import Candidate
from vero.evaluation import (
    EvaluationAcknowledgement,
    EvaluationBudget,
    EvaluationEngine,
    EvaluationLimits,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSet,
    EvaluationSummary,
    ObjectiveSpec,
)
from vero.optimization.models import (
    CandidateProposal,
    OptimizationContext,
    OptimizationResult,
)
from vero.optimization.protocols import (
    CandidateProducer,
    CandidateEvaluationGateway,
    OptimizationStrategy,
    SelectionPolicy,
)
from vero.optimization.strategy import ObjectiveSelectionPolicy
from vero.workspace import Workspace


@dataclass(frozen=True)
class _ProductionOutcome:
    """Candidates and evaluations created while executing one proposal."""

    candidate: Candidate | None
    trial_candidates: tuple[Candidate, ...]
    trial_evaluations: tuple[EvaluationRecord, ...]

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        if self.candidate is None:
            return self.trial_candidates
        return (*self.trial_candidates, self.candidate)


class _ScopedEvaluationGateway(CandidateEvaluationGateway):
    def __init__(
        self,
        *,
        optimizer: Optimizer,
        proposal: CandidateProposal,
        parent: Candidate,
        workspace: Workspace,
        round_number: int,
    ):
        self.optimizer = optimizer
        self.proposal = proposal
        self.parent = parent
        self.workspace = workspace
        self.round_number = round_number
        self._count = 0
        self._last_candidate_id = parent.id
        self._trial_candidates: list[Candidate] = []
        self._trial_evaluations: list[EvaluationRecord] = []

    @property
    def last_candidate_id(self) -> str:
        return self._last_candidate_id

    @property
    def last_candidate_version(self) -> str | None:
        if not self._trial_candidates:
            return None
        return self._trial_candidates[-1].version

    @property
    def trial_candidates(self) -> tuple[Candidate, ...]:
        return tuple(self._trial_candidates)

    @property
    def trial_evaluations(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._trial_evaluations)

    async def evaluate_current(
        self,
        *,
        description: str = "Evaluate agent checkpoint",
    ) -> EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement:
        if not description.strip():
            raise ValueError("checkpoint description must not be empty")
        version = (
            await self.workspace.save(description)
            if await self.workspace.is_dirty()
            else await self.workspace.current_version()
        )
        self._count += 1
        candidate = Candidate(
            id=f"{self.proposal.id}:trial:{self._count}",
            version=version,
            parent_id=self._last_candidate_id,
            created_at=datetime.now(UTC),
            description=description,
            metadata={
                **self.proposal.metadata,
                "producer_id": self.proposal.producer_id,
                "proposal_id": self.proposal.id,
                "round": self.round_number,
                "trial": self._count,
            },
        )
        result = await self.optimizer.engine.evaluate(
            backend_id=self.optimizer.backend_id,
            request=self.optimizer._request(candidate),
            objective_spec=self.optimizer.objective,
        )
        evaluation_id = (
            result.id if isinstance(result, EvaluationRecord) else result.evaluation_id
        )
        record = self.optimizer.engine.database.get_evaluation(evaluation_id)
        if record is None:
            raise RuntimeError(
                f"evaluation engine did not index completed evaluation {evaluation_id!r}"
            )
        self._last_candidate_id = candidate.id
        self._trial_candidates.append(candidate)
        self._trial_evaluations.append(record)
        return result

    def budget(self) -> EvaluationBudget | None:
        ledger = self.optimizer.engine.budget_ledger
        if ledger is None:
            return None
        return ledger.get(
            self.optimizer.backend_id,
            self.optimizer.evaluation_set,
        )


@dataclass
class Optimizer:
    """Schedule proposal, production, evaluation, and selection rounds."""

    workspace: Workspace
    engine: EvaluationEngine
    backend_id: str
    evaluation_set: EvaluationSet
    objective: ObjectiveSpec
    strategy: OptimizationStrategy
    producers: dict[str, CandidateProducer]
    selection: SelectionPolicy = field(default_factory=ObjectiveSelectionPolicy)
    parameters: dict[str, JsonValue] = field(default_factory=dict)
    limits: EvaluationLimits = field(default_factory=EvaluationLimits)
    seed: int | None = None
    max_candidates: int = 1
    max_rounds: int = 100
    max_concurrency: int = 1
    session_id: str | None = None

    def _best(self, records: list[EvaluationRecord]) -> EvaluationRecord | None:
        return self.selection.select(records, self.objective)

    def _request(self, candidate: Candidate) -> EvaluationRequest:
        return EvaluationRequest(
            candidate=candidate,
            evaluation_set=self.evaluation_set,
            parameters=self.parameters,
            limits=self.limits,
            seed=self.seed,
        )

    async def evaluate_candidate(self, candidate: Candidate) -> EvaluationRecord:
        return await self.engine.evaluate_record(
            backend_id=self.backend_id,
            request=self._request(candidate),
            objective_spec=self.objective,
        )

    @staticmethod
    def _workspace_name(proposal: CandidateProposal) -> str:
        digest = hashlib.sha256(proposal.id.encode()).hexdigest()[:12]
        return f"vero-candidate-{digest}"

    async def _produce_candidate(
        self,
        *,
        proposal: CandidateProposal,
        context: OptimizationContext,
        parent: Candidate,
    ) -> _ProductionOutcome:
        try:
            producer = self.producers[proposal.producer_id]
        except KeyError as error:
            raise ValueError(
                f"unknown candidate producer: {proposal.producer_id!r}"
            ) from error

        candidate_workspace = await self.workspace.copy(
            name=self._workspace_name(proposal),
            from_version=parent.version,
        )
        try:
            before = await candidate_workspace.current_version()
            if before != parent.version:
                raise ValueError(
                    f"candidate workspace is at {before!r}, "
                    f"expected parent {parent.version!r}"
                )
            evaluation = _ScopedEvaluationGateway(
                optimizer=self,
                proposal=proposal,
                parent=parent,
                workspace=candidate_workspace,
                round_number=context.round,
            )
            change = await producer.produce(
                proposal=proposal,
                context=context,
                workspace=candidate_workspace,
                evaluation=evaluation,
            )
            if change is None:
                return _ProductionOutcome(
                    candidate=None,
                    trial_candidates=evaluation.trial_candidates,
                    trial_evaluations=evaluation.trial_evaluations,
                )
            version = (
                await candidate_workspace.save(change.description)
                if await candidate_workspace.is_dirty()
                else await candidate_workspace.current_version()
            )
            if version == parent.version and not evaluation.trial_candidates:
                return _ProductionOutcome(
                    candidate=None,
                    trial_candidates=(),
                    trial_evaluations=(),
                )
            if version == evaluation.last_candidate_version:
                return _ProductionOutcome(
                    candidate=None,
                    trial_candidates=evaluation.trial_candidates,
                    trial_evaluations=evaluation.trial_evaluations,
                )
            metadata = dict(proposal.metadata)
            metadata.update(change.metadata)
            metadata["producer_id"] = proposal.producer_id
            metadata["proposal_id"] = proposal.id
            metadata["round"] = context.round
            candidate = Candidate(
                id=proposal.id,
                version=version,
                parent_id=evaluation.last_candidate_id,
                created_at=datetime.now(UTC),
                description=change.description,
                metadata=metadata,
            )
            return _ProductionOutcome(
                candidate=candidate,
                trial_candidates=evaluation.trial_candidates,
                trial_evaluations=evaluation.trial_evaluations,
            )
        finally:
            await candidate_workspace.destroy()

    async def run(
        self,
        *,
        baseline: Candidate | None = None,
        skip_baseline_evaluation: bool = False,
    ) -> OptimizationResult:
        if self.max_candidates < 0:
            raise ValueError("max_candidates must be non-negative")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if not self.producers and self.max_candidates:
            raise ValueError("at least one candidate producer is required")

        if baseline is None:
            version = await self.workspace.current_version()
            baseline = Candidate.from_version(version)

        backend_provenance = self.engine.backends.resolve(self.backend_id).provenance
        existing_baselines = [
            record
            for record in self.engine.database.evaluations.values()
            if record.request.candidate.id == baseline.id
            and record.request.candidate.version == baseline.version
            and record.backend_id == self.backend_id
            and record.request.evaluation_set == self.evaluation_set
            and record.request.parameters == self.parameters
            and record.request.limits == self.limits
            and record.request.seed == self.seed
            and record.backend == backend_provenance
            and record.objective_spec == self.objective
        ]
        if skip_baseline_evaluation:
            if not existing_baselines:
                raise ValueError(
                    "skip_baseline_evaluation requires an existing compatible baseline"
                )
            baseline_record = existing_baselines[-1]
        else:
            baseline_record = await self.evaluate_candidate(baseline)

        compatible = [
            record
            for record in self.engine.database.evaluations.values()
            if record.backend_id == self.backend_id
            and record.request.evaluation_set == self.evaluation_set
            and record.request.parameters == self.parameters
            and record.request.limits == self.limits
            and record.request.seed == self.seed
            and record.backend == backend_provenance
            and record.objective_spec == self.objective
        ]
        candidate_records = {
            record.request.candidate.id: record.request.candidate
            for record in compatible
        }
        reachable = {baseline.id}
        changed = True
        while changed:
            changed = False
            for candidate in candidate_records.values():
                if candidate.id in reachable:
                    continue
                if candidate.parent_id in reachable:
                    reachable.add(candidate.id)
                    changed = True
        evaluations = [
            record for record in compatible if record.request.candidate.id in reachable
        ]
        if baseline_record not in evaluations:
            evaluations.insert(0, baseline_record)
        evaluations.sort(key=lambda record: (record.completed_at, record.id))
        candidates: dict[str, Candidate] = {
            candidate_id: candidate
            for candidate_id, candidate in candidate_records.items()
            if candidate_id in reachable
        }
        candidates[baseline.id] = baseline
        proposal_ids = {
            str(candidate.metadata.get("proposal_id", candidate.id))
            for candidate in candidates.values()
            if candidate.id != baseline.id and "producer_id" in candidate.metadata
        }
        generated = len(proposal_ids)
        completed_rounds = [
            int(candidate.metadata["round"])
            for candidate in candidates.values()
            if isinstance(candidate.metadata.get("round"), int)
        ]
        start_round = max(completed_rounds, default=generated - 1) + 1
        semaphore = asyncio.Semaphore(self.max_concurrency)

        for round_number in range(start_round, self.max_rounds):
            if generated >= self.max_candidates:
                break
            best = self._best(evaluations)
            context = OptimizationContext(
                session_id=self.session_id or self.engine.evaluator.session_dir.name,
                round=round_number,
                workspace=self.workspace,
                baseline=baseline_record,
                evaluations=tuple(evaluations),
                candidates=dict(candidates),
                best=best,
            )
            proposals = list(await self.strategy.propose(context))
            if not proposals:
                break
            remaining = self.max_candidates - generated
            proposals = proposals[:remaining]
            proposal_ids = [proposal.id for proposal in proposals]
            if len(proposal_ids) != len(set(proposal_ids)):
                raise ValueError("strategy returned duplicate proposal IDs")
            if any(proposal_id in candidates for proposal_id in proposal_ids):
                raise ValueError("strategy reused an existing candidate ID")

            async def produce(proposal: CandidateProposal) -> _ProductionOutcome:
                parent_id = proposal.parent_id or (
                    best.request.candidate.id if best is not None else baseline.id
                )
                try:
                    parent = candidates[parent_id]
                except KeyError as error:
                    raise ValueError(
                        f"proposal {proposal.id!r} names unknown parent {parent_id!r}"
                    ) from error
                async with semaphore:
                    return await self._produce_candidate(
                        proposal=proposal,
                        context=context,
                        parent=parent,
                    )

            outcomes = await asyncio.gather(
                *(produce(proposal) for proposal in proposals)
            )
            meaningful_outcomes = [
                outcome for outcome in outcomes if outcome.candidates
            ]
            if not meaningful_outcomes:
                break
            generated += len(meaningful_outcomes)
            for outcome in meaningful_outcomes:
                evaluations.extend(outcome.trial_evaluations)
                for candidate in outcome.candidates:
                    if candidate.id in candidates:
                        raise ValueError(
                            f"candidate producer reused candidate ID {candidate.id!r}"
                        )
                    candidates[candidate.id] = candidate

            produced = [
                outcome.candidate
                for outcome in meaningful_outcomes
                if outcome.candidate is not None
            ]

            async def evaluate(candidate: Candidate) -> EvaluationRecord:
                async with semaphore:
                    return await self.evaluate_candidate(candidate)

            evaluations.extend(
                await asyncio.gather(*(evaluate(candidate) for candidate in produced))
            )

        return OptimizationResult(
            baseline=baseline_record,
            evaluations=tuple(evaluations),
            candidates=tuple(candidates.values()),
            best=self._best(evaluations),
        )
