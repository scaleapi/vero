"""Batch-oriented optimizer over versioned program candidates."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import JsonValue

from vero.candidate import Candidate
from vero.evaluation import (
    EvaluationEngine,
    EvaluationLimits,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSet,
    ObjectiveSpec,
)
from vero.optimization.models import (
    CandidateProposal,
    OptimizationContext,
    OptimizationResult,
)
from vero.optimization.protocols import (
    CandidateProducer,
    OptimizationStrategy,
    SelectionPolicy,
)
from vero.optimization.strategy import ObjectiveSelectionPolicy
from vero.workspace import Workspace


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

    def _best(self, records: list[EvaluationRecord]) -> EvaluationRecord | None:
        return self.selection.select(records, self.objective)

    async def evaluate_candidate(self, candidate: Candidate) -> EvaluationRecord:
        return await self.engine.evaluate_record(
            backend_id=self.backend_id,
            request=EvaluationRequest(
                candidate=candidate,
                evaluation_set=self.evaluation_set,
                parameters=self.parameters,
                limits=self.limits,
                seed=self.seed,
            ),
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
    ) -> Candidate | None:
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
        before = await candidate_workspace.current_version()
        if before != parent.version:
            await candidate_workspace.destroy()
            raise ValueError(
                f"candidate workspace is at {before!r}, expected parent {parent.version!r}"
            )

        try:
            change = await producer.produce(
                proposal=proposal,
                context=context,
                workspace=candidate_workspace,
            )
            if change is None:
                await candidate_workspace.destroy()
                return None
            version = (
                await candidate_workspace.save(change.description)
                if await candidate_workspace.is_dirty()
                else await candidate_workspace.current_version()
            )
            if version == parent.version:
                await candidate_workspace.destroy()
                return None
            metadata = dict(proposal.metadata)
            metadata.update(change.metadata)
            metadata["producer_id"] = proposal.producer_id
            return Candidate(
                id=proposal.id,
                version=version,
                parent_id=parent.id,
                created_at=datetime.now(UTC),
                description=change.description,
                metadata=metadata,
            )
        except BaseException:
            await candidate_workspace.destroy()
            raise

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

        existing_baselines = [
            record
            for record in self.engine.database.evaluations.values()
            if record.request.candidate.id == baseline.id
            and record.request.candidate.version == baseline.version
            and record.backend_id == self.backend_id
            and record.request.evaluation_set == self.evaluation_set
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

        evaluations = [baseline_record]
        candidates: dict[str, Candidate] = {baseline.id: baseline}
        generated = 0
        semaphore = asyncio.Semaphore(self.max_concurrency)

        for round_number in range(self.max_rounds):
            if generated >= self.max_candidates:
                break
            best = self._best(evaluations)
            context = OptimizationContext(
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

            async def produce(proposal: CandidateProposal) -> Candidate | None:
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

            produced = [
                candidate
                for candidate in await asyncio.gather(
                    *(produce(proposal) for proposal in proposals)
                )
                if candidate is not None
            ]
            if not produced:
                break
            generated += len(produced)
            for candidate in produced:
                candidates[candidate.id] = candidate

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
