"""Transport-neutral agent frontend over the canonical evaluation engine."""

from __future__ import annotations

import asyncio
import posixpath
from pathlib import Path

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.candidate import Candidate
from vero.evaluation import (
    AllCases,
    CaseResourceExporter,
    DisclosureLevel,
    EvaluationAcknowledgement,
    EvaluationAuthorization,
    EvaluationBudget,
    EvaluationCancelledError,
    EvaluationExecutionError,
    EvaluationLimits,
    EvaluationModel,
    EvaluationRecord,
    EvaluationReceipt,
    EvaluationRequest,
    EvaluationSet,
    EvaluationSummary,
    ObjectiveSpec,
    project_evaluation,
)
from vero.evaluation.engine import EvaluationEngine
from vero.evaluation.persistence import _atomic_write_json
from vero.runtime.context import (
    AgentContextDirectory,
    AgentDisclosureLedger,
    context_digest,
    make_evaluation_receipt,
    narrower_disclosure,
)
from vero.harbor.transport import CandidateTransport
from vero.sandbox import LocalSandbox


class EvaluationAccessError(RuntimeError):
    """Raised when an agent requests an evaluation it may not perform."""


class SubmissionDisabledError(RuntimeError):
    """Raised when submission is disabled for this optimization task."""


class EvaluationAccessPolicy(EvaluationModel):
    """Agent access to one backend-owned evaluation-set partition."""

    backend_id: str
    evaluation_set_name: str
    partition: str | None = None
    objective: ObjectiveSpec | None = None
    disclosure: DisclosureLevel = DisclosureLevel.AGGREGATE
    expose_case_resources: bool = False
    agent_evaluable: bool = True
    min_aggregate_cases: int = Field(default=5, ge=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    allowed_parameters: list[str] = Field(default_factory=list)
    limits: EvaluationLimits | None = None

    @field_validator("backend_id", "evaluation_set_name")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evaluation access identity must not be empty")
        return value

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("evaluation access partition must not be empty")
        return value

    @property
    def key(self) -> tuple[str, str, str | None]:
        return (self.backend_id, self.evaluation_set_name, self.partition)

    @model_validator(mode="after")
    def validate_parameters(self) -> EvaluationAccessPolicy:
        if any(not name.strip() for name in self.parameters):
            raise ValueError("fixed evaluation parameter names must not be empty")
        if len(self.allowed_parameters) != len(set(self.allowed_parameters)):
            raise ValueError("allowed evaluation parameters must be unique")
        if any(not name.strip() for name in self.allowed_parameters):
            raise ValueError("allowed evaluation parameter names must not be empty")
        overlap = set(self.parameters) & set(self.allowed_parameters)
        if overlap:
            raise ValueError(
                "fixed and agent-controlled parameters overlap: "
                + ", ".join(sorted(overlap))
            )
        return self


class SidecarEvaluationRequest(EvaluationModel):
    """Agent request; candidate identity is established by the transport."""

    backend_id: str
    evaluation_set: EvaluationSet
    version: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    limits: EvaluationLimits | None = None
    seed: int | None = None

    @field_validator("backend_id")
    @classmethod
    def validate_backend_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("backend_id must not be empty")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("candidate version must not be empty")
        return value


EvaluationProjection = EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement


class SidecarEvaluationResult(EvaluationModel):
    disclosure: DisclosureLevel
    receipt: EvaluationReceipt

    @model_validator(mode="after")
    def validate_projection(self) -> SidecarEvaluationResult:
        if self.receipt.disclosure != self.disclosure:
            raise ValueError("sidecar disclosure must match its receipt")
        return self


class EvaluationAccessStatus(EvaluationModel):
    backend_id: str
    evaluation_set_name: str
    partition: str | None
    disclosure: DisclosureLevel
    expose_case_resources: bool
    min_aggregate_cases: int
    allowed_parameters: list[str]
    limits: EvaluationLimits | None = None
    budget: EvaluationBudget | None = None


class SidecarStatus(EvaluationModel):
    submit_enabled: bool
    evaluation_access: list[EvaluationAccessStatus]


class Submission(EvaluationModel):
    candidate: Candidate


class EvaluationSidecar:
    """Meter, evaluate, and disclose candidates imported from an agent repo.

    The sidecar supports any number of registered backends. Unknown evaluation
    sets fail closed, and every accepted request supplies an explicit canonical
    authorization to the engine.
    """

    def __init__(
        self,
        *,
        engine: EvaluationEngine,
        candidate_transport: CandidateTransport,
        access_policies: list[EvaluationAccessPolicy],
        agent_volume: Path | None = None,
        admin_volume: Path | None = None,
        submit_enabled: bool = False,
    ):
        self.engine = engine
        self.candidate_transport = candidate_transport
        self.agent_volume = Path(agent_volume) if agent_volume is not None else None
        self.admin_volume = Path(admin_volume) if admin_volume is not None else None
        self.submit_enabled = submit_enabled
        self._context_lock = asyncio.Lock()
        self._context_initialized = False
        self._disclosures = AgentDisclosureLedger(
            self.engine.evaluator.session_dir / "agent-context.json"
        )
        self._context_directory = (
            AgentContextDirectory(
                sandbox=LocalSandbox(self.agent_volume.parent),
                root=str(self.agent_volume),
                session_dir=self.engine.evaluator.session_dir,
            )
            if self.agent_volume is not None
            else None
        )
        self._policies: dict[tuple[str, str, str | None], EvaluationAccessPolicy] = {}
        for policy in access_policies:
            if policy.key in self._policies:
                raise ValueError(
                    f"duplicate evaluation access policy for {policy.key!r}"
                )
            if policy.backend_id not in engine.backends:
                raise ValueError(
                    f"access policy references unknown backend {policy.backend_id!r}"
                )
            self._policies[policy.key] = policy

    def _policy(
        self,
        backend_id: str,
        evaluation_set: EvaluationSet,
    ) -> EvaluationAccessPolicy:
        key = (backend_id, evaluation_set.name, evaluation_set.partition)
        policy = self._policies.get(key)
        if policy is None or not policy.agent_evaluable:
            raise EvaluationAccessError(
                "the requested backend and evaluation set are not agent-evaluable"
            )
        return policy

    async def _enforce_aggregate_floor(
        self,
        policy: EvaluationAccessPolicy,
        evaluation_set: EvaluationSet,
    ) -> None:
        if policy.disclosure != DisclosureLevel.AGGREGATE:
            return
        if isinstance(evaluation_set.selection, AllCases):
            return
        backend = self.engine.backends.resolve(policy.backend_id)
        cost = await backend.resolve_cost(evaluation_set)
        if cost.cases is None:
            raise EvaluationAccessError(
                "aggregate subset evaluation requires a backend with exact case costs"
            )
        if cost.cases < policy.min_aggregate_cases:
            raise EvaluationAccessError(
                f"aggregate subset evaluations must cover at least "
                f"{policy.min_aggregate_cases} cases; requested {cost.cases}"
            )

    def _visible_projections(
        self,
    ) -> list[tuple[EvaluationRecord, DisclosureLevel, EvaluationProjection]]:
        projections = []
        for evaluation_id, entry in self._disclosures.model.evaluations.items():
            record = self.engine.database.get_evaluation(evaluation_id)
            if record is None:
                continue
            policy = self._policies.get(
                (
                    record.backend_id,
                    record.request.evaluation_set.name,
                    record.request.evaluation_set.partition,
                )
            )
            if policy is None or not policy.agent_evaluable:
                continue
            disclosure = narrower_disclosure(
                entry.maximum_disclosure,
                policy.disclosure,
            )
            projections.append(
                (record, disclosure, project_evaluation(record, disclosure))
            )
        return projections

    async def _write_candidate_index(
        self,
        projections: list[
            tuple[EvaluationRecord, DisclosureLevel, EvaluationProjection]
        ],
    ) -> None:
        assert self._context_directory is not None
        root = self._context_directory.path("candidates")
        sandbox = self._context_directory.sandbox
        if await sandbox.exists(root):
            await sandbox.remove(root, recursive=True)
        await sandbox.mkdir(root)
        candidates = {
            record.request.candidate.id: record.request.candidate
            for record, _, _ in projections
        }
        index = []
        for candidate in sorted(
            candidates.values(),
            key=lambda item: (item.created_at, item.id),
        ):
            digest = context_digest(candidate.id)
            directory = self._context_directory.path("candidates", digest)
            await sandbox.mkdir(directory)
            await sandbox.write_file(
                posixpath.join(directory, "candidate.json"),
                candidate.model_dump_json(indent=2) + "\n",
            )
            index.append(
                {
                    "candidate_id": candidate.id,
                    "version": candidate.version,
                    "parent_id": candidate.parent_id,
                    "native_ref": candidate.version,
                    "metadata_path": f"{digest}/candidate.json",
                    "parent_patch_path": None,
                }
            )
        await self._context_directory.write_json(
            self._context_directory.path("candidates", "index.json"),
            {"schema_version": 1, "candidates": index},
        )

    async def _write_case_resources(self) -> None:
        assert self._context_directory is not None
        root = self._context_directory.path("cases")
        sandbox = self._context_directory.sandbox
        if await sandbox.exists(root):
            await sandbox.remove(root, recursive=True)
        await sandbox.mkdir(root)
        index = []
        for policy in self._policies.values():
            backend = self.engine.backends.resolve(policy.backend_id)
            if (
                not policy.agent_evaluable
                or not policy.expose_case_resources
                or not isinstance(backend, CaseResourceExporter)
            ):
                continue
            evaluation_set = EvaluationSet(
                name=policy.evaluation_set_name,
                partition=policy.partition,
            )
            digest = context_digest(evaluation_set.budget_key(policy.backend_id))
            resource_root = self._context_directory.path("cases", digest)
            await sandbox.mkdir(resource_root)
            await self._context_directory.write_json(
                posixpath.join(resource_root, "manifest.json"),
                {
                    "schema_version": 1,
                    "backend_id": policy.backend_id,
                    "evaluation_set": evaluation_set.model_dump(mode="json"),
                    "resources_path": "resources",
                },
            )
            resources = posixpath.join(resource_root, "resources")
            await sandbox.mkdir(resources)
            await backend.export_case_resources(
                evaluation_set=evaluation_set,
                destination=resources,
                sandbox=sandbox,
            )
            index.append(
                {
                    "backend_id": policy.backend_id,
                    "evaluation_set": evaluation_set.model_dump(mode="json"),
                    "path": digest,
                }
            )
        await self._context_directory.write_json(
            self._context_directory.path("cases", "index.json"),
            {"schema_version": 1, "case_resources": index},
        )

    async def initialize_context(self) -> None:
        if self._context_directory is None:
            return
        async with self._context_lock:
            await self._context_directory.reset()
            await self._context_directory.write_header(
                session_id=self.engine.evaluator.session_id,
                round_number=None,
                proposal_id=None,
                parent_candidate_id=None,
            )
            projections = self._visible_projections()
            await self._write_candidate_index(projections)
            await self._context_directory.write_evaluations(projections)
            await self._write_case_resources()
            self._context_initialized = True

    async def _refresh_context(self) -> None:
        if self._context_directory is None:
            return
        if not self._context_initialized:
            await self.initialize_context()
            return
        async with self._context_lock:
            projections = self._visible_projections()
            await self._write_candidate_index(projections)
            await self._context_directory.write_evaluations(projections)

    async def evaluate(
        self,
        request: SidecarEvaluationRequest,
    ) -> SidecarEvaluationResult:
        policy = self._policy(request.backend_id, request.evaluation_set)
        unknown_parameters = sorted(
            set(request.parameters) - set(policy.allowed_parameters)
        )
        if unknown_parameters:
            raise EvaluationAccessError(
                "evaluation parameters are not agent-controllable: "
                + ", ".join(unknown_parameters)
            )
        if policy.limits is not None and request.limits is not None:
            raise EvaluationAccessError(
                "evaluation limits are fixed by the access policy and cannot be "
                "overridden by the agent"
            )
        await self._enforce_aggregate_floor(policy, request.evaluation_set)
        candidate = await self.candidate_transport.import_candidate(request.version)
        parameters = {**policy.parameters, **request.parameters}
        canonical_request = EvaluationRequest(
            candidate=candidate,
            evaluation_set=request.evaluation_set,
            parameters=parameters,
            limits=policy.limits or request.limits or EvaluationLimits(),
            seed=request.seed,
        )
        try:
            result = await self.engine.evaluate(
                backend_id=request.backend_id,
                request=canonical_request,
                objective_spec=policy.objective,
                authorization=EvaluationAuthorization(
                    may_evaluate=True,
                    meter_budget=True,
                    disclosure=policy.disclosure,
                    expose_case_resources=policy.expose_case_resources,
                ),
            )
        except (EvaluationExecutionError, EvaluationCancelledError) as error:
            record = self.engine.database.get_evaluation(error.evaluation_id)
            if record is not None:
                await self._disclosures.remember(record.id, policy.disclosure)
                await asyncio.shield(self._refresh_context())
            raise
        evaluation_id = (
            result.id if isinstance(result, EvaluationRecord) else result.evaluation_id
        )
        record = self.engine.database.get_evaluation(evaluation_id)
        if record is None:
            raise RuntimeError(
                f"evaluation engine did not index completed evaluation {evaluation_id!r}"
            )
        maximum = await self._disclosures.remember(record.id, policy.disclosure)
        disclosure = narrower_disclosure(maximum, policy.disclosure)
        await self._refresh_context()
        return SidecarEvaluationResult(
            disclosure=disclosure,
            receipt=make_evaluation_receipt(record, disclosure),
        )

    async def submit(self, version: str | None = None) -> Submission:
        if not self.submit_enabled:
            raise SubmissionDisabledError("candidate submission is disabled")
        candidate = await self.candidate_transport.import_candidate(version)
        submission = Submission(candidate=candidate)
        if self.admin_volume is not None:
            await asyncio.to_thread(
                _atomic_write_json,
                self.admin_volume / "submission.json",
                submission.model_dump(mode="json"),
            )
        return submission

    def status(self) -> SidecarStatus:
        access: list[EvaluationAccessStatus] = []
        for policy in self._policies.values():
            if not policy.agent_evaluable:
                continue
            evaluation_set = EvaluationSet(
                name=policy.evaluation_set_name,
                partition=policy.partition,
            )
            budget = (
                self.engine.budget_ledger.get(policy.backend_id, evaluation_set)
                if self.engine.budget_ledger is not None
                else None
            )
            access.append(
                EvaluationAccessStatus(
                    backend_id=policy.backend_id,
                    evaluation_set_name=policy.evaluation_set_name,
                    partition=policy.partition,
                    disclosure=policy.disclosure,
                    expose_case_resources=policy.expose_case_resources,
                    min_aggregate_cases=policy.min_aggregate_cases,
                    allowed_parameters=list(policy.allowed_parameters),
                    limits=policy.limits,
                    budget=budget,
                )
            )
        return SidecarStatus(
            submit_enabled=self.submit_enabled,
            evaluation_access=access,
        )
