"""Transport-neutral agent frontend over the canonical evaluation engine."""

from __future__ import annotations

import asyncio
import json
import posixpath
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.candidate import Candidate
from vero.evaluation import (
    AllCases,
    CaseResourceExporter,
    DisclosureLevel,
    EvaluationAcknowledgement,
    EvaluationAuthorization,
    EvaluationBudget,
    EvaluationBudgetExceeded,
    EvaluationCancelledError,
    EvaluationDeniedError,
    EvaluationExecutionError,
    EvaluationInfrastructureError,
    EvaluationLimits,
    EvaluationModel,
    EvaluationReceipt,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationRequestError,
    EvaluationSet,
    EvaluationSummary,
    EvaluationTerminatedError,
    ObjectiveSpec,
    project_evaluation,
)
from vero.evaluation.engine import EvaluationEngine
from vero.evaluation.persistence import _atomic_write_json
from vero.harbor.transport import CandidateTransferError, CandidateTransport
from vero.runtime.context import (
    AgentContextDirectory,
    AgentDisclosureLedger,
    context_digest,
    make_evaluation_receipt,
    narrower_disclosure,
)
from vero.sandbox import LocalSandbox


class EvaluationAccessError(RuntimeError):
    """Raised when an agent requests an evaluation it may not perform."""


class SubmissionDisabledError(RuntimeError):
    """Raised when submission is disabled for this optimization task."""


class EvaluationJobNotFoundError(LookupError):
    """Raised when an agent requests an unknown evaluation job."""


class SidecarEvaluationPolicy(EvaluationModel):
    """Agent access to one backend-owned evaluation-set partition."""

    backend_id: str
    evaluation_set_name: str
    partition: str | None = None
    objective: ObjectiveSpec | None = None
    disclosure: DisclosureLevel = DisclosureLevel.AGGREGATE
    expose_case_resources: bool = False
    agent_evaluable: bool = True
    # k-anonymity floor: aggregate subset evals must cover >= this many cases so
    # a single held-out label can't be read off one case at a time. Default 5
    # (not 1) so a build that omits it is safe rather than unfloored.
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
    def validate_parameters(self) -> SidecarEvaluationPolicy:
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


class EvaluationJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SidecarEvaluationJob(EvaluationModel):
    """Durable agent-facing lifecycle for one sidecar evaluation request."""

    job_id: str
    status: EvaluationJobStatus
    backend_id: str
    evaluation_set: EvaluationSet
    version: str | None = None
    evaluation_id: str | None = None
    receipt: EvaluationReceipt | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> SidecarEvaluationJob:
        terminal = self.status in {
            EvaluationJobStatus.COMPLETE,
            EvaluationJobStatus.FAILED,
            EvaluationJobStatus.CANCELLED,
        }
        if terminal != (self.completed_at is not None):
            raise ValueError("only terminal evaluation jobs have completed_at")
        if self.status == EvaluationJobStatus.COMPLETE and self.receipt is None:
            raise ValueError("complete evaluation jobs require a receipt")
        if self.receipt is not None:
            if self.status != EvaluationJobStatus.COMPLETE:
                raise ValueError("only complete evaluation jobs may have a receipt")
            if self.evaluation_id != self.receipt.evaluation_id:
                raise ValueError("evaluation job and receipt identities must match")
        if self.error is not None and self.status not in {
            EvaluationJobStatus.FAILED,
            EvaluationJobStatus.CANCELLED,
        }:
            raise ValueError("only failed or cancelled jobs may have an error")
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
    inference_usage: dict[str, JsonValue] | None = None
    evaluation_jobs: list[SidecarEvaluationJob] = Field(default_factory=list)


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
        access_policies: list[SidecarEvaluationPolicy],
        agent_volume: Path | None = None,
        admin_volume: Path | None = None,
        inference_usage_path: Path | None = None,
        inference_limits: dict[str, dict[str, JsonValue]] | None = None,
        submit_enabled: bool = False,
        disclose_budget: bool = True,
    ):
        self.engine = engine
        self.candidate_transport = candidate_transport
        self.agent_volume = Path(agent_volume) if agent_volume is not None else None
        self.admin_volume = Path(admin_volume) if admin_volume is not None else None
        self.inference_usage_path = (
            Path(inference_usage_path) if inference_usage_path is not None else None
        )
        self.inference_limits = inference_limits or {}
        self.submit_enabled = submit_enabled
        self.disclose_budget = disclose_budget
        self._context_lock = asyncio.Lock()
        self._context_initialized = False
        self._evaluation_jobs_dir = (
            self.engine.evaluator.session_dir / "evaluation-jobs"
        )
        self._evaluation_jobs_dir.mkdir(parents=True, exist_ok=True)
        self._evaluation_jobs: dict[str, SidecarEvaluationJob] = {}
        self._evaluation_job_tasks: dict[str, asyncio.Task[None]] = {}
        self._load_evaluation_jobs()
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
        self._policies: dict[tuple[str, str, str | None], SidecarEvaluationPolicy] = {}
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

    def _load_evaluation_jobs(self) -> None:
        """Restore terminal jobs and mark interrupted in-flight jobs explicitly."""

        for path in sorted(self._evaluation_jobs_dir.glob("*.json")):
            try:
                job = SidecarEvaluationJob.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if job.status in {
                EvaluationJobStatus.QUEUED,
                EvaluationJobStatus.RUNNING,
            }:
                job = job.model_copy(
                    update={
                        "status": EvaluationJobStatus.FAILED,
                        "error": "evaluation job was interrupted by a sidecar restart",
                        "completed_at": datetime.now(UTC),
                    }
                )
                _atomic_write_json(path, job.model_dump(mode="json"))
            self._evaluation_jobs[job.job_id] = job

    async def _save_evaluation_job(self, job: SidecarEvaluationJob) -> None:
        self._evaluation_jobs[job.job_id] = job
        await asyncio.to_thread(
            _atomic_write_json,
            self._evaluation_jobs_dir / f"{job.job_id}.json",
            job.model_dump(mode="json"),
        )

    async def _update_evaluation_job(
        self,
        job_id: str,
        **updates: object,
    ) -> SidecarEvaluationJob:
        job = self._evaluation_jobs[job_id].model_copy(update=updates)
        job = SidecarEvaluationJob.model_validate(job)
        await self._save_evaluation_job(job)
        return job

    def evaluation_job(self, job_id: str) -> SidecarEvaluationJob:
        try:
            return self._evaluation_jobs[job_id]
        except KeyError as error:
            raise EvaluationJobNotFoundError(
                f"unknown evaluation job {job_id!r}"
            ) from error

    async def start_evaluation_job(
        self,
        request: SidecarEvaluationRequest,
    ) -> SidecarEvaluationJob:
        """Admit an evaluation and return without waiting for its result."""

        job = SidecarEvaluationJob(
            job_id=str(uuid4()),
            status=EvaluationJobStatus.QUEUED,
            backend_id=request.backend_id,
            evaluation_set=request.evaluation_set,
            version=request.version,
            created_at=datetime.now(UTC),
        )
        await self._save_evaluation_job(job)
        admitted = asyncio.Event()
        task = asyncio.create_task(
            self._run_detached_job(job.job_id, request, admitted),
            name=f"vero-evaluation-job-{job.job_id}",
        )
        self._evaluation_job_tasks[job.job_id] = task
        task.add_done_callback(
            lambda _task, job_id=job.job_id: self._evaluation_job_tasks.pop(
                job_id, None
            )
        )
        await admitted.wait()
        return self.evaluation_job(job.job_id)

    async def _execute_tracked_job(
        self,
        job_id: str,
        request: SidecarEvaluationRequest,
        admitted: asyncio.Event,
    ) -> SidecarEvaluationResult:
        """Run one evaluation, recording it through its job lifecycle.

        Shared by the synchronous and detached entry points so every
        evaluation is a tracked job visible in status. Re-raises on failure
        after recording the terminal state, so the synchronous caller surfaces
        the real error (mapped to HTTP by the app handlers) and the detached
        wrapper can record-and-swallow.
        """
        try:
            async with self.engine.agent_evaluation_scope():
                policy, canonical_request = await self._prepare_evaluation(request)
                await self._update_evaluation_job(
                    job_id,
                    status=EvaluationJobStatus.RUNNING,
                    version=canonical_request.candidate.version,
                )
                admitted.set()
                result = await self._evaluate_prepared(
                    backend_id=request.backend_id,
                    policy=policy,
                    request=canonical_request,
                )
                await self._update_evaluation_job(
                    job_id,
                    status=EvaluationJobStatus.COMPLETE,
                    evaluation_id=result.receipt.evaluation_id,
                    receipt=result.receipt,
                    completed_at=datetime.now(UTC),
                )
                return result
        except asyncio.CancelledError:
            admitted.set()
            await asyncio.shield(
                self._update_evaluation_job(
                    job_id,
                    status=EvaluationJobStatus.CANCELLED,
                    receipt=None,
                    error="evaluation job was cancelled",
                    completed_at=datetime.now(UTC),
                )
            )
            raise
        except Exception as error:  # the terminal record remains in agent context
            admitted.set()
            evaluation_id = getattr(error, "evaluation_id", None)
            await self._update_evaluation_job(
                job_id,
                status=EvaluationJobStatus.FAILED,
                evaluation_id=evaluation_id,
                receipt=None,
                error=self._evaluation_job_error(error),
                completed_at=datetime.now(UTC),
            )
            raise

    async def _run_detached_job(
        self,
        job_id: str,
        request: SidecarEvaluationRequest,
        admitted: asyncio.Event,
    ) -> None:
        """Detached wrapper: the terminal state is already recorded on the job
        (which the agent polls), so a non-cancellation failure is swallowed
        here to avoid an unretrieved background-task exception."""
        try:
            await self._execute_tracked_job(job_id, request, admitted)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    @staticmethod
    def _evaluation_job_error(error: Exception) -> str:
        # Terminating conditions subclass EvaluationExecutionError, so surface
        # their reason first: an out-of-budget or auth-failed run should say so.
        if isinstance(error, EvaluationTerminatedError):
            return f"evaluation terminated: {error}"
        if isinstance(error, EvaluationBudgetExceeded):
            return "evaluation budget exhausted"
        if isinstance(error, EvaluationInfrastructureError):
            return "infrastructure failure"
        if isinstance(error, (EvaluationDeniedError, EvaluationAccessError)):
            return "evaluation denied"
        if isinstance(error, EvaluationRequestError):
            return "invalid evaluation request"
        if isinstance(error, CandidateTransferError):
            return "candidate version could not be imported"
        # Unmapped failure: include the exception type (a class name, so no
        # secret leak) so the agent and operators can tell failures apart
        # instead of guessing from an opaque "evaluation failed".
        return f"evaluation failed: {type(error).__name__}"

    def _policy(
        self,
        backend_id: str,
        evaluation_set: EvaluationSet,
    ) -> SidecarEvaluationPolicy:
        key = (backend_id, evaluation_set.name, evaluation_set.partition)
        policy = self._policies.get(key)
        if policy is None or not policy.agent_evaluable:
            raise EvaluationAccessError(
                "the requested backend and evaluation set are not agent-evaluable"
            )
        return policy

    async def _enforce_aggregate_floor(
        self,
        policy: SidecarEvaluationPolicy,
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
        """Run an evaluation synchronously, recording it as a tracked job so it
        is visible in status exactly like a detached one, then return its
        result (or re-raise its failure)."""
        job = SidecarEvaluationJob(
            job_id=str(uuid4()),
            status=EvaluationJobStatus.QUEUED,
            backend_id=request.backend_id,
            evaluation_set=request.evaluation_set,
            version=request.version,
            created_at=datetime.now(UTC),
        )
        await self._save_evaluation_job(job)
        return await self._execute_tracked_job(job.job_id, request, asyncio.Event())

    async def _prepare_evaluation(
        self,
        request: SidecarEvaluationRequest,
    ) -> tuple[SidecarEvaluationPolicy, EvaluationRequest]:
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
        return policy, EvaluationRequest(
            candidate=candidate,
            evaluation_set=request.evaluation_set,
            parameters=parameters,
            limits=policy.limits or request.limits or EvaluationLimits(),
            seed=request.seed,
        )

    async def _evaluate_prepared(
        self,
        *,
        backend_id: str,
        policy: SidecarEvaluationPolicy,
        request: EvaluationRequest,
    ) -> SidecarEvaluationResult:
        try:
            result = await self.engine.evaluate(
                backend_id=backend_id,
                request=request,
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
                    # Budget is enforced regardless; disclosure is what we gate.
                    budget=budget if self.disclose_budget else None,
                )
            )
        inference_usage: dict[str, JsonValue] | None = None
        observed_scopes: dict[str, object] = {}
        if self.inference_usage_path is not None and self.inference_usage_path.exists():
            try:
                value = self.inference_usage_path.read_text(encoding="utf-8")
                parsed = json.loads(value)
                scopes = parsed.get("scopes") if isinstance(parsed, dict) else None
                if isinstance(scopes, dict):
                    observed_scopes = scopes
            except (OSError, ValueError):
                observed_scopes = {}
        if self.inference_limits and self.disclose_budget:
            inference_usage = {}
            for name, limits in self.inference_limits.items():
                observed = observed_scopes.get(name)
                usage = observed if isinstance(observed, dict) else {}
                requests = usage.get("requests", 0)
                total_tokens = usage.get("total_tokens", 0)
                max_requests = limits.get("max_requests")
                max_tokens = limits.get("max_tokens")
                inference_usage[name] = {
                    **limits,
                    **usage,
                    "remaining_requests": (
                        None
                        if max_requests is None
                        else max(0, int(max_requests) - int(requests))
                    ),
                    "remaining_tokens": (
                        None
                        if max_tokens is None
                        else max(0, int(max_tokens) - int(total_tokens))
                    ),
                }
        return SidecarStatus(
            submit_enabled=self.submit_enabled,
            evaluation_access=access,
            inference_usage=inference_usage,
            evaluation_jobs=sorted(
                self._evaluation_jobs.values(),
                key=lambda job: (job.created_at, job.job_id),
                reverse=True,
            )[:100],
        )
