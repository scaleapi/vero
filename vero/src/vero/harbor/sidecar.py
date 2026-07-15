"""Transport-neutral agent frontend over the canonical evaluation engine."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.candidate import Candidate
from vero.evaluation import (
    AllCases,
    DisclosureLevel,
    EvaluationAcknowledgement,
    EvaluationAuthorization,
    EvaluationBudget,
    EvaluationLimits,
    EvaluationModel,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSet,
    EvaluationSummary,
    ObjectiveSpec,
)
from vero.evaluation.engine import EvaluationEngine
from vero.evaluation.persistence import _atomic_write_json
from vero.harbor.transport import CandidateTransport


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
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)
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


EvaluationProjection = (
    EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement
)


class SidecarEvaluationResult(EvaluationModel):
    disclosure: DisclosureLevel
    result: EvaluationProjection
    result_path: str | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> SidecarEvaluationResult:
        expected = {
            DisclosureLevel.FULL: EvaluationRecord,
            DisclosureLevel.AGGREGATE: EvaluationSummary,
            DisclosureLevel.NONE: EvaluationAcknowledgement,
        }[self.disclosure]
        if not isinstance(self.result, expected):
            raise ValueError(
                f"{self.disclosure.value} disclosure requires {expected.__name__}"
            )
        return self


class EvaluationAccessStatus(EvaluationModel):
    backend_id: str
    evaluation_set_name: str
    partition: str | None
    disclosure: DisclosureLevel
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
        self._policies: dict[
            tuple[str, str, str | None], EvaluationAccessPolicy
        ] = {}
        for policy in access_policies:
            if policy.key in self._policies:
                raise ValueError(f"duplicate evaluation access policy for {policy.key!r}")
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

    def _write_projection(
        self,
        result: EvaluationProjection,
    ) -> str | None:
        if self.agent_volume is None:
            return None
        evaluation_id = (
            result.id if isinstance(result, EvaluationRecord) else result.evaluation_id
        )
        destination = self.agent_volume / "results" / f"{evaluation_id}.json"
        _atomic_write_json(destination, result.model_dump(mode="json"))
        return str(destination)

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
        await self._enforce_aggregate_floor(policy, request.evaluation_set)
        candidate = await self.candidate_transport.import_candidate(request.version)
        parameters = {**policy.parameters, **request.parameters}
        canonical_request = EvaluationRequest(
            candidate=candidate,
            evaluation_set=request.evaluation_set,
            parameters=parameters,
            limits=policy.limits or request.limits,
            seed=request.seed,
        )
        result = await self.engine.evaluate(
            backend_id=request.backend_id,
            request=canonical_request,
            objective_spec=policy.objective,
            authorization=EvaluationAuthorization(
                may_evaluate=True,
                meter_budget=True,
                disclosure=policy.disclosure,
            ),
        )
        return SidecarEvaluationResult(
            disclosure=policy.disclosure,
            result=result,
            result_path=self._write_projection(result),
        )

    async def submit(self, version: str | None = None) -> Submission:
        if not self.submit_enabled:
            raise SubmissionDisabledError("candidate submission is disabled")
        candidate = await self.candidate_transport.import_candidate(version)
        submission = Submission(candidate=candidate)
        if self.admin_volume is not None:
            _atomic_write_json(
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
