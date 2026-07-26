"""Authorized filesystem context exposed to optimization agents."""

from __future__ import annotations

import asyncio
import hashlib
import json
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Sequence

from pydantic import Field

from vero.candidate import Candidate
from vero.candidate_repository import CandidateRepository
from vero.evaluation import (
    CaseResourceExporter,
    DisclosureLevel,
    EvaluationAccessPolicy,
    EvaluationAcknowledgement,
    EvaluationBudget,
    EvaluationEngine,
    EvaluationPlan,
    EvaluationPrincipal,
    EvaluationReceipt,
    EvaluationRecord,
    EvaluationSet,
    EvaluationSummary,
    project_evaluation,
)
from vero.evaluation.store.persistence import _atomic_write_json
from vero.models import StrictModel
from vero.sandbox import Sandbox
from vero.workspace import Workspace

AGENT_CONTEXT_DIRECTORY = ".evals"
# Top-level names inside the context tree. The `evals` CLI, the skill, and the
# harbor instruction templates all address these paths by name.
RESULTS_SUBDIRECTORY = "results"
TASKS_SUBDIRECTORY = "tasks"
CANDIDATES_SUBDIRECTORY = "candidates"
PLAN_FILENAME = "plan.json"
_DISCLOSURE_RANK = {
    DisclosureLevel.NONE: 0,
    DisclosureLevel.AGGREGATE: 1,
    DisclosureLevel.FULL: 2,
}


def context_digest(value: str) -> str:
    """Return a path-safe identity for an arbitrary public identifier."""

    return hashlib.sha256(value.encode()).hexdigest()


def evaluation_result_path(evaluation_id: str) -> str:
    return (
        f"{AGENT_CONTEXT_DIRECTORY}/{RESULTS_SUBDIRECTORY}/"
        f"{context_digest(evaluation_id)}/evaluation.json"
    )


@dataclass(frozen=True)
class ContextPlanEntry:
    """One evaluation set as the agent may see it, resolved by the caller.

    Callers own policy filtering nuances that differ per topology (which sets
    exist, budget lookup, budget disclosure); the directory owns rendering.
    """

    backend_id: str
    backend: object
    evaluation_set: EvaluationSet
    access: EvaluationAccessPolicy
    budget: EvaluationBudget | None = None


def narrower_disclosure(
    left: DisclosureLevel,
    right: DisclosureLevel,
) -> DisclosureLevel:
    return left if _DISCLOSURE_RANK[left] <= _DISCLOSURE_RANK[right] else right


def make_evaluation_receipt(
    record: EvaluationRecord,
    disclosure: DisclosureLevel,
) -> EvaluationReceipt:
    result: EvaluationSummary | EvaluationAcknowledgement
    if disclosure == DisclosureLevel.NONE:
        projected = project_evaluation(record, DisclosureLevel.NONE)
        assert isinstance(projected, EvaluationAcknowledgement)
        result = projected
    else:
        projected = project_evaluation(record, DisclosureLevel.AGGREGATE)
        assert isinstance(projected, EvaluationSummary)
        result = projected
    return EvaluationReceipt(
        evaluation_id=record.id,
        status=record.report.status,
        disclosure=disclosure,
        result=result,
        result_path=evaluation_result_path(record.id),
    )


class AgentDisclosureEntry(StrictModel):
    evaluation_id: str
    maximum_disclosure: DisclosureLevel


class AgentDisclosureLedgerModel(StrictModel):
    schema_version: Literal[1] = 1
    evaluations: dict[str, AgentDisclosureEntry] = Field(default_factory=dict)


class AgentDisclosureLedger:
    """Durably records which evaluations may enter agent-facing context."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        if path.exists():
            self.model = AgentDisclosureLedgerModel.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        else:
            self.model = AgentDisclosureLedgerModel()

    def get(self, evaluation_id: str) -> DisclosureLevel | None:
        entry = self.model.evaluations.get(evaluation_id)
        return entry.maximum_disclosure if entry is not None else None

    async def remember(
        self,
        evaluation_id: str,
        disclosure: DisclosureLevel,
    ) -> DisclosureLevel:
        """Record first exposure and never broaden it on a later call."""

        async with self._lock:
            existing = self.model.evaluations.get(evaluation_id)
            effective = (
                disclosure
                if existing is None
                else narrower_disclosure(existing.maximum_disclosure, disclosure)
            )
            if existing is None or effective != existing.maximum_disclosure:
                self.model.evaluations[evaluation_id] = AgentDisclosureEntry(
                    evaluation_id=evaluation_id,
                    maximum_disclosure=effective,
                )
                await asyncio.to_thread(
                    _atomic_write_json,
                    self.path,
                    self.model.model_dump(mode="json"),
                )
            return effective


class AgentContextDirectory:
    """Render authorized records into one sandbox filesystem tree."""

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        root: str,
        session_dir: Path,
    ):
        self.sandbox = sandbox
        self.root = root.rstrip("/")
        self.session_dir = session_dir

    def path(self, *parts: str) -> str:
        path = PurePosixPath(self.root)
        for part in parts:
            path /= part
        return path.as_posix()

    async def write_json(self, path: str, value: object) -> None:
        await self.sandbox.write_file(
            path,
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        )

    async def reset(self) -> None:
        if await self.sandbox.exists(self.root):
            await self.unseal()
            for name in await self.sandbox.list_dir(self.root):
                await self.sandbox.remove(self.path(name), recursive=True)
        else:
            await self.sandbox.mkdir(self.root)

    async def unseal(self) -> None:
        if not await self.sandbox.exists(self.root):
            return
        result = await self.sandbox.run(["chmod", "-R", "u+w", self.root])
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"failed to unseal {self.root}")

    async def seal(self) -> None:
        result = await self.sandbox.run(["chmod", "-R", "a-w", self.root])
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"failed to seal {self.root}")

    async def write_header(
        self,
        *,
        session_id: str,
        round_number: int | None,
        proposal_id: str | None,
        parent_candidate_id: str | None,
    ) -> None:
        readme = """# Evaluation context

This directory is generated and read-only. It contains everything you are
authorized to inspect about how your program is evaluated:

- `results/` — past evaluation results (scores, per-case results, traces,
  artifacts). Start at `results/index.json`.
- `tasks/` — the task resources you may read, when exposed. Start at
  `tasks/index.json`.
- `candidates/` — prior program versions, with Git refs you can pass to
  `git show` or `git diff`. Start at `candidates/index.json`.
- `plan.json` — the named evaluations you may invoke, their base case
  selection, disclosure level, and remaining budget.

Full-disclosure results put each case and long trace in its own file; their
artifact paths resolve below that result's `artifacts/` directory.

Use ordinary filesystem and Git commands to analyze this context. Do not copy
it into the program: candidate versions that track `.evals` are rejected.
"""
        await self.sandbox.write_file(self.path("README.md"), readme)
        await self.write_json(
            self.path("manifest.json"),
            {
                "schema_version": 1,
                "session_id": session_id,
                "round": round_number,
                "proposal_id": proposal_id,
                "parent_candidate_id": parent_candidate_id,
                "snapshot_semantics": "generation",
            },
        )

    async def write_evaluations(
        self,
        projections: Sequence[
            tuple[
                EvaluationRecord,
                DisclosureLevel,
                EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement,
            ]
        ],
    ) -> None:
        root = self.path(RESULTS_SUBDIRECTORY)
        if await self.sandbox.exists(root):
            await self.sandbox.remove(root, recursive=True)
        await self.sandbox.mkdir(root)
        index: list[dict[str, object]] = []
        for record, disclosure, projection in sorted(
            projections,
            key=lambda item: (item[0].completed_at, item[0].id),
        ):
            digest = context_digest(record.id)
            relative_path = f"{digest}/evaluation.json"
            evaluation_root = self.path(RESULTS_SUBDIRECTORY, digest)
            await self.sandbox.mkdir(evaluation_root)
            missing_artifacts: list[str] = []
            if isinstance(projection, EvaluationRecord):
                payload = projection.model_dump(mode="json")
                cases = payload["report"].pop("cases")
                case_files: list[dict[str, object]] = []
                for case_model, case_payload in zip(
                    projection.report.cases,
                    cases,
                    strict=True,
                ):
                    case_digest = context_digest(case_model.case_id)
                    case_root = self.path(
                        RESULTS_SUBDIRECTORY, digest, "cases", case_digest
                    )
                    await self.sandbox.mkdir(case_root)
                    execution_trace = case_payload.pop("execution_trace", None)
                    evaluation_trace = case_payload.pop("evaluation_trace", None)
                    case_document: dict[str, object] = {
                        "schema_version": 1,
                        "result": case_payload,
                    }
                    if execution_trace is not None:
                        await self.write_json(
                            posixpath.join(case_root, "execution-trace.json"),
                            execution_trace,
                        )
                        case_document["execution_trace_path"] = "execution-trace.json"
                    if evaluation_trace is not None:
                        await self.write_json(
                            posixpath.join(case_root, "evaluation-trace.json"),
                            evaluation_trace,
                        )
                        case_document["evaluation_trace_path"] = "evaluation-trace.json"
                    await self.write_json(
                        posixpath.join(case_root, "result.json"),
                        case_document,
                    )
                    case_files.append(
                        {
                            "case_id": case_model.case_id,
                            "path": f"cases/{case_digest}/result.json",
                        }
                    )
                payload["case_files"] = case_files
                source_root = self.session_dir / "evaluations" / record.id / "artifacts"
                resolved_source_root = source_root.resolve()
                artifact_paths = {
                    artifact.path for artifact in projection.report.artifacts
                }
                for case in projection.report.cases:
                    artifact_paths.update(artifact.path for artifact in case.artifacts)
                for artifact_path in sorted(artifact_paths):
                    source = source_root / Path(*PurePosixPath(artifact_path).parts)
                    try:
                        resolved_source = source.resolve(strict=True)
                    except (OSError, RuntimeError):
                        missing_artifacts.append(artifact_path)
                        continue
                    contains_symlink = source.is_symlink() or (
                        source.is_dir()
                        and any(path.is_symlink() for path in source.rglob("*"))
                    )
                    if (
                        not resolved_source.is_relative_to(resolved_source_root)
                        or contains_symlink
                    ):
                        missing_artifacts.append(artifact_path)
                        continue
                    await self.sandbox.upload(
                        str(source),
                        self.path(RESULTS_SUBDIRECTORY, digest, "artifacts", artifact_path),
                    )
                document = {
                    "schema_version": 1,
                    "disclosure": disclosure.value,
                    "result": payload,
                    "artifacts_path": "artifacts",
                    "missing_artifacts": missing_artifacts,
                }
            else:
                document = {
                    "schema_version": 1,
                    "disclosure": disclosure.value,
                    "result": projection.model_dump(mode="json"),
                }
            await self.write_json(
                self.path(RESULTS_SUBDIRECTORY, relative_path),
                document,
            )
            index.append(
                {
                    "evaluation_id": record.id,
                    "candidate_id": record.request.candidate.id,
                    "candidate_version": record.request.candidate.version,
                    "evaluation": record.request.evaluation_set.name,
                    "partition": record.request.evaluation_set.partition,
                    "disclosure": disclosure.value,
                    "path": relative_path,
                }
            )
        await self.write_json(
            self.path(RESULTS_SUBDIRECTORY, "index.json"),
            {"schema_version": 1, "evaluations": index},
        )

    async def write_case_resources(
        self,
        entries: Sequence[ContextPlanEntry],
    ) -> None:
        root = self.path(TASKS_SUBDIRECTORY)
        if await self.sandbox.exists(root):
            await self.sandbox.remove(root, recursive=True)
        await self.sandbox.mkdir(root)
        index: list[dict[str, object]] = []
        for entry in entries:
            # expose_case_resources implies agent_visible (model validation).
            if not (
                entry.access.expose_case_resources
                and isinstance(entry.backend, CaseResourceExporter)
            ):
                continue
            digest = context_digest(
                entry.evaluation_set.budget_key(entry.backend_id)
            )
            resource_root = self.path(TASKS_SUBDIRECTORY, digest)
            await self.sandbox.mkdir(resource_root)
            await self.write_json(
                posixpath.join(resource_root, "manifest.json"),
                {
                    "schema_version": 1,
                    "backend_id": entry.backend_id,
                    "evaluation_set": entry.evaluation_set.model_dump(mode="json"),
                    "resources_path": "resources",
                },
            )
            resources = posixpath.join(resource_root, "resources")
            await self.sandbox.mkdir(resources)
            await entry.backend.export_case_resources(
                evaluation_set=entry.evaluation_set,
                destination=resources,
                sandbox=self.sandbox,
            )
            index.append(
                {
                    "backend_id": entry.backend_id,
                    "evaluation_set": entry.evaluation_set.model_dump(mode="json"),
                    "path": digest,
                }
            )
        await self.write_json(
            posixpath.join(root, "index.json"),
            {"schema_version": 1, "case_resources": index},
        )

    async def write_evaluation_plan(
        self,
        entries: Sequence[ContextPlanEntry],
    ) -> None:
        evaluations = []
        for entry in entries:
            access = entry.access
            if not access.agent_visible and not access.agent_can_evaluate:
                continue
            evaluations.append(
                {
                    "name": entry.evaluation_set.name,
                    "partition": entry.evaluation_set.partition,
                    "base_selection": entry.evaluation_set.selection.model_dump(
                        mode="json"
                    ),
                    "agent_can_evaluate": access.agent_can_evaluate,
                    "agent_selection": access.agent_selection.value,
                    "disclosure": access.disclosure.value,
                    "expose_case_resources": access.expose_case_resources,
                    "budget": (
                        entry.budget.model_dump(mode="json")
                        if entry.budget is not None
                        else None
                    ),
                }
            )
        await self.write_json(
            self.path(PLAN_FILENAME),
            {"schema_version": 1, "evaluations": evaluations},
        )


class WorkspaceContextManager:
    """Build and refresh one proposal's generation-scoped context snapshot."""

    def __init__(
        self,
        *,
        session_id: str,
        session_dir: Path,
        round_number: int,
        proposal_id: str,
        parent: Candidate,
        workspace: Workspace,
        candidate_repository: CandidateRepository,
        engine: EvaluationEngine,
        backend_id: str,
        evaluation_plan: EvaluationPlan,
        candidates: Sequence[Candidate],
        evaluations: Sequence[EvaluationRecord],
        ledger: AgentDisclosureLedger,
    ):
        self.session_id = session_id
        self.session_dir = session_dir
        self.round_number = round_number
        self.proposal_id = proposal_id
        self.parent = parent
        self.workspace = workspace
        self.candidate_repository = candidate_repository
        self.engine = engine
        self.backend_id = backend_id
        self.evaluation_plan = evaluation_plan
        self.candidates = {candidate.id: candidate for candidate in candidates}
        self.evaluations = {record.id: record for record in evaluations}
        self.ledger = ledger
        self.root = posixpath.join(
            self.workspace.project_path,
            AGENT_CONTEXT_DIRECTORY,
        )
        self.directory = AgentContextDirectory(
            sandbox=workspace.sandbox,
            root=self.root,
            session_dir=session_dir,
        )
        self._lock = asyncio.Lock()

    async def _projections(
        self,
    ) -> list[
        tuple[
            EvaluationRecord,
            DisclosureLevel,
            EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement,
        ]
    ]:
        projections = []
        for record in self.evaluations.values():
            decision = await self.engine.authorize(
                record.backend_id,
                record.request,
                EvaluationPrincipal.AGENT,
            )
            if not decision.viewable:
                continue
            maximum = await self.ledger.remember(record.id, decision.disclosure)
            disclosure = narrower_disclosure(maximum, decision.disclosure)
            projections.append(
                (record, disclosure, project_evaluation(record, disclosure))
            )
        return projections

    async def _write_candidate_history(self) -> None:
        await self.candidate_repository.materialize_agent_history(
            tuple(self.candidates.values()),
            workspace=self.workspace,
            destination=self.directory.path(CANDIDATES_SUBDIRECTORY),
        )

    def _context_entries(self) -> list[ContextPlanEntry]:
        ledger = self.engine.budget_ledger
        backend = self.engine.backends.resolve(self.backend_id)
        return [
            ContextPlanEntry(
                backend_id=self.backend_id,
                backend=backend,
                evaluation_set=definition.evaluation_set,
                access=definition.access,
                budget=(
                    ledger.get(
                        self.backend_id,
                        definition.evaluation_set,
                        EvaluationPrincipal.AGENT,
                    )
                    if ledger is not None
                    else None
                ),
            )
            for definition in self.evaluation_plan.evaluations
        ]

    async def initialize(self) -> None:
        async with self._lock:
            await self.directory.reset()
            await self.directory.write_header(
                session_id=self.session_id,
                round_number=self.round_number,
                proposal_id=self.proposal_id,
                parent_candidate_id=self.parent.id,
            )
            entries = self._context_entries()
            await self._write_candidate_history()
            await self.directory.write_evaluations(await self._projections())
            await self.directory.write_evaluation_plan(entries)
            await self.directory.write_case_resources(entries)
            await self.directory.seal()

    async def add_evaluation(
        self,
        record: EvaluationRecord,
        disclosure: DisclosureLevel,
    ) -> EvaluationReceipt:
        async with self._lock:
            self.candidates[record.request.candidate.id] = record.request.candidate
            self.evaluations[record.id] = record
            maximum = await self.ledger.remember(record.id, disclosure)
            effective = narrower_disclosure(maximum, disclosure)
            await self.directory.unseal()
            await self._write_candidate_history()
            await self.directory.write_evaluations(await self._projections())
            await self.directory.write_evaluation_plan(self._context_entries())
            await self.directory.seal()
            return make_evaluation_receipt(record, effective)
