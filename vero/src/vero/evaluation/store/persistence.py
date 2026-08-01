"""Atomic persistence for canonical evaluation records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import Field, field_validator, model_validator

from vero.candidate import Candidate
from vero.evaluation.models import (
    BackendProvenance,
    CaseResult,
    EvaluationPrincipal,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    ObjectiveResult,
    ObjectiveSpec,
)
from vero.evaluation.scoring.objective import select_best_evaluation
from vero.models import StrictModel

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        # This is the only window where we still own the raw descriptor: once
        # os.fdopen succeeds the file object owns it and closes it exactly once
        # when the block below exits, so closing it again from a shared failure
        # path was a double close. Three writers reach this helper concurrently
        # through asyncio.to_thread, and on CPython that second close can land
        # on an unrelated descriptor that has since been handed the same number.
        os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        # Fsyncing the file only makes its bytes durable; the rename that
        # publishes them is a change to the parent directory, so a hard power
        # loss or container kill right here can leave the old contents behind or
        # no file at all. Every durable artifact vero has commits through this
        # one helper (the evaluation manifest, the per-case checkpoints,
        # database.json, budgets.json, the session manifest, and the disclosure
        # ledger), so fsync the directory as well. Some platforms refuse to open
        # a directory for fsync, and this is a durability upgrade rather than
        # part of the write itself, so an OSError from either step is ignored
        # instead of failing a write that already landed.
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _case_digest(case_id: str) -> str:
    return hashlib.sha256(case_id.encode()).hexdigest()


class CaseFileReference(StrictModel):
    case_id: str
    path: str

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("case ID must not be empty")
        return value

    @model_validator(mode="after")
    def validate_path(self) -> CaseFileReference:
        expected = f"cases/{_case_digest(self.case_id)}.json"
        if self.path != expected:
            raise ValueError(f"case path must be {expected!r}")
        return self


class EvaluationManifest(StrictModel):
    schema_version: Literal[1] = 1
    lifecycle: Literal["complete"] = "complete"
    id: str
    request: EvaluationRequest
    report: EvaluationReport
    case_files: list[CaseFileReference] = Field(default_factory=list)
    backend_id: str
    backend: BackendProvenance
    # Persist the principal so a record reconstructed from this source-of-truth
    # directory keeps its true provenance instead of silently defaulting to
    # SYSTEM. Older manifests without the field read back as SYSTEM, matching
    # the historical EvaluationRecord default.
    principal: EvaluationPrincipal = EvaluationPrincipal.SYSTEM
    objective_spec: ObjectiveSpec | None = None
    objective: ObjectiveResult | None = None
    created_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def validate_manifest(self) -> EvaluationManifest:
        if self.report.cases:
            raise ValueError("manifest report must store cases in case_files")
        ids = [reference.case_id for reference in self.case_files]
        if len(ids) != len(set(ids)):
            raise ValueError("case file references must be unique")
        if (self.objective_spec is None) != (self.objective is None):
            raise ValueError(
                "objective_spec and objective must both be present or absent"
            )
        return self

    @classmethod
    def from_record(cls, record: EvaluationRecord) -> EvaluationManifest:
        return cls(
            id=record.id,
            request=record.request,
            report=record.report.model_copy(update={"cases": []}),
            case_files=[
                CaseFileReference(
                    case_id=case.case_id,
                    path=f"cases/{_case_digest(case.case_id)}.json",
                )
                for case in record.report.cases
            ],
            backend_id=record.backend_id,
            backend=record.backend,
            principal=record.principal,
            objective_spec=record.objective_spec,
            objective=record.objective,
            created_at=record.created_at,
            completed_at=record.completed_at,
        )


class RunningEvaluationManifest(StrictModel):
    schema_version: Literal[1] = 1
    lifecycle: Literal["running"] = "running"
    id: str
    request: EvaluationRequest
    backend_id: str
    backend: BackendProvenance
    objective_spec: ObjectiveSpec | None = None
    created_at: datetime


class CaseCheckpointStore:
    """Per-evaluation case checkpoints safe for concurrent async writers."""

    def __init__(self, cases_dir: Path):
        self.cases_dir = cases_dir
        self._lock = asyncio.Lock()

    def path_for(self, case_id: str) -> Path:
        return self.cases_dir / f"{_case_digest(case_id)}.json"

    async def save(self, result: CaseResult) -> None:
        if not isinstance(result, CaseResult):
            raise TypeError("result must be a CaseResult")
        async with self._lock:
            await asyncio.to_thread(
                _atomic_write_json,
                self.path_for(result.case_id),
                result.model_dump(mode="json"),
            )

    async def load(self, case_id: str) -> CaseResult | None:
        path = self.path_for(case_id)
        async with self._lock:
            if not path.exists():
                return None
            try:
                result = CaseResult.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as error:
                raise ValueError(f"invalid case checkpoint {path}: {error}") from error
        if result.case_id != case_id:
            raise ValueError(
                f"case checkpoint {path} contains ID {result.case_id!r}, expected {case_id!r}"
            )
        return result

    async def load_all(self) -> list[CaseResult]:
        async with self._lock:
            paths = (
                sorted(self.cases_dir.glob("*.json")) if self.cases_dir.exists() else []
            )
            results: list[CaseResult] = []
            for path in paths:
                try:
                    results.append(
                        CaseResult.model_validate_json(path.read_text(encoding="utf-8"))
                    )
                except Exception as error:
                    raise ValueError(
                        f"invalid case checkpoint {path}: {error}"
                    ) from error
            return results


class EvaluationStore:
    """Persist and reconstruct one evaluation directory."""

    manifest_basename = "evaluation.json"

    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.cases = CaseCheckpointStore(result_dir / "cases")
        self.artifact_dir = result_dir / "artifacts"

    @property
    def manifest_path(self) -> Path:
        return self.result_dir / self.manifest_basename

    def _validate_artifacts(self, record: EvaluationRecord) -> None:
        artifact_root = self.artifact_dir.resolve()
        artifacts = list(record.report.artifacts)
        for case in record.report.cases:
            artifacts.extend(case.artifacts)
        for artifact in artifacts:
            resolved = (self.artifact_dir / artifact.path).resolve()
            if not resolved.is_relative_to(artifact_root):
                raise ValueError(
                    f"artifact path {artifact.path!r} escapes evaluation artifact directory"
                )

    def write_running(
        self,
        *,
        evaluation_id: str,
        request: EvaluationRequest,
        backend_id: str,
        backend: BackendProvenance,
        objective_spec: ObjectiveSpec | None,
        created_at: datetime,
    ) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        manifest = RunningEvaluationManifest(
            id=evaluation_id,
            request=request,
            backend_id=backend_id,
            backend=backend,
            objective_spec=objective_spec,
            created_at=created_at,
        )
        _atomic_write_json(self.manifest_path, manifest.model_dump(mode="json"))

    async def save(self, record: EvaluationRecord) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._validate_artifacts(record)
        for case in record.report.cases:
            await self.cases.save(case)
        _atomic_write_json(
            self.manifest_path,
            EvaluationManifest.from_record(record).model_dump(mode="json"),
        )

    def load(self) -> EvaluationRecord:
        try:
            manifest = EvaluationManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise ValueError(
                f"invalid evaluation manifest {self.manifest_path}: {error}"
            ) from error

        cases: list[CaseResult] = []
        for reference in manifest.case_files:
            case_path = self.result_dir / reference.path
            try:
                case = CaseResult.model_validate_json(
                    case_path.read_text(encoding="utf-8")
                )
            except Exception as error:
                raise ValueError(
                    f"invalid evaluation case file {case_path}: {error}"
                ) from error
            if case.case_id != reference.case_id:
                raise ValueError(
                    f"evaluation case file {case_path} contains ID {case.case_id!r}, "
                    f"expected {reference.case_id!r}"
                )
            cases.append(case)

        record = EvaluationRecord(
            id=manifest.id,
            request=manifest.request,
            report=manifest.report.model_copy(update={"cases": cases}),
            backend_id=manifest.backend_id,
            backend=manifest.backend,
            principal=manifest.principal,
            objective_spec=manifest.objective_spec,
            objective=manifest.objective,
            created_at=manifest.created_at,
            completed_at=manifest.completed_at,
        )
        self._validate_artifacts(record)
        return record


@dataclass
class EvaluationDatabase:
    """Thread-safe in-memory index of candidates and evaluation records."""

    id: str
    candidates: dict[str, Candidate] = field(default_factory=dict)
    evaluations: dict[str, EvaluationRecord] = field(default_factory=dict)
    listeners: list[Callable[[EvaluationRecord], None]] = field(
        default_factory=list,
        repr=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def add_evaluation(self, record: EvaluationRecord) -> None:
        with self._lock:
            existing_record = self.evaluations.get(record.id)
            if existing_record is not None:
                if existing_record != record:
                    raise ValueError(
                        f"evaluation ID {record.id!r} already has a different record"
                    )
                return
            candidate = record.request.candidate
            existing_candidate = self.candidates.get(candidate.id)
            if existing_candidate is not None and existing_candidate != candidate:
                raise ValueError(
                    f"candidate ID {candidate.id!r} already has a different identity"
                )
            self.candidates[candidate.id] = candidate
            self.evaluations[record.id] = record
            for listener in self.listeners:
                listener(record)

    def get_evaluation(self, evaluation_id: str) -> EvaluationRecord | None:
        with self._lock:
            return self.evaluations.get(evaluation_id)

    def get_evaluations(
        self,
        evaluation_ids: list[str] | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
        reverse: bool = False,
        filter_fn: Callable[[EvaluationRecord], bool] | None = None,
    ) -> list[EvaluationRecord]:
        with self._lock:
            ids = list(self.evaluations) if evaluation_ids is None else evaluation_ids
            records = [self.evaluations[evaluation_id] for evaluation_id in ids]
        if filter_fn is not None:
            records = [record for record in records if filter_fn(record)]
        if reverse:
            records.reverse()
        records = records[offset:]
        return records if limit is None else records[:limit]

    def get_best(
        self,
        objective_spec: ObjectiveSpec,
        *,
        backend_ids: set[str] | None = None,
        evaluation_sets: list[EvaluationSet] | None = None,
        exclude_candidate_id: str | None = None,
    ) -> EvaluationRecord | None:
        with self._lock:
            records = [
                record
                for record in self.evaluations.values()
                if record.objective_spec == objective_spec
                and (backend_ids is None or record.backend_id in backend_ids)
                and (
                    evaluation_sets is None
                    or record.request.evaluation_set in evaluation_sets
                )
                and record.request.candidate.id != exclude_candidate_id
            ]
        return select_best_evaluation(records)

    def serialize(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "id": self.id,
                "candidates": {
                    candidate_id: candidate.model_dump(mode="json")
                    for candidate_id, candidate in self.candidates.items()
                },
                "evaluations": {
                    evaluation_id: record.model_dump(mode="json")
                    for evaluation_id, record in self.evaluations.items()
                },
            }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EvaluationDatabase:
        if data.get("schema_version") != 1:
            raise ValueError("unsupported evaluation database schema")
        database = cls(id=data["id"])
        for candidate_id, value in data.get("candidates", {}).items():
            candidate = Candidate.model_validate(value)
            if candidate.id != candidate_id:
                raise ValueError(
                    f"candidate map key {candidate_id!r} does not match candidate ID {candidate.id!r}"
                )
            database.candidates[candidate_id] = candidate
        for evaluation_id, value in data.get("evaluations", {}).items():
            record = EvaluationRecord.model_validate(value)
            if record.id != evaluation_id:
                raise ValueError(
                    f"evaluation map key {evaluation_id!r} does not match record ID {record.id!r}"
                )
            database.add_evaluation(record)
        return database

    def to_json(self) -> str:
        return json.dumps(self.serialize(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, value: str) -> EvaluationDatabase:
        return cls.deserialize(json.loads(value))

    def save_to_file(self, path: Path) -> None:
        with self._lock:
            _atomic_write_json(path, self.serialize())

    @classmethod
    def load_from_file(cls, path: Path) -> EvaluationDatabase:
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def load_reconciled(
        cls,
        *,
        database_path: Path,
        evaluations_dir: Path,
        database_id: str,
    ) -> EvaluationDatabase:
        """Load the index and repair it from canonical completed evaluations."""

        existed = database_path.exists()
        database: EvaluationDatabase | None = None
        if existed:
            try:
                database = cls.load_from_file(database_path)
            except Exception as error:
                # A crash in the middle of a write leaves a truncated index, and
                # raising here bricked the session for good: every later run of
                # it died on load even though this file is only a cache of the
                # per-evaluation manifests that get reconciled in below. Rebuild
                # from those instead of refusing to start, and log it, since the
                # operator otherwise has no way to tell that an index they will
                # find rewritten on disk was ever damaged.
                logger.warning(
                    "Rebuilding unreadable evaluation database %s from the "
                    "per-evaluation manifests: %s",
                    database_path,
                    error,
                )
                # Treat the damaged file as absent so the repaired index is
                # written back even when no new evaluation appeared on disk.
                existed = False
        if database is None:
            database = cls(id=database_id)
        if database.id != database_id:
            # A readable index naming a different session is a real "wrong
            # session" signal (a copied or crossed session directory), never a
            # torn write, so it stays a hard error rather than a rebuild.
            raise ValueError(
                f"evaluation database belongs to {database.id!r}, not {database_id!r}"
            )

        completed = cls.from_evaluations_dir(
            evaluations_dir,
            database_id=database_id,
        )
        changed = not existed
        for evaluation_id, record in completed.evaluations.items():
            if evaluation_id not in database.evaluations:
                changed = True
            database.add_evaluation(record)
        if changed:
            database.save_to_file(database_path)
        return database

    @classmethod
    def from_evaluations_dir(
        cls,
        evaluations_dir: Path,
        *,
        database_id: str | None = None,
    ) -> EvaluationDatabase:
        database = cls(id=database_id or evaluations_dir.parent.name)
        if not evaluations_dir.exists():
            return database
        for result_dir in sorted(evaluations_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            if not (result_dir / EvaluationStore.manifest_basename).exists():
                continue
            try:
                manifest = json.loads(
                    (result_dir / EvaluationStore.manifest_basename).read_text(
                        encoding="utf-8"
                    )
                )
                if manifest.get("lifecycle", "complete") != "complete":
                    continue
                database.add_evaluation(EvaluationStore(result_dir).load())
            except Exception as error:
                logger.warning("Skipping corrupt evaluation %s: %s", result_dir, error)
        return database
