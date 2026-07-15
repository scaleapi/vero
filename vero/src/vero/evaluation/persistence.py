"""Atomic schema-v2 persistence for canonical evaluation records."""

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

from vero.core.db.candidate import Candidate
from vero.evaluation.models import (
    BackendProvenance,
    EvaluationModel,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    ObjectiveResult,
    ObjectiveSpec,
)
from vero.evaluation.objective import select_best_evaluation

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
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _case_digest(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


class CaseFileReference(EvaluationModel):
    case_id: str
    path: str

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, case_id: str) -> str:
        if not case_id.strip():
            raise ValueError("case ID must not be empty")
        return case_id

    @model_validator(mode="after")
    def validate_path(self) -> CaseFileReference:
        expected = f"cases/{_case_digest(self.case_id)}.json"
        if self.path != expected:
            raise ValueError(f"case path must be {expected!r}")
        return self


class EvaluationManifest(EvaluationModel):
    schema_version: Literal[2] = 2
    lifecycle: Literal["complete"] = "complete"
    id: str
    request: EvaluationRequest
    report: EvaluationReport
    case_files: list[CaseFileReference] = Field(default_factory=list)
    backend_id: str
    backend: BackendProvenance
    objective_spec: ObjectiveSpec | None = None
    objective: ObjectiveResult | None = None
    created_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def validate_manifest(self) -> EvaluationManifest:
        if self.report.cases:
            raise ValueError("manifest report must store cases in case_files")
        case_ids = [case_file.case_id for case_file in self.case_files]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case file references must be unique")
        if (self.objective_spec is None) != (self.objective is None):
            raise ValueError("objective_spec and objective must both be present or absent")
        return self

    @classmethod
    def from_record(cls, record: EvaluationRecord) -> EvaluationManifest:
        case_files = [
            CaseFileReference(
                case_id=case.case_id,
                path=f"cases/{_case_digest(case.case_id)}.json",
            )
            for case in record.report.cases
        ]
        return cls(
            id=record.id,
            request=record.request,
            report=record.report.model_copy(update={"cases": []}),
            case_files=case_files,
            backend_id=record.backend_id,
            backend=record.backend,
            objective_spec=record.objective_spec,
            objective=record.objective,
            created_at=record.created_at,
            completed_at=record.completed_at,
        )


class RunningEvaluationManifest(EvaluationModel):
    schema_version: Literal[2] = 2
    lifecycle: Literal["running"] = "running"
    id: str
    request: EvaluationRequest
    backend_id: str
    backend: BackendProvenance
    objective_spec: ObjectiveSpec | None = None
    created_at: datetime


class CaseCheckpointStore:
    """Per-evaluation canonical case store safe for asynchronous checkpoints."""

    def __init__(self, cases_dir: Path):
        self.cases_dir = cases_dir
        self._lock = asyncio.Lock()

    def path_for(self, case_id: str) -> Path:
        return self.cases_dir / f"{_case_digest(case_id)}.json"

    async def save(self, result) -> None:
        from vero.evaluation.models import CaseResult

        if not isinstance(result, CaseResult):
            raise TypeError("result must be a CaseResult")
        async with self._lock:
            _atomic_write_json(
                self.path_for(result.case_id),
                result.model_dump(mode="json"),
            )

    async def load(self, case_id: str):
        from vero.evaluation.models import CaseResult

        path = self.path_for(case_id)
        async with self._lock:
            if not path.exists():
                return None
            try:
                result = CaseResult.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as error:
                raise ValueError(f"invalid case checkpoint {path}: {error}") from error
        if result.case_id != case_id:
            raise ValueError(
                f"case checkpoint {path} contains ID {result.case_id!r}, expected {case_id!r}"
            )
        return result

    async def load_all(self):
        from vero.evaluation.models import CaseResult

        async with self._lock:
            paths = sorted(self.cases_dir.glob("*.json")) if self.cases_dir.exists() else []
            results = []
            for path in paths:
                try:
                    results.append(
                        CaseResult.model_validate_json(path.read_text(encoding="utf-8"))
                    )
                except Exception as error:
                    raise ValueError(f"invalid case checkpoint {path}: {error}") from error
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

    def _validate_artifact_paths(self, record: EvaluationRecord) -> None:
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
        _atomic_write_json(
            self.manifest_path,
            manifest.model_dump(mode="json"),
        )

    async def save(self, record: EvaluationRecord) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._validate_artifact_paths(record)
        for case in record.report.cases:
            await self.cases.save(case)
        manifest = EvaluationManifest.from_record(record)
        _atomic_write_json(
            self.manifest_path,
            manifest.model_dump(mode="json"),
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

        cases = []
        for case_file in manifest.case_files:
            case_path = self.result_dir / case_file.path
            try:
                from vero.evaluation.models import CaseResult

                case = CaseResult.model_validate_json(case_path.read_text(encoding="utf-8"))
            except Exception as error:
                raise ValueError(f"invalid evaluation case file {case_path}: {error}") from error
            if case.case_id != case_file.case_id:
                raise ValueError(
                    f"evaluation case file {case_path} contains ID {case.case_id!r}, "
                    f"expected {case_file.case_id!r}"
                )
            cases.append(case)

        report = manifest.report.model_copy(update={"cases": cases})
        record = EvaluationRecord(
            id=manifest.id,
            request=manifest.request,
            report=report,
            backend_id=manifest.backend_id,
            backend=manifest.backend,
            objective_spec=manifest.objective_spec,
            objective=manifest.objective,
            created_at=manifest.created_at,
            completed_at=manifest.completed_at,
        )
        self._validate_artifact_paths(record)
        return record


@dataclass
class EvaluationDatabase:
    """In-memory schema-v2 index of candidates and evaluation records."""

    id: str
    candidates: dict[tuple[str, str], Candidate] = field(default_factory=dict)
    evaluations: dict[str, EvaluationRecord] = field(default_factory=dict)
    datasets: dict[str, Any] = field(default_factory=dict)
    listeners: list[Callable[[EvaluationRecord], None]] = field(
        default_factory=list,
        repr=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    def add_evaluation(self, record: EvaluationRecord) -> None:
        with self._lock:
            if record.id in self.evaluations:
                return
            self.candidates.setdefault(record.request.candidate.id, record.request.candidate)
            self.evaluations[record.id] = record
            for listener in self.listeners:
                listener(record)

    def get_evaluation(self, evaluation_id: str) -> EvaluationRecord | None:
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
        ids = list(self.evaluations) if evaluation_ids is None else evaluation_ids
        records = [self.evaluations[evaluation_id] for evaluation_id in ids]
        if filter_fn is not None:
            records = [record for record in records if filter_fn(record)]
        if reverse:
            records.reverse()
        records = records[offset:]
        if limit is not None:
            records = records[:limit]
        return records

    def get_evaluations_df(
        self,
        evaluations: list[EvaluationRecord] | None = None,
    ):
        """Return a compact tabular view without embedding case payloads."""
        import pandas as pd

        records = evaluations if evaluations is not None else self.get_evaluations()
        rows = []
        for record in records:
            row = {
                "id": record.id,
                "candidate_commit": record.request.candidate.commit,
                "repo_name": record.request.candidate.repo_name,
                "backend_id": record.backend_id,
                "evaluation_set": record.request.evaluation_set.name,
                "partition": record.request.evaluation_set.partition,
                "status": record.report.status.value,
                "objective_value": (
                    record.objective.value if record.objective is not None else None
                ),
                "feasible": (
                    record.objective.feasible if record.objective is not None else None
                ),
                "created_at": record.created_at,
                "completed_at": record.completed_at,
            }
            row.update(
                {
                    f"metric/{metric}": value
                    for metric, value in record.report.metrics.items()
                }
            )
            rows.append(row)

        frame = pd.DataFrame(rows)
        if "id" in frame.columns:
            frame.set_index("id", inplace=True)
        return frame

    def get_best(
        self,
        objective_spec: ObjectiveSpec,
        *,
        backend_ids: set[str] | None = None,
        evaluation_sets: list[EvaluationSet] | None = None,
        exclude_candidate: Candidate | tuple[str, str] | None = None,
    ) -> EvaluationRecord | None:
        excluded_id = (
            exclude_candidate.id
            if isinstance(exclude_candidate, Candidate)
            else exclude_candidate
        )
        records = [
            record
            for record in self.evaluations.values()
            if record.objective_spec == objective_spec
            and (backend_ids is None or record.backend_id in backend_ids)
            and (
                evaluation_sets is None
                or record.request.evaluation_set in evaluation_sets
            )
            and (
                excluded_id is None or record.request.candidate.id != excluded_id
            )
        ]
        return select_best_evaluation(records)

    @staticmethod
    def _serialize_candidate_id(candidate_id: tuple[str, str]) -> str:
        return json.dumps(candidate_id, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _deserialize_candidate_id(candidate_id: str) -> tuple[str, str]:
        try:
            parts = json.loads(candidate_id)
        except json.JSONDecodeError:
            # Read early schema-v2 development snapshots without rewriting them.
            parts = candidate_id.split("|", maxsplit=1)
        if (
            not isinstance(parts, list)
            and not isinstance(parts, tuple)
        ) or len(parts) != 2 or not all(isinstance(part, str) for part in parts):
            raise ValueError(f"invalid serialized candidate ID: {candidate_id!r}")
        return parts[0], parts[1]

    def serialize(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 2,
                "id": self.id,
                "candidates": {
                    self._serialize_candidate_id(candidate_id): candidate.model_dump(
                        mode="json"
                    )
                    for candidate_id, candidate in self.candidates.items()
                },
                "evaluations": {
                    evaluation_id: record.model_dump(mode="json")
                    for evaluation_id, record in self.evaluations.items()
                },
                "datasets": self.datasets,
            }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EvaluationDatabase:
        if data.get("schema_version") != 2:
            from vero.evaluation.legacy import deserialize_legacy_database

            return deserialize_legacy_database(data)
        database = cls(id=data["id"])
        for candidate_id, candidate in data.get("candidates", {}).items():
            database.candidates[database._deserialize_candidate_id(candidate_id)] = (
                Candidate.model_validate(candidate)
            )
        for evaluation_id, record in data.get("evaluations", {}).items():
            evaluation = EvaluationRecord.model_validate(record)
            if evaluation.id != evaluation_id:
                raise ValueError(
                    f"evaluation map key {evaluation_id!r} does not match record ID "
                    f"{evaluation.id!r}"
                )
            database.evaluations[evaluation_id] = evaluation
            database.candidates.setdefault(
                evaluation.request.candidate.id,
                evaluation.request.candidate,
            )
        database.datasets = data.get("datasets", {})
        return database

    def to_json(self) -> str:
        return json.dumps(self.serialize(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_value: str) -> EvaluationDatabase:
        return cls.deserialize(json.loads(json_value))

    def save_to_file(self, path: Path) -> None:
        with self._lock:
            _atomic_write_json(path, self.serialize())

    @classmethod
    def load_from_file(cls, path: Path) -> EvaluationDatabase:
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_evaluations_dir(
        cls,
        evaluations_dir: Path,
        *,
        db_id: str | None = None,
    ) -> EvaluationDatabase:
        database = cls(id=db_id or evaluations_dir.parent.name)
        if not evaluations_dir.exists():
            return database
        for result_dir in sorted(evaluations_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            manifest_path = result_dir / EvaluationStore.manifest_basename
            if not manifest_path.exists():
                continue
            try:
                database.add_evaluation(EvaluationStore(result_dir).load())
            except Exception as error:
                logger.warning("Skipping corrupt evaluation %s: %s", result_dir, error)

        # Schema-v1 result directories remain readable during the migration. A
        # temporary symlink index lets the legacy loader see only legacy-shaped
        # directories, avoiding spurious warnings for canonical evaluations.
        from vero.core.constants import evaluation_parameters_basename

        legacy_result_dirs = [
            result_dir
            for result_dir in sorted(evaluations_dir.iterdir())
            if result_dir.is_dir()
            and (result_dir / evaluation_parameters_basename).exists()
        ]
        if legacy_result_dirs:
            from vero.core.db.database import ExperimentDatabase
            from vero.evaluation.legacy import convert_experiment_database

            with tempfile.TemporaryDirectory(prefix="vero-legacy-evaluations-") as root:
                index = Path(root) / "experiments"
                index.mkdir()
                for result_dir in legacy_result_dirs:
                    (index / result_dir.name).symlink_to(
                        result_dir.resolve(),
                        target_is_directory=True,
                    )
                legacy = ExperimentDatabase.from_experiments_dir(
                    index,
                    db_id=database.id,
                )
                converted = convert_experiment_database(legacy)
                for record in converted.evaluations.values():
                    database.add_evaluation(record)
        return database
