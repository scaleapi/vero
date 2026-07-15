"""Optional uv-based adapter for Python tasks defined with scale-vero-tasks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from vero.evaluation.backend import EvaluationContext
from vero.evaluation.command import CommandBackend, CommandBackendConfig
from vero.evaluation.models import (
    AllCases,
    BackendProvenance,
    CaseIds,
    CaseRange,
    EvaluationCost,
    EvaluationModel,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
)


def _default_uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise ValueError("uv is required to configure a Python task backend")
    return str(Path(executable).resolve())


class PythonTaskBackendConfig(EvaluationModel):
    """Configuration for an external task harness and editable target package."""

    harness_root: str
    module: str
    task: str
    cases_path: str
    target_project_directory: str = "."
    evaluation_set_name: str = "default"
    partition: str | None = None
    uv_executable: str = Field(default_factory=_default_uv)
    python_executable: str = "python"
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)

    @field_validator("harness_root", "cases_path")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        if not value.strip() or not Path(value).is_absolute():
            raise ValueError("Python task backend paths must be absolute")
        return value

    @field_validator("module", "task", "python_executable", "evaluation_set_name")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Python task backend identity must not be empty")
        return value

    @field_validator("uv_executable")
    @classmethod
    def validate_uv_executable(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("uv_executable must not be empty")
        return value

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Python task partition must not be empty")
        return value

    @field_validator("target_project_directory")
    @classmethod
    def validate_target_project_directory(cls, value: str) -> str:
        path = Path(value)
        if not value.strip() or path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "Python task target_project_directory must stay within the "
                "candidate workspace"
            )
        return path.as_posix()

    @model_validator(mode="after")
    def validate_filesystem(self) -> PythonTaskBackendConfig:
        if not Path(self.harness_root).is_dir():
            raise ValueError("Python task harness_root must be an existing directory")
        if not Path(self.cases_path).is_file():
            raise ValueError("Python task cases_path must be an existing file")
        return self


class PythonTaskBackend:
    """Run an external Python task harness against an editable candidate."""

    name = "python-task"
    version = "1"

    def __init__(self, config: PythonTaskBackendConfig):
        self.config = config
        target = "{workspace}"
        if config.target_project_directory != ".":
            target += f"/{config.target_project_directory}"
        self._command = CommandBackend(
            CommandBackendConfig(
                harness_root=config.harness_root,
                command=[
                    config.uv_executable,
                    "run",
                    "--project",
                    "{harness}",
                    "--with-editable",
                    target,
                    config.python_executable,
                    "-m",
                    "vero_tasks.runner",
                    "--module",
                    config.module,
                    "--task",
                    config.task,
                    "--cases",
                    "{input:cases}",
                    "--request",
                    "{request}",
                    "--report",
                    "{report}",
                ],
                environment=config.environment,
                passthrough_environment=config.passthrough_environment,
                staged_inputs={"cases": config.cases_path},
            )
        )

    @property
    def provenance(self) -> BackendProvenance:
        return BackendProvenance.from_config(
            name=self.name,
            version=self.version,
            config=self.config,
        )

    def _cases(self) -> list[object]:
        path = Path(self.config.cases_path)
        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("cases")
        if not isinstance(value, list):
            raise ValueError("Python task case file must contain a case list")
        return value

    def _case_ids(self) -> list[str]:
        case_ids = [
            str(case["id"])
            if isinstance(case, dict) and case.get("id") is not None
            else str(index)
            for index, case in enumerate(self._cases())
        ]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Python task case IDs must be unique")
        return case_ids

    def _validate_evaluation_set(self, evaluation_set: EvaluationSet) -> None:
        if evaluation_set.name != self.config.evaluation_set_name:
            raise ValueError(
                f"Python task backend owns evaluation set "
                f"{self.config.evaluation_set_name!r}, not {evaluation_set.name!r}"
            )
        if evaluation_set.partition != self.config.partition:
            raise ValueError(
                f"Python task backend owns partition {self.config.partition!r}, "
                f"not {evaluation_set.partition!r}"
            )

        case_ids = self._case_ids()
        selection = evaluation_set.selection
        if isinstance(selection, CaseRange) and selection.stop > len(case_ids):
            raise ValueError(
                f"case range stops at {selection.stop}, but the evaluation set "
                f"contains {len(case_ids)} cases"
            )
        if isinstance(selection, CaseIds):
            unknown = sorted(set(selection.ids) - set(case_ids))
            if unknown:
                raise ValueError(f"unknown Python task case IDs: {unknown}")

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost:
        self._validate_evaluation_set(evaluation_set)
        selection = evaluation_set.selection
        if isinstance(selection, CaseIds):
            return EvaluationCost(cases=len(selection.ids))
        if isinstance(selection, CaseRange):
            return EvaluationCost(cases=selection.stop - selection.start)
        if isinstance(selection, AllCases):
            return EvaluationCost(cases=len(self._case_ids()))
        raise AssertionError(f"unsupported case selection: {selection}")

    def validate_request(self, request: EvaluationRequest) -> None:
        self._command.validate_request(request)
        self._validate_evaluation_set(request.evaluation_set)

    def sanitize_error(self, message: str) -> str:
        return self._command.sanitize_error(message)

    async def evaluate(
        self,
        *,
        context: EvaluationContext,
        request: EvaluationRequest,
    ) -> EvaluationReport:
        target_root = context.workspace.sandbox.host_path(context.workspace.project_path)
        if target_root is not None:
            target_root = target_root.resolve()
            cases_path = Path(self.config.cases_path).resolve()
            if cases_path == target_root or cases_path.is_relative_to(target_root):
                raise ValueError("Python task cases must live outside the editable target")
        return await self._command.evaluate(context=context, request=request)
