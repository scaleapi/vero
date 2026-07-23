"""Optional uv-based adapter for Python tasks defined with scale-vero-tasks."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

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
from vero.sandbox import Sandbox


def _default_uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise ValueError("uv is required to configure a Python task backend")
    return str(Path(executable).resolve())


class PythonTaskEvaluationConfig(EvaluationModel):
    """One named/partitioned dataset owned by a Python task backend."""

    name: str
    cases_path: str
    partition: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Python task evaluation name must not be empty")
        return value

    @field_validator("cases_path")
    @classmethod
    def validate_cases_path(cls, value: str) -> str:
        if not value.strip() or not Path(value).is_absolute():
            raise ValueError("Python task cases_path must be absolute")
        return value

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Python task partition must not be empty")
        return value


class PythonTaskBackendConfig(EvaluationModel):
    """Configuration for an external task harness and editable target package."""

    harness_root: str
    module: str
    task: str
    evaluations: list[PythonTaskEvaluationConfig]
    target_project_directory: str = "."
    uv_executable: str = Field(default_factory=_default_uv)
    python_executable: str = "python"
    environment: dict[str, str] = Field(default_factory=dict)
    passthrough_environment: list[str] = Field(default_factory=list)

    @field_validator("harness_root")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        if not value.strip() or not Path(value).is_absolute():
            raise ValueError("Python task backend paths must be absolute")
        return value

    @field_validator("module", "task", "python_executable")
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
        if not self.evaluations:
            raise ValueError("Python task backend requires at least one evaluation")
        keys = [(item.name, item.partition) for item in self.evaluations]
        if len(keys) != len(set(keys)):
            raise ValueError("Python task evaluation name/partition pairs must be unique")
        for evaluation in self.evaluations:
            if not Path(evaluation.cases_path).is_file():
                raise ValueError(
                    "Python task cases_path must be an existing file: "
                    f"{evaluation.cases_path}"
                )
        return self


class PythonTaskBackend:
    """Run an external Python task harness against an editable candidate."""

    name = "python-task"
    version = "2"

    def __init__(self, config: PythonTaskBackendConfig):
        self.config = config
        target = "{workspace}"
        if config.target_project_directory != ".":
            target += f"/{config.target_project_directory}"
        self._target = target
        self._commands: dict[tuple[str, str | None], CommandBackend] = {}

    def _source(self, evaluation_set: EvaluationSet) -> PythonTaskEvaluationConfig:
        for source in self.config.evaluations:
            if (source.name, source.partition) == (
                evaluation_set.name,
                evaluation_set.partition,
            ):
                return source
        raise ValueError(
            "Python task backend does not own evaluation "
            f"{evaluation_set.name!r} partition {evaluation_set.partition!r}"
        )

    def _command(self, evaluation_set: EvaluationSet) -> CommandBackend:
        source = self._source(evaluation_set)
        key = (source.name, source.partition)
        command = self._commands.get(key)
        if command is not None:
            return command
        command = CommandBackend(
            CommandBackendConfig(
                harness_root=self.config.harness_root,
                command=[
                    self.config.uv_executable,
                    "run",
                    "--project",
                    "{harness}",
                    "--with-editable",
                    self._target,
                    self.config.python_executable,
                    "-m",
                    "vero_tasks.runner",
                    "--module",
                    self.config.module,
                    "--task",
                    self.config.task,
                    "--cases",
                    "{input:cases}",
                    "--request",
                    "{request}",
                    "--report",
                    "{report}",
                ],
                environment=self.config.environment,
                passthrough_environment=self.config.passthrough_environment,
                staged_inputs={"cases": source.cases_path},
            )
        )
        self._commands[key] = command
        return command

    @property
    def provenance(self) -> BackendProvenance:
        return BackendProvenance.from_config(
            name=self.name,
            version=self.version,
            config=self.config,
        )

    def _cases(self, evaluation_set: EvaluationSet) -> list[object]:
        path = Path(self._source(evaluation_set).cases_path)
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

    def _case_ids(self, evaluation_set: EvaluationSet) -> list[str]:
        case_ids = [
            str(case["id"])
            if isinstance(case, dict) and case.get("id") is not None
            else str(index)
            for index, case in enumerate(self._cases(evaluation_set))
        ]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Python task case IDs must be unique")
        return case_ids

    def _selected_cases(
        self, evaluation_set: EvaluationSet
    ) -> list[tuple[str, object]]:
        self._validate_evaluation_set(evaluation_set)
        cases = self._cases(evaluation_set)
        case_ids = self._case_ids(evaluation_set)
        selection = evaluation_set.selection
        if isinstance(selection, AllCases):
            indexes = list(range(len(cases)))
        elif isinstance(selection, CaseRange):
            indexes = list(range(selection.start, selection.stop))
        elif isinstance(selection, CaseIds):
            by_id = {case_id: index for index, case_id in enumerate(case_ids)}
            indexes = [by_id[case_id] for case_id in selection.ids]
        else:  # pragma: no cover - closed discriminated union
            raise AssertionError(f"unsupported case selection: {selection}")
        return [(case_ids[index], cases[index]) for index in indexes]

    async def export_case_resources(
        self,
        *,
        evaluation_set: EvaluationSet,
        destination: str,
        sandbox: Sandbox,
    ) -> None:
        index = []
        for case_id, case in self._selected_cases(evaluation_set):
            digest = hashlib.sha256(case_id.encode()).hexdigest()
            filename = f"{digest}.json"
            await sandbox.write_file(
                str(PurePosixPath(destination) / filename),
                json.dumps(case, ensure_ascii=False, indent=2, default=str) + "\n",
            )
            index.append({"case_id": case_id, "path": filename})
        await sandbox.write_file(
            str(PurePosixPath(destination) / "index.json"),
            json.dumps(
                {
                    "schema_version": 1,
                    "evaluation_set": evaluation_set.model_dump(mode="json"),
                    "cases": index,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    def _validate_evaluation_set(self, evaluation_set: EvaluationSet) -> None:
        self._source(evaluation_set)

        case_ids = self._case_ids(evaluation_set)
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
            return EvaluationCost(cases=len(self._case_ids(evaluation_set)))
        raise AssertionError(f"unsupported case selection: {selection}")

    def validate_request(self, request: EvaluationRequest) -> None:
        self._command(request.evaluation_set).validate_request(request)
        self._validate_evaluation_set(request.evaluation_set)

    def sanitize_error(self, message: str) -> str:
        source = self.config.evaluations[0]
        return self._command(
            EvaluationSet(name=source.name, partition=source.partition)
        ).sanitize_error(message)

    async def evaluate(
        self,
        *,
        context: EvaluationContext,
        request: EvaluationRequest,
    ) -> EvaluationReport:
        target_root = context.workspace.sandbox.host_path(
            context.workspace.project_path
        )
        if target_root is not None:
            target_root = target_root.resolve()
            cases_path = Path(self._source(request.evaluation_set).cases_path).resolve()
            if cases_path == target_root or cases_path.is_relative_to(target_root):
                raise ValueError(
                    "Python task cases must live outside the editable target"
                )
        return await self._command(request.evaluation_set).evaluate(
            context=context,
            request=request,
        )
