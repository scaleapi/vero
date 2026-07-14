"""Compatibility backend for the existing Python/uv VeroTask evaluator."""

from __future__ import annotations

import math
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import Field

from vero.core.constants import default_minimum_score
from vero.core.db.result import ExperimentResultStatus
from vero.core.evaluation import BaseEvaluationParameters
from vero.evaluation.backend import EvaluationContext
from vero.evaluation.legacy import _convert_case
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
    EvaluationStatus,
)
from vero.evaluation.security import sanitize_evaluation_report, sanitize_text
from vero.evaluator import Evaluator as LegacyEvaluator


class VeroTaskBackendConfig(EvaluationModel):
    session_id: str
    vero_home: str
    task: str | None = None
    task_project: str | None = None
    task_module: str | None = None
    hooks: list[str] = Field(default_factory=lambda: ["setup_logging"])
    subprocess_env_vars: list[str] = Field(default_factory=list)
    error_rate_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    use_threading: bool = False


class _CheckedOutWorkspace:
    """Delegate to a workspace while making its current candidate checkout explicit."""

    def __init__(self, workspace):
        self._workspace = workspace

    def __getattr__(self, name):
        return getattr(self._workspace, name)

    @property
    def sandbox(self):
        return self._workspace.sandbox

    @property
    def root(self):
        return self._workspace.root

    @property
    def project_path(self):
        return self._workspace.project_path

    @property
    def name(self):
        return self._workspace.name

    async def current_version(self):
        return await self._workspace.current_version()

    async def is_dirty(self):
        return await self._workspace.is_dirty()

    @asynccontextmanager
    async def at(self, version_id: str):
        current = await self.current_version()
        if current != version_id:
            raise ValueError(
                f"checked-out workspace is at {current!r}, expected {version_id!r}"
            )
        yield


class VeroTaskBackend:
    name = "vero-task"
    version = "1"

    def __init__(
        self,
        config: VeroTaskBackendConfig,
        *,
        subprocess_env_vars: Any = None,
    ):
        self.config = config
        # Environment callables are runtime capabilities and deliberately do not
        # enter the serializable provenance config. The config records only the
        # declared variable names.
        self.subprocess_env_vars = (
            subprocess_env_vars
            if subprocess_env_vars is not None
            else config.subprocess_env_vars
        )
        self._known_secrets: set[str] = set()
        self._known_secrets.update(self._secret_values())

    @property
    def provenance(self) -> BackendProvenance:
        return BackendProvenance.from_config(
            name=self.name,
            version=self.version,
            config=self.config,
        )

    @property
    def sessions_dir(self) -> Path:
        return Path(self.config.vero_home) / "sessions"

    @property
    def dataset_cache(self) -> Path:
        return Path(self.config.vero_home) / "datasets"

    def _secret_values(self) -> list[str]:
        values = list(self._known_secrets)
        values.extend(
            os.environ[name]
            for name in self.config.subprocess_env_vars
            if name in os.environ
        )
        source = self.subprocess_env_vars
        if isinstance(source, (str, Path)) and Path(source).is_file():
            from vero.utils.subprocess_env import load_env_file

            values.extend(load_env_file(source).values())
        return values

    def _resolve_environment_source(self) -> tuple[Any, list[str]]:
        source = self.subprocess_env_vars
        secrets = self._secret_values()
        if not isinstance(source, list):
            return source, secrets

        resolved = []
        for specification in source:
            if isinstance(specification, str):
                resolved.append(specification)
                value = os.environ.get(specification)
            else:
                name, factory = specification
                value = factory()
                resolved.append((name, lambda value=value: value))
            if value:
                secrets.append(value)
                self._known_secrets.add(value)
        return resolved, secrets

    def sanitize_error(self, message: str) -> str:
        return sanitize_text(message, self._secret_values())

    def validate_request(self, request: EvaluationRequest) -> None:
        payload = request.model_dump_json()
        if any(
            secret in payload
            for secret in self._secret_values()
            if len(secret) >= 4
        ):
            raise ValueError(
                "task parameters must not contain configured secret values; pass "
                "secrets through subprocess_env_vars"
            )

    def _sanitize_staging_logs(
        self,
        staging_session: Path,
        secrets: list[str],
    ) -> None:
        if not secrets:
            return
        experiments = staging_session / "experiments"
        if not experiments.exists():
            return
        for path in experiments.rglob("*"):
            if not path.is_file() or path.stat().st_size > 10_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            sanitized = sanitize_text(text, secrets)
            if sanitized != text:
                path.write_text(sanitized, encoding="utf-8")

    def _split_size(self, evaluation_set: EvaluationSet) -> int:
        from vero.core.dataset.store import load_dataset

        if evaluation_set.partition is None:
            raise ValueError("VeroTask evaluation sets require a dataset partition")
        dataset = load_dataset(
            self.sessions_dir,
            self.dataset_cache,
            self.config.session_id,
            evaluation_set.name,
        )
        if evaluation_set.partition not in dataset:
            raise ValueError(
                f"dataset {evaluation_set.name!r} has no partition "
                f"{evaluation_set.partition!r}"
            )
        return len(dataset[evaluation_set.partition])

    def _sample_ids(self, evaluation_set: EvaluationSet) -> list[int] | None:
        size = self._split_size(evaluation_set)
        selection = evaluation_set.selection
        if isinstance(selection, AllCases):
            return None
        if isinstance(selection, CaseIds):
            sample_ids = []
            for case_id in selection.ids:
                try:
                    sample_id = int(case_id)
                except ValueError as error:
                    raise ValueError(
                        f"VeroTask case ID must be a non-negative integer: {case_id!r}"
                    ) from error
                if sample_id < 0 or str(sample_id) != case_id:
                    raise ValueError(
                        f"VeroTask case ID must be a canonical non-negative integer: {case_id!r}"
                    )
                if sample_id >= size:
                    raise ValueError(
                        f"VeroTask case ID {sample_id} is outside partition of size {size}"
                    )
                sample_ids.append(sample_id)
            return sample_ids
        if isinstance(selection, CaseRange):
            if selection.start >= size:
                raise ValueError(
                    f"case range starts at {selection.start}, outside partition of size {size}"
                )
            return list(range(selection.start, min(selection.stop, size)))
        raise AssertionError(f"unsupported selection: {selection}")

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost:
        sample_ids = self._sample_ids(evaluation_set)
        cases = self._split_size(evaluation_set) if sample_ids is None else len(sample_ids)
        return EvaluationCost(cases=cases)

    async def evaluate(
        self,
        *,
        context: EvaluationContext,
        request: EvaluationRequest,
    ) -> EvaluationReport:
        if request.evaluation_set.partition is None:
            raise ValueError("VeroTask evaluation sets require a dataset partition")
        sample_ids = self._sample_ids(request.evaluation_set)
        workspace = _CheckedOutWorkspace(context.workspace)

        # The legacy task runner still speaks its original file protocol. Keep
        # those files in a backend-private staging home inside this canonical
        # evaluation instead of emitting sibling schema-v1 result directories.
        staging_home = context.result_dir / "backend-staging"
        staging_session = staging_home / "sessions" / self.config.session_id
        staging_session.mkdir(parents=True, exist_ok=True)
        source_mapping = (
            Path(self.config.vero_home)
            / "sessions"
            / self.config.session_id
            / "datasets.json"
        )
        if source_mapping.exists():
            shutil.copy2(source_mapping, staging_session / "datasets.json")
        else:
            # Cost resolution normally proves this exists. Keeping the staging
            # layout valid also makes the backend independently testable.
            (staging_session / "datasets.json").write_text("{}\n")
        source_cache = Path(self.config.vero_home) / "datasets"
        staging_cache = staging_home / "datasets"
        if source_cache.exists() and not staging_cache.exists():
            staging_cache.symlink_to(source_cache.resolve(), target_is_directory=True)
        elif not staging_cache.exists():
            staging_cache.mkdir()

        environment_source, secret_values = self._resolve_environment_source()
        evaluator = LegacyEvaluator(
            workspace,
            self.config.session_id,
            vero_home=staging_home,
            use_copy=False,
            hooks=self.config.hooks,
            subprocess_env_vars=environment_source,
            task_project=Path(self.config.task_project)
            if self.config.task_project
            else None,
            task_module=self.config.task_module,
        )
        parameters = BaseEvaluationParameters(
            max_concurrency=request.limits.max_concurrency,
            error_rate_threshold=self.config.error_rate_threshold,
            timeout=request.limits.timeout_seconds,
            sample_timeout=request.limits.case_timeout_seconds,
            task_params=request.parameters,
            retry_config=request.limits.retry_config,
            use_threading=self.config.use_threading,
        )
        experiment = await evaluator.evaluate(
            commit=request.candidate.commit,
            dataset_id=request.evaluation_set.name,
            split=request.evaluation_set.partition,
            task=self.config.task,
            sample_ids=sample_ids,
            evaluation_parameters=parameters,
            use_copy=False,
        )
        self._sanitize_staging_logs(staging_session, secret_values)
        cases = []
        diagnostics = []
        for sample_id, sample in sorted(experiment.result.sample_results.items()):
            case, case_diagnostics = _convert_case(sample_id, sample)
            cases.append(case)
            diagnostics.extend(case_diagnostics)

        score = experiment.result.score(fill_score=default_minimum_score)
        metrics = {
            "error_rate": experiment.result.error_rate(),
            "num_results": float(len(cases)),
        }
        if score is not None and math.isfinite(score):
            metrics["score"] = score
        status = (
            EvaluationStatus.SUCCESS
            if experiment.result.status == ExperimentResultStatus.SUCCESS
            else EvaluationStatus.FAILED
        )
        report = EvaluationReport(
            status=status,
            metrics=metrics,
            cases=cases,
            diagnostics=diagnostics,
            error="VeroTask evaluation failed"
            if status == EvaluationStatus.FAILED
            else None,
        )
        return sanitize_evaluation_report(report, secret_values)
