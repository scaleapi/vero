from __future__ import annotations

import json
import logging
import os
import random
import traceback
from pathlib import Path

import yaml
from rich.panel import Panel
from rich.syntax import Syntax

from .core.cli_adapters import UvRunParameters
from .core.constants import (
    evaluation_parameters_basename,
    evaluation_results_basename,
    pytest_report_basename,
    result_metadata_basename,
    samples_dir_name,
)
from .core.db.candidate import Candidate
from .core.db.database import Experiment, ExperimentDatabase
from .core.db.dataset import DatasetSubset
from .core.db.result import ExperimentResult, SampleResult
from .core.db.run import ExperimentRun
from .core.evaluation import BaseEvaluationParameters, EvaluationParameters
from .core.sessions import (
    clear_result_cache,
    get_experiment_dir,
    get_session_dir,
    get_vero_home_dir,
    initialize_result_store,
    load_all_sample_results,
    save_json_to_cache,
)
from .core.task.utils import get_discover_cmd, get_run_cmd
from .exceptions import ExperimentRunFailedError
from .logging import setup_console
from .utils import run_subprocess_with_tee
from .workspace import Workspace
from .workspace.git import GitWorkspace

console = setup_console()

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates experiment runs by checking out commits and running tasks in subprocesses."""

    def __init__(
        self,
        workspace: Workspace,
        session_id: str,
        *,
        vero_home: Path | None = None,
        use_copy: bool = False,
        hooks: list[str] | None = None,
        sync: bool = False,
        subprocess_env_vars: list | Path | str | None = None,
        task_project: Path | None = None,
        task_module: str | None = None,
    ):
        self.workspace = workspace
        self.session_id = session_id
        self.vero_home = vero_home or get_vero_home_dir()
        self.use_copy = use_copy
        self.hooks = hooks if hooks is not None else ["setup_logging"]
        self.sync = sync
        self._subprocess_env_vars = subprocess_env_vars
        self.task_project = task_project
        self.task_module = task_module
        self.on_experiment: list = []  # Callbacks fired after each evaluate()

    @property
    def sessions_dir(self) -> Path:
        return self.vero_home / "sessions"

    @property
    def dataset_cache(self) -> Path:
        return self.vero_home / "datasets"

    @property
    def subprocess_env(self) -> dict[str, str] | None:
        """Build subprocess env on demand from var names. Returns None to inherit os.environ."""
        if self._subprocess_env_vars is None:
            return None
        from vero.utils.subprocess_env import build_subprocess_env

        return build_subprocess_env(self._subprocess_env_vars)

    def _get_subprocess_env_with_vero_home(self) -> dict[str, str]:
        """Build subprocess env and ensure VERO_HOME_DIR is set."""
        env = self.subprocess_env
        if env is None:
            from vero.utils.subprocess_env import build_subprocess_env

            env = build_subprocess_env()
        env["VERO_HOME_DIR"] = str(self.vero_home)
        return env

    @staticmethod
    def log_evaluation_results(result: ExperimentResult) -> None:
        """Logs the evaluation results to the console."""
        stats = (
            result.sample_results_statistics(
                as_dict=True, convert_lists_to_strings=True
            )
            or {}
        )
        if len(stats) > 0:
            syntax = Syntax(
                yaml.dump(stats, sort_keys=False),
                "yaml",
                theme="monokai",
                line_numbers=False,
            )
            console.print(
                Panel(
                    syntax,
                    title="[bold green]⚙️  Evaluation Statistics[/bold green]",
                    border_style="green",
                )
            )
        else:
            console.print(f"No ExperimentResult found for run {result.run_id}.")

    def load_sample_results_from_cache(
        self, evaluation_parameters: EvaluationParameters
    ) -> dict[int, SampleResult]:
        """Load the sample results from the cache.

        Tries to load from per-sample files first (new format), then falls back
        to the single JSON file (legacy format) for backward compatibility.
        """
        sample_results = load_all_sample_results(
            self.sessions_dir, self.session_id, evaluation_parameters.result_id
        )

        if not sample_results:
            logger.warning(
                f"No sample results found for run {evaluation_parameters.run.id}."
            )

        return sample_results

    def _get_uv_params(
        self, agent_project_path: Path | str
    ) -> tuple[UvRunParameters, Path | str]:
        """Build UvRunParameters and determine cwd for subprocess.

        When task_project is set, runs uv in the task project and layers
        the agent code on top via --with-editable. Otherwise runs in the
        agent project directly (backward compat).

        Returns:
            (uv_params, cwd) tuple.
        """
        if self.task_project:
            return (
                UvRunParameters.from_env(
                    project=str(self.task_project),
                    with_editable=str(agent_project_path),
                ),
                self.task_project,
            )
        return UvRunParameters.from_env(
            project=str(agent_project_path)
        ), agent_project_path

    async def _discover_tasks(self, project_path: Path | str) -> dict:
        """Discover tasks via isolated subprocess.

        Args:
            project_path: Path to the agent project.

        Returns:
            Dictionary with package name and task metadata.
        """
        uv_params, cwd = self._get_uv_params(project_path)
        cmd = [*uv_params.get_cmd(), *get_discover_cmd(task_module=self.task_module)]
        result = await run_subprocess_with_tee(
            cmd,
            timeout=60,
            cwd=str(cwd),
            flush=False,
            tee_stdout=False,
            env=self._get_subprocess_env_with_vero_home(),
        )

        if result.returncode != 0:
            raise ExperimentRunFailedError(
                f"Task discovery failed. Error: {result.stderr}.",
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=int(result.returncode),
            )

        return json.loads(result.stdout)

    async def _run_task(
        self,
        project_path: Path | str,
        task_name: str,
        params_file: Path,
        timeout: int = 60 * 10,
    ) -> dict | None:
        """Execute task via isolated subprocess.

        Args:
            project_path: Path to the user's project.
            task_name: Name of the task to execute.
            params_file: Path to JSON file containing EvaluationParameters.
            timeout: Subprocess timeout in seconds.

        Returns:
            Metrics dictionary from task execution, or None if parsing fails.
        """
        uv_params, cwd = self._get_uv_params(project_path)
        cmd = [
            *uv_params.get_cmd(),
            *get_run_cmd(
                task_name, params_file, hooks=self.hooks, task_module=self.task_module
            ),
        ]
        result = await run_subprocess_with_tee(
            cmd,
            timeout=timeout,
            cwd=cwd,
            flush=True,
            env=self._get_subprocess_env_with_vero_home(),
        )
        logger.info("Subprocess complete!")

        # Save subprocess output for debugging
        log_dir = params_file.parent
        if result.stderr:
            (log_dir / "subprocess_stderr.log").write_text(result.stderr)
        if result.stdout:
            (log_dir / "subprocess_stdout.log").write_text(result.stdout)
        if result.returncode != 0:
            (log_dir / "subprocess_returncode.txt").write_text(str(result.returncode))
            logger.warning(
                f"Subprocess exited with code {result.returncode}. "
                f"Stderr: {result.stderr[:500] if result.stderr else '(empty)'}"
            )

        # Read metrics from file (written by task subprocess)
        metrics_path = log_dir / "metrics.json"
        if metrics_path.exists():
            try:
                return json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse {metrics_path} as JSON")
                return None
        else:
            logger.warning(f"Metrics file not found at {metrics_path}")
            return None

    async def _run_task_in_subprocess(
        self,
        params: EvaluationParameters,
        workspace: Workspace,
    ) -> None:
        """Run task via vero.task_utils subprocess.

        Args:
            params: Evaluation parameters (must have task set).
            workspace: Workspace to run in.

        Raises:
            ExperimentRunFailedError: If task discovery or execution fails.
        """

        # Discover available tasks first
        try:
            discovery_result = await self._discover_tasks(workspace.project_path)
        except Exception as e:
            error_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            raise ExperimentRunFailedError(
                f"Task discovery failed. Error: {error_str}.",
                stdout="",
                stderr=error_str,
                returncode=1,
            )

        # Validate the requested task exists
        available_tasks = discovery_result.get("tasks", {})
        if params.task not in available_tasks:
            available_names = list(available_tasks.keys())
            raise ExperimentRunFailedError(
                f"Task '{params.task}' not found in package '{discovery_result.get('package', 'unknown')}'.\n"
                f"Available tasks: {available_names if available_names else '(none found)'}\n"
                f"Ensure your task is registered in vero_tasks/__init__.py",
                stdout="",
                stderr="",
                returncode=1,
            )

        # Validate required environment variables
        required_env = available_tasks[params.task].get("required_env_vars", [])
        if required_env:
            missing = [v for v in required_env if not os.environ.get(v)]
            if missing:
                raise ExperimentRunFailedError(
                    f"Task '{params.task}' requires environment variables that are not set: "
                    f"{', '.join(missing)}. Set them before running.",
                    stdout="",
                    stderr="",
                    returncode=1,
                )

        # Run the task
        result_dir = get_experiment_dir(
            self.sessions_dir, self.session_id, params.result_id
        )
        params_file = result_dir / evaluation_parameters_basename
        logger.info(
            f"Running task '{params.task}' via vero.task_utils in {workspace.project_path}"
        )
        try:
            metrics = await self._run_task(
                workspace.project_path,
                params.task,
                params_file,
                timeout=params.timeout,
            )
            logger.info(f"Task completed with metrics: {metrics}")
        except Exception as e:
            error_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            raise ExperimentRunFailedError(
                f"Task execution failed. Error: {error_str}.",
                stdout="",
                stderr=error_str,
                returncode=1,
            )

    async def run(
        self,
        evaluation_parameters: EvaluationParameters,
        use_copy: bool | None = None,
    ) -> ExperimentResult:
        """Run an experiment by checking out the candidate commit and running tasks via uv.

        Args:
            evaluation_parameters: The parameters for the evaluation.
            use_copy: Override for self.use_copy. If True, creates a temporary isolated copy
                of the workspace (always clean). If False, uses the current workspace (requires clean state).

        Returns:
            ExperimentResult with sample results and metadata.
        """
        use_copy = use_copy if use_copy is not None else self.use_copy

        if not use_copy:
            return await self._run_in_workspace(evaluation_parameters, self.workspace)

        async with self.workspace.temp_copy(
            from_version=evaluation_parameters.run.candidate.commit,
        ) as temp_workspace:
            return await self._run_in_workspace(evaluation_parameters, temp_workspace)

    async def evaluate(
        self,
        commit: str,
        dataset_id: str,
        split: str,
        task: str | None = None,
        sample_ids: list[int] | None = None,
        db: ExperimentDatabase | None = None,
        evaluation_parameters: BaseEvaluationParameters | None = None,
        use_copy: bool | None = None,
    ) -> Experiment:
        """Full evaluation lifecycle: resolve commit → run → create experiment → DB → hooks.

        This is the single entry point for all evaluations. Both Policy.evaluate_commit()
        and ExperimentRunnerTool delegate here.

        Args:
            commit: Git commit hash or ref to evaluate.
            dataset_id: Dataset ID in the session store.
            split: Dataset split to evaluate.
            task: Task name to execute.
            sample_ids: Specific sample IDs to evaluate (None = all).
            db: ExperimentDatabase to record the experiment in.
            evaluation_parameters: Base eval params (timeout, concurrency, etc.).
            use_copy: Whether to create a temporary copy for the eval.

        Returns:
            The completed Experiment with results.
        """
        from .core.db.database import Experiment

        # Resolve commit ref to canonical version ID
        try:
            if isinstance(self.workspace, GitWorkspace):
                full_hash = await self.workspace.resolve_ref(commit)
            else:
                full_hash = commit
        except Exception as e:
            raise ValueError(
                f"Cannot resolve commit '{commit}': {e}. "
                f"Make sure the commit exists in the repository."
            )

        # Build candidate
        candidate = None
        if db is not None:
            candidate = db.get_candidate((self.workspace.name, full_hash))
        if candidate is None:
            candidate = Candidate(commit=full_hash, repo_name=self.workspace.name)

        # Build run
        dataset_subset = DatasetSubset(
            split=split, sample_ids=sample_ids, dataset_id=dataset_id
        )
        run = ExperimentRun(candidate=candidate, dataset_subset=dataset_subset)

        # Build eval params
        base_params = evaluation_parameters or BaseEvaluationParameters()
        params = EvaluationParameters(
            **base_params.model_dump(),
            run=run,
            dataset_id=dataset_id,
            task=task,
            session_id=self.session_id,
        )

        # Run
        result = await self.run(params, use_copy=use_copy)

        # Create experiment
        experiment = Experiment(run=run, result=result)

        # Add to DB
        if db is not None:
            db.add_experiment(experiment)

        # Fire post-eval callbacks (may be sync or async)
        import asyncio as _asyncio

        for callback in self.on_experiment:
            try:
                result = callback(experiment)
                if _asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"on_experiment callback failed: {e}")

        return experiment

    async def _run_in_workspace(
        self, params: EvaluationParameters, workspace: Workspace
    ) -> ExperimentResult:
        """Run an experiment by checking out the candidate commit and running tasks via uv."""

        # We cannot execute with a dirty workspace, as this may introduce side effects on the evaluation results.
        if await workspace.is_dirty():
            raise RuntimeError(
                "Evaluator cannot execute. There are unsaved changes in the workspace."
            )

        # Update the evaluation parameters with the dataset loader and session_id
        params.session_id = self.session_id

        # Initialize the directory to store the evaluation and pytest report files
        result_dir = initialize_result_store(
            self.sessions_dir, self.session_id, params.result_id
        )

        save_json_to_cache(
            self.sessions_dir,
            self.session_id,
            params.result_id,
            basename=evaluation_parameters_basename,
            data=params,
        )
        logger.info(
            f"Saved evaluation parameters to cache: {result_dir / evaluation_parameters_basename}"
        )

        # Git-specific: fetch from remote if configured
        if self.sync and isinstance(workspace, GitWorkspace):
            await workspace.maybe_fetch()

        # Clear any stale cached results before running to avoid reading old data if run fails
        clear_result_cache(
            self.sessions_dir,
            self.session_id,
            params.result_id,
            result_basenames=[pytest_report_basename, evaluation_results_basename],
        )

        # Transfer data into the sandbox before running
        experiment_dir = str(
            get_experiment_dir(self.sessions_dir, self.session_id, params.result_id)
        )
        await workspace.sandbox.upload(experiment_dir, experiment_dir)

        # Upload dataset cache so subprocess can load it
        from vero.core.dataset.store import _read_mapping

        mapping = _read_mapping(self.sessions_dir, self.session_id)
        dataset_fp = mapping.get(params.dataset_id or "")
        if dataset_fp:
            cache_entry = str(self.dataset_cache / dataset_fp)
            await workspace.sandbox.upload(cache_entry, cache_entry)
        # Also upload the session datasets.json mapping
        session_dir = str(get_session_dir(self.sessions_dir, self.session_id))
        datasets_json = f"{session_dir}/datasets.json"
        await workspace.sandbox.upload(datasets_json, datasets_json)

        # Switch to the candidate version and run the evaluation in a subprocess
        async with workspace.at(params.run.candidate.commit):
            await self._run_task_in_subprocess(params, workspace)

        # Transfer results back from the sandbox
        await workspace.sandbox.download(experiment_dir, experiment_dir)

        sample_results = self.load_sample_results_from_cache(params)

        if not sample_results:
            raise ExperimentRunFailedError(
                f"No sample results found for run {params.run.id}! Likely because execution failed.",
                returncode=1,
            )
        else:
            result = ExperimentResult.create_with_status(
                id=params.result_id,
                error_rate=params.error_rate_threshold,
                run_id=params.run.id,
                sample_results=sample_results,
            )

            # Write result metadata to disk so the DB can be reconstructed from experiments/
            save_json_to_cache(
                self.sessions_dir,
                self.session_id,
                params.result_id,
                basename=result_metadata_basename,
                data={
                    "id": result.id,
                    "run_id": result.run_id,
                    "status": result.status.value,
                },
            )

            self.log_evaluation_results(result)
            return result


def _resolve_vero_dependency(isolated_dir: Path, original_project_dir: Path) -> None:
    """Resolve the vero path dependency in pyproject.toml after isolation.

    When a project is isolated (copied to a new location), relative path
    dependencies in ``[tool.uv.sources]`` break. This function resolves
    the ``scale-vero`` dependency to an absolute path via ``uv add``.

    Raises ValueError if any *other* relative path dependencies are found,
    since those are unsupported and would silently break.
    """
    import subprocess
    import tomllib

    pyproject_path = isolated_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    if not sources:
        return

    for name, source in sources.items():
        if not isinstance(source, dict) or "path" not in source:
            continue

        rel_path = source["path"]
        if not rel_path.startswith(".") and not rel_path.startswith("/"):
            continue  # Not a relative path

        if "vero" in name.lower():
            # Always resolve to the known vero package directory rather than
            # trusting the relative path (which may be stale or wrong).
            from vero.core.constants import PACKAGE_DIR

            abs_path = PACKAGE_DIR
            editable_flag = ["--editable"] if source.get("editable") else []
            subprocess.run(
                ["uv", "add", *editable_flag, "--dev", str(abs_path)],
                cwd=isolated_dir,
                capture_output=True,
                check=True,
            )
            logger.info(f"Resolved {name} dependency: {rel_path} -> {abs_path}")
        else:
            raise ValueError(
                f"Unsupported relative path dependency '{name}' "
                f"(path={rel_path!r}) in {pyproject_path}. "
                f"Only vero is handled during isolation."
            )


def isolate_project(
    project_path: Path | str,
    session_id: str,
    git_ref: str = "HEAD",
    *,
    sessions_dir: Path,
) -> Path:
    """Copy a project into a fresh, standalone git repo.

    Useful when the project lives inside a monorepo or has uncommitted changes.
    Extracts files at *git_ref* via ``git archive`` (falling back to a plain
    copy when the source is not a git repo), then ``git init`` + ``git commit``
    so the result is a clean, self-contained repository.

    Relative path dependencies on vero in pyproject.toml are resolved to
    absolute paths so they remain valid after the copy.

    Args:
        project_path: Path to the project directory.
        session_id: Session ID (isolated copy is placed under the session dir).
        git_ref: Git ref to archive from (default: HEAD).
        sessions_dir: Path to the sessions root directory.

    Returns:
        Path to the isolated project root.
    """
    import shutil
    import subprocess

    project_path = Path(project_path).resolve()
    isolated_dir = (sessions_dir / session_id) / project_path.name
    isolated_dir.mkdir(parents=True, exist_ok=True)

    repo_root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project_path,
        capture_output=True,
        text=True,
    )

    if repo_root_result.returncode == 0:
        repo_root_path = Path(repo_root_result.stdout.strip())
        project_rel = project_path.relative_to(repo_root_path)
        strip = len(project_rel.parts)

        archive = subprocess.Popen(
            ["git", "archive", git_ref, str(project_rel)],
            cwd=repo_root_path,
            stdout=subprocess.PIPE,
        )
        subprocess.run(
            ["tar", "xf", "-", "--strip-components", str(strip)],
            cwd=isolated_dir,
            stdin=archive.stdout,
            check=True,
        )
        archive.wait()
    else:
        shutil.copytree(project_path, isolated_dir, dirs_exist_ok=True)

    # Resolve vero dependency before git init (so it's in the initial commit)
    _resolve_vero_dependency(isolated_dir, project_path)

    subprocess.run(["git", "init"], cwd=isolated_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "add", "."], cwd=isolated_dir, capture_output=True, check=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "Initial commit (isolated)",
        ],
        cwd=isolated_dir,
        capture_output=True,
        check=True,
    )

    if repo_root_result.returncode == 0:
        subprocess.run(
            ["git", "remote", "add", "origin", repo_root_result.stdout.strip()],
            cwd=isolated_dir,
            capture_output=True,
        )

    logger.info(f"Isolated project: {project_path} -> {isolated_dir}")
    return isolated_dir


async def run_evaluation(
    project_path: Path | str,
    dataset: str | Path,
    split: str,
    task: str | None = None,
    commit: str | None = None,
    sample_ids: list[int] | None = None,
    num_samples: int | None = None,
    task_params: dict | None = None,
    seed: int = 42,
    timeout: int = 3600,
    per_sample_timeout: int = 180,
    create_temporary_copy: bool = False,
    isolate: bool = False,
    hooks: list[str] | None = None,
    session_id: str | None = None,
    max_concurrency: int | None = None,
    subprocess_env_vars: list[str] | Path | str | None = None,
    task_project: Path | str | None = None,
    task_module: str | None = None,
    vero_home: Path | None = None,
) -> ExperimentResult:
    """Run an evaluation using the given parameters.

    Args:
        project_path: Path to the agent project to evaluate.
        dataset: Dataset, DatasetDict, path to saved dataset, or dataset ID string.
        split: Dataset split to evaluate.
        task: Task name to execute from vero_tasks module.
        commit: Commit to evaluate.
        sample_ids: List of sample IDs to evaluate.
        num_samples: Number of samples to evaluate.
        task_params: Task-specific parameters for the evaluation.
        seed: Random seed for sample selection.
        timeout: Overall timeout for the evaluation subprocess in seconds.
        per_sample_timeout: Timeout for a single sample in seconds.
        create_temporary_copy: Whether to create a temporary copy for the evaluation.
        isolate: Whether to copy the project into a fresh git repo before evaluating.
        hooks: List of hook names to execute before task.
        session_id: Session ID.
        max_concurrency: Maximum concurrent tasks.
        subprocess_env_vars: Environment variable names to pass to task subprocesses.
        task_project: Path to a separate task project. When set, evaluator runs
            uv in the task project and layers the agent via --with-editable.
        task_module: Explicit Python module to import for task registration
            (e.g. "my_eval_tasks.vero_tasks"). If None, auto-discovers.
        vero_home: Path to the vero home directory. Defaults to ~/.vero.

    Returns:
        The experiment result.

    Raises:
        ExperimentRunFailedError: If the evaluation fails.
    """
    from vero.core.dataset.store import resolve_and_save_dataset

    vh = vero_home or get_vero_home_dir()
    sessions_dir = vh / "sessions"
    dataset_cache = vh / "datasets"

    if task_params is None:
        task_params = {}

    if session_id is None:
        from uuid import uuid4

        session_id = str(uuid4())
        logger.info(f"Auto-generated session ID: {session_id}")

    if isolate:
        project_path = isolate_project(
            project_path, session_id, sessions_dir=sessions_dir
        )

    workspace = await GitWorkspace.create(project_path)

    # Resolve and save dataset
    dataset_id = resolve_and_save_dataset(
        dataset, sessions_dir, dataset_cache, session_id
    )

    evaluator = Evaluator(
        workspace=workspace,
        use_copy=create_temporary_copy,
        hooks=hooks,
        session_id=session_id,
        vero_home=vh,
        subprocess_env_vars=subprocess_env_vars,
        task_project=Path(task_project) if task_project else None,
        task_module=task_module,
    )

    if commit is None:
        commit = await workspace.current_version()
        logger.warning(f"No commit provided, using current commit: {commit}.")

    # Sample data if num_samples is provided
    if num_samples is not None and sample_ids is None:
        from vero.core.dataset.store import load_dataset as _load_ds

        ds = _load_ds(sessions_dir, dataset_cache, session_id, dataset_id)
        rng = random.Random(seed)
        sample_ids = rng.sample(range(len(ds[split])), num_samples)

    # Build base eval params
    eval_params = BaseEvaluationParameters(
        timeout=timeout,
        sample_timeout=per_sample_timeout,
        task_params=task_params,
    )
    if max_concurrency is not None:
        eval_params.max_concurrency = max_concurrency

    experiment = await evaluator.evaluate(
        commit=commit,
        dataset_id=dataset_id,
        split=split,
        task=task,
        sample_ids=sample_ids,
        evaluation_parameters=eval_params,
        use_copy=create_temporary_copy,
    )

    result_dir = get_experiment_dir(sessions_dir, session_id, experiment.id)
    console.print(f"Result available at {result_dir / samples_dir_name}")
    return experiment.result
