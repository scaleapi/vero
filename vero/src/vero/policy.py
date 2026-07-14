from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from vero.agents.base import BaseAgent
from vero.artifacts import FileSystemArtifact
from vero.core.dataset import (
    SplitAccess,
    default_split_accesses,
)
from vero.core.db.database import Experiment, ExperimentDatabase
from vero.core.evaluation import BaseEvaluationParameters
from vero.filesystem import AccessRule, AccessType
from vero.logging import (
    SessionLogger,
    log_evaluations_to_wandb,
    log_experiments_to_wandb,
)
from vero.sandbox import Sandbox
from vero.session import BestVersion, Session
from vero.tools import ToolRegistry
from vero.tools.experiment_runner import SplitBudget
from vero.utils import random_readable_id, recursively_serialize
from vero.workspace import GitWorkspace, Workspace

if TYPE_CHECKING:
    import wandb
    from datasets import DatasetDict
    from jinja2 import Template

    DatasetT = Path | str | DatasetDict

    from vero.evaluation import (
        BudgetLedger,
        EvaluationBackend,
        EvaluationDatabase,
        EvaluationEngine,
        EvaluationLimits,
        EvaluationRecord,
        EvaluationSet,
        ObjectiveSpec,
    )
    from vero.optimization import CandidateProducer, OptimizationRun, ProgramPolicy

logger = logging.getLogger(__name__)

__all__ = ["Policy", "Session", "BestVersion"]


@dataclass
class Policy:
    """Composable policy for AI agent optimization.

    Holds all configuration and runtime state. Composes an Agent backend
    for execution. Use as an async context manager:

        async with policy:
            await policy.step(prompt, max_turns)
            await policy.evaluate_version(commit, split)

    Attributes:
        project_path: Path to the agent project (git repo with a uv package).
        dataset: Path to a HuggingFace DatasetDict on disk, or a dict mapping
            dataset IDs to paths.
        agent: Agent backend (VeroAgent or ClaudeCodeAgent).
        task: Task name to execute from the agent's vero_tasks module.
        task_project: Separate uv project containing task/eval code. When set,
            the evaluator runs in this project and layers the agent code via
            ``uv run --with-editable``. Default: same as project_path.
        task_module: Explicit Python module for task registration (e.g.
            ``my_eval_tasks.vero_tasks``). Default: auto-discover from
            ``{package}.vero_tasks``.
        ref: Git ref (branch, tag, or commit) to start from. Default: ``"main"``.
        isolate: Copy the project into a fresh git repo before optimizing.
            Useful for monorepos or dirty working trees.
        use_copy: Create an isolated workspace copy for evaluation (True), use the
            current workspace directly (False), or use a named branch (str).
        skills: Skills/cookbook artifacts to materialize in ``_vero/skills/``.
            A path, dict of ``{namespace: path}``, or dict of
            ``{namespace: {name: content}}``.
        artifacts: Filesystem artifacts to materialize in the agent's workspace
            (e.g. DatasetArtifact, TracesArtifact, SkillsArtifact).
        budget: Explicit per-split budget list. Overrides train_budget /
            validation_budget convenience fields.
        prompt_kwargs: Extra variables passed to Jinja prompt templates.
        subprocess_env_vars: Environment variable names (or callables) to pass
            to evaluation subprocesses.
        evaluation_parameters: Timeout, concurrency, retry settings for evals.
        max_turns: Maximum agent optimization turns. Default: 200.
        filesystem_accesses: Programmatic access rules (overrides .veroaccess).
        filesystem_default_access: Default access level when no rules match.
        split_accesses: Dataset split visibility (viewable vs non-viewable).
        train_budget: Convenience field — evaluation runs on train split.
        validation_budget: Convenience field — evaluation runs on validation split.
        train_sample_budget: Convenience field — sample budget on train split.
        validation_sample_budget: Convenience field — sample budget on validation split.
        instructions_template: Jinja2 template (path or Template) for agent instructions.
        prompt_template: Jinja2 template (path or Template) for per-turn prompts.
        session_id: Unique session identifier. Auto-generated if not provided.
        metadata: Arbitrary metadata dict (logged to wandb, included in config).
        on_event: Callbacks fired with serialized event dicts during agent execution.
        enable_wandb: Enable Weights & Biases experiment logging.
        wandb_project: Wandb project name.
    """

    # --- Core ---
    project_path: Path | str | None = None
    dataset: DatasetT | dict[str, DatasetT] | None = None
    agent: BaseAgent | None = None
    workspace: Workspace | None = None
    optimizer: CandidateProducer | None = None
    backends: dict[str, EvaluationBackend] | None = None
    default_backend_id: str | None = None
    evaluation_set: EvaluationSet | None = None
    objective: ObjectiveSpec | None = None
    program_limits: EvaluationLimits | None = None
    program_parameters: dict[str, Any] = field(default_factory=dict)
    program_seed: int | None = None
    program_budget_ledger: BudgetLedger | None = None
    max_candidates: int = 1
    task: str | None = None
    task_project: Path | str | None = None
    task_module: str | None = None
    ref: str = "main"
    isolate: bool = False
    use_copy: bool | str = True
    skills: Path | str | dict[str, Path | str] | None = None
    artifacts: list[FileSystemArtifact] = field(default_factory=list)
    budget: list[SplitBudget] | None = None
    prompt_kwargs: dict[str, Any] = field(default_factory=dict)
    subprocess_env_vars: (
        list[str | tuple[str, Callable[[], str | None]]] | Path | str | None
    ) = None
    optimizer_env_file: Path | str | None = None
    evaluation_parameters: BaseEvaluationParameters = field(
        default_factory=BaseEvaluationParameters
    )
    max_turns: int = 200

    # --- Permissions ---
    filesystem_accesses: list[AccessRule] | None = None
    filesystem_default_access: AccessType = field(default=AccessType.WRITE)
    split_accesses: list[SplitAccess] = field(
        default_factory=lambda: list(default_split_accesses)
    )

    # Convenience fields — normalized into budget during init
    train_budget: int | None = None
    validation_budget: int | None = None
    train_sample_budget: int | None = None
    validation_sample_budget: int | None = None

    # --- Context ---
    instructions_template: str | Template | None = None
    prompt_template: str | Template | None = None

    # --- Sandbox ---
    sandbox: Sandbox | None = None

    # --- Storage ---
    vero_home: Path | str | None = None

    # --- Metadata ---
    session_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Event callbacks ---
    on_event: list[Callable[[dict], Any]] = field(default_factory=list)

    # --- Logging ---
    enable_event_log: bool = True
    enable_general_log: bool = True
    enable_console: bool = True
    console_verbose: bool = True
    use_default_logging: bool = True

    # --- Wandb ---
    enable_wandb: bool = False
    wandb_project: str = ""
    wandb_run: wandb.Run | None = field(default=None, repr=False)

    # --- Runtime ---
    session: Session | None = field(default=None, repr=False)
    session_logger: SessionLogger | None = field(default=None, repr=False)
    program_policy: ProgramPolicy | None = field(default=None, init=False, repr=False)
    evaluation_engine: EvaluationEngine | None = field(default=None, init=False, repr=False)
    evaluation_db: EvaluationDatabase | None = field(default=None, init=False, repr=False)
    program_run: OptimizationRun | None = field(default=None, init=False, repr=False)

    @property
    def is_program_policy(self) -> bool:
        """Whether this Policy uses generic program evaluation rather than VeroTask."""
        return self.dataset is None and any(
            value is not None
            for value in (
                self.workspace,
                self.optimizer,
                self.backends,
                self.evaluation_set,
                self.objective,
            )
        )

    @property
    def _vero_home(self) -> Path:
        if self.vero_home:
            return Path(self.vero_home)
        from vero.core.sessions import get_vero_home_dir

        return get_vero_home_dir()

    @property
    def sessions_dir(self) -> Path:
        return self._vero_home / "sessions"

    @property
    def dataset_cache(self) -> Path:
        return self._vero_home / "datasets"

    def initialized(self, validate: bool = False) -> bool:
        initialized = bool(self.session and self.session.workspace)

        if validate and not initialized:
            raise ValueError("Session is not initialized! Run `init()` first.")

        return initialized

    async def init(self) -> None:
        """Initialize all runtime state. Call before step()/evaluate_version()."""

        from vero.core.sessions import create_session_dir

        # Load optimizer env file first — sets env vars for LLM calls in the parent process
        if self.optimizer_env_file:
            from vero.utils.subprocess_env import apply_env_file

            apply_env_file(self.optimizer_env_file)

        if self.use_default_logging:
            from vero.logging import setup_logging

            setup_logging()

        # Create session early — populate fields as we go
        if self.session_id is None:
            self.session_id = str(uuid4())
        logger.debug(f"Session ID: {self.session_id}")
        create_session_dir(self.sessions_dir, self.session_id)

        configured_project_path = (
            Path(self.project_path)
            if self.project_path is not None
            else Path(self.workspace.project_path)
            if self.workspace is not None
            else None
        )
        if configured_project_path is None:
            raise ValueError("Policy requires project_path or workspace")

        self.session = Session(
            session_id=self.session_id,
            project_path=configured_project_path,
            vero_home=self._vero_home,
        )

        if self.is_program_policy:
            await self._init_program_policy()
            return

        if self.dataset is None or self.agent is None:
            raise ValueError(
                "VeroTask Policy requires both dataset and agent; generic program "
                "Policy requires backends, evaluation_set, and objective"
            )

        # Git workspace — create via sandbox.run() git commands
        project_path = Path(self.project_path)
        if self.isolate:
            from vero.evaluator import isolate_project

            project_path = isolate_project(
                project_path, self.session_id, self.ref, sessions_dir=self.sessions_dir
            )
            self.ref = "main"  # Fresh repo has only main

        # Create sandbox if not provided. The sandbox is a bare execution
        # environment (the "computer"); access rules are applied later in
        # _prepare_sandbox() after workspace and dataset are resolved.
        if self.sandbox is None:
            from vero.sandbox import LocalSandbox

            self.sandbox = await LocalSandbox.create()

        workspace = await GitWorkspace.from_path(self.sandbox, str(project_path))

        # Resolve ref to a commit hash (the snapshot we start from)
        self.session.base_version = await workspace.resolve_ref(self.ref)

        from datasets import DatasetDict

        # Generate a branch name for the agent to work on
        if isinstance(self.use_copy, str):
            prefix = self.use_copy
        else:
            # infer a name from the dataset
            ds_id = ""
            if isinstance(self.dataset, dict) and not isinstance(
                self.dataset, DatasetDict
            ):
                ds_id = next(iter(self.dataset))
            elif isinstance(self.dataset, (str, Path)) and Path(self.dataset).exists():
                ds_id = Path(self.dataset).stem
            else:
                ds_id = (
                    str(self.dataset) if isinstance(self.dataset, str) else "dataset"
                )
            prefix = f"{workspace.name}-{ds_id}"
        branch_name = f"{prefix}-{random_readable_id()}".replace("_", "-")

        # Create the branch — on a new copy or on the current workspace
        if self.use_copy:
            workspace = await workspace.copy(
                name=branch_name, from_version=self.session.base_version
            )
            logger.info(f"Created workspace copy: {branch_name}")
        else:
            if await workspace.branch_exists(branch_name):
                existing_commit = await workspace.get_head_commit(branch_name)
                if existing_commit != self.session.base_version:
                    raise ValueError(
                        f"Branch '{branch_name}' already exists at {existing_commit[:8]} "
                        f"but base_version is {self.session.base_version[:8]}. "
                        f"Cannot reuse a branch pointing to a different commit."
                    )
                await workspace.checkout_branch(branch_name)
            else:
                await workspace.checkout_branch(
                    branch_name, create=True, from_ref=self.session.base_version
                )
            logger.info(f"Created branch: {branch_name}")

        self.session.base_branch = branch_name
        self.session.workspace = workspace
        self.session.project_path = Path(workspace.project_path)

        # Dataset — resolve and save to store
        from vero.core.dataset.store import resolve_and_save_dataset

        if isinstance(self.dataset, dict) and not isinstance(self.dataset, DatasetDict):
            if len(self.dataset) > 1:
                raise NotImplementedError(
                    "Multiple datasets are not yet supported. "
                    f"Got {len(self.dataset)} datasets: {list(self.dataset.keys())}"
                )
            ds_id, value = next(iter(self.dataset.items()))
            dataset_id = resolve_and_save_dataset(
                value, self.sessions_dir, self.dataset_cache, self.session_id, ds_id
            )
        else:
            dataset_id = resolve_and_save_dataset(
                self.dataset, self.sessions_dir, self.dataset_cache, self.session_id
            )
        self.session.dataset_id = dataset_id

        # Skills (context store artifacts) — normalize to dict[str, Path]
        # Accepts: Path, str (path), or dict[str, Path | str (path) | dict[str, str] (inline content)]
        if isinstance(self.skills, dict):
            from vero.core.sessions import get_session_dir

            resolved: dict[str, Path] = {}
            for ns, value in self.skills.items():
                if isinstance(value, dict):
                    # Inline content: write to session dir as files
                    ns_dir = (
                        get_session_dir(self.sessions_dir, self.session_id)
                        / "skills"
                        / ns
                    )
                    ns_dir.mkdir(parents=True, exist_ok=True)
                    for name, content in value.items():
                        filename = name if "." in name else f"{name}.md"
                        (ns_dir / filename).write_text(content)
                    resolved[ns] = ns_dir
                else:
                    resolved[ns] = Path(value)
            self.session.skills = resolved
        elif self.skills is not None:
            self.session.skills = {"skills": Path(self.skills)}

        # Split budget
        self._build_split_budget()
        self._validate_budget_splits()
        self.session.budget = self.budget

        # Canonical engine with the existing Python task runner isolated behind
        # one approved VeroTask backend.
        await self._init_vero_task_evaluation()

        # Filesystem
        if self.filesystem_accesses is None:
            from vero.core.veroaccess import resolve_filesystem_accesses

            self.filesystem_accesses = resolve_filesystem_accesses(
                Path(self.session.workspace.project_path)
            )

        # Remaining session fields from config
        self.session.split_accesses = self.split_accesses

        # Materialize artifacts to _vero/ directory
        if self.artifacts:
            await self._setup_vero_dir()
            self.filesystem_accesses.append(
                AccessRule(access_type=AccessType.READ, pattern="_vero/**")
            )
            vero_dir = f"{self.session.workspace.project_path}/_vero"
            sandbox = self.session.workspace.sandbox
            for artifact in self.artifacts:
                await artifact.on_init(self, vero_dir, sandbox)

        self.session.workspace.set_access(
            self.filesystem_accesses, self.filesystem_default_access
        )
        self.session.evaluation_parameters = self.evaluation_parameters
        self.session.task = self.task

        # Instructions
        self._render_instructions()

        # Session logger (events, general logs, console)
        from vero.core.sessions import get_session_dir

        self.session_logger = SessionLogger(
            session_dir=get_session_dir(self.sessions_dir, self.session_id),
            enable_event_log=self.enable_event_log,
            enable_general_log=self.enable_general_log,
            enable_console=self.enable_console,
            console_verbose=self.console_verbose,
            console_title=type(self.agent).__name__,
        )
        self.on_event.append(self.session_logger)

        # Initialize agent
        self.agent.init(self.session)

    async def _init_program_policy(self) -> None:
        """Initialize the canonical dataset-free Policy path."""
        from vero.core.sessions import (
            get_session_db_path,
            get_session_experiments_dir,
        )
        from vero.evaluation import (
            BackendRegistry,
            EvaluationDatabase,
            EvaluationEngine,
            EvaluationLimits,
            Evaluator as ProgramEvaluator,
        )
        from vero.optimization import AgentCandidateProducer, ProgramPolicy

        if not self.backends:
            raise ValueError("generic program Policy requires at least one backend")
        if self.evaluation_set is None:
            raise ValueError("generic program Policy requires evaluation_set")
        if self.objective is None:
            raise ValueError("generic program Policy requires objective")
        backend_id = self.default_backend_id
        if backend_id is None:
            if len(self.backends) != 1:
                raise ValueError(
                    "default_backend_id is required when multiple backends are registered"
                )
            backend_id = next(iter(self.backends))

        if self.workspace is None:
            if self.sandbox is None:
                from vero.sandbox import LocalSandbox

                self.sandbox = await LocalSandbox.create()
            self.workspace = await GitWorkspace.from_path(
                self.sandbox,
                str(self.project_path),
            )
        workspace = self.workspace
        if await workspace.is_dirty():
            raise ValueError("generic program Policy requires a clean workspace")
        if isinstance(workspace, GitWorkspace):
            base_version = await workspace.resolve_ref(self.ref)
            base_branch = await workspace.current_branch()
        else:
            base_version = await workspace.current_version()
            base_branch = None

        self.session.workspace = workspace
        self.session.project_path = Path(workspace.project_path)
        self.session.base_version = base_version
        self.session.base_branch = base_branch

        database_path = get_session_db_path(self.sessions_dir, self.session_id)
        database = (
            EvaluationDatabase.load_from_file(database_path)
            if database_path.exists()
            else EvaluationDatabase.from_evaluations_dir(
                get_session_experiments_dir(self.sessions_dir, self.session_id),
                db_id=self.session_id,
            )
        )
        evaluator = ProgramEvaluator(
            workspace=workspace,
            sessions_dir=self.sessions_dir,
            session_id=self.session_id,
            use_copy=self.use_copy if isinstance(self.use_copy, bool) else True,
        )
        engine = EvaluationEngine(
            evaluator=evaluator,
            backends=BackendRegistry(self.backends),
            database=database,
            database_path=database_path,
            budget_ledger=self.program_budget_ledger,
        )

        if self.filesystem_accesses is None:
            from vero.core.veroaccess import resolve_filesystem_accesses

            self.filesystem_accesses = resolve_filesystem_accesses(
                Path(workspace.project_path)
            )
        workspace.set_access(
            self.filesystem_accesses,
            self.filesystem_default_access,
        )
        self._render_instructions()

        producer = self.optimizer
        self.evaluation_db = database
        self.evaluation_engine = engine
        self.program_policy = ProgramPolicy(
            workspace=workspace,
            engine=engine,
            backend_id=backend_id,
            evaluation_set=self.evaluation_set,
            objective=self.objective,
            optimizer=producer,
            parameters=self.program_parameters,
            limits=self.program_limits or EvaluationLimits(),
            seed=self.program_seed,
            max_candidates=self.max_candidates,
            base_version=base_version,
        )
        self.session.program_policy = self.program_policy
        self.session.evaluation_engine = engine
        self.session.evaluation_database = database
        self.session.evaluation_backend_id = backend_id
        self.session.evaluation_objective = self.objective
        if producer is None and self.agent is not None:
            tool_sets = getattr(self.agent, "tool_sets", None)
            if isinstance(tool_sets, list):
                from vero.tools import EvaluationRunnerTool, EvaluationViewer

                if not any(isinstance(tool, EvaluationRunnerTool) for tool in tool_sets):
                    tool_sets.append(EvaluationRunnerTool())
                if not any(isinstance(tool, EvaluationViewer) for tool in tool_sets):
                    tool_sets.append(EvaluationViewer())
            self.agent.init(self.session)
            self.program_policy.optimizer = AgentCandidateProducer(
                self.agent,
                prompt=self.prompt,
                max_turns=self.max_turns,
                on_event=self._fire_event,
            )
        self.session.evaluator = evaluator  # type: ignore[assignment]
        self.session.db = database  # type: ignore[assignment]

    async def _init_vero_task_evaluation(self) -> None:
        """Route the dataset-oriented API through canonical evaluation contracts."""
        from vero.core.sessions import (
            get_session_db_path,
            get_session_experiments_dir,
            get_session_dir,
        )
        from vero.evaluation import (
            BackendRegistry,
            BudgetLedger,
            EvaluationBudget,
            EvaluationDatabase,
            EvaluationEngine,
            Evaluator as ProgramEvaluator,
            VeroTaskBackend,
            VeroTaskBackendConfig,
            VeroTaskEvaluatorAdapter,
            compatibility_objective,
            evaluation_database_to_experiment_database,
            evaluation_record_to_experiment,
        )

        assert self.session is not None
        assert self.session.workspace is not None

        database_path = get_session_db_path(self.sessions_dir, self.session_id)
        database = (
            EvaluationDatabase.load_from_file(database_path)
            if database_path.exists()
            else EvaluationDatabase(id=self.session_id)
        )
        reconstructed = EvaluationDatabase.from_evaluations_dir(
            get_session_experiments_dir(self.sessions_dir, self.session_id),
            db_id=self.session_id,
        )
        for record in reconstructed.evaluations.values():
            database.add_evaluation(record)

        environment_names = []
        if isinstance(self.subprocess_env_vars, list):
            environment_names = [
                item if isinstance(item, str) else item[0]
                for item in self.subprocess_env_vars
            ]
        elif self.subprocess_env_vars is not None:
            environment_names = [str(self.subprocess_env_vars)]

        backend_id = "vero-task"
        backend = VeroTaskBackend(
            VeroTaskBackendConfig(
                session_id=self.session_id,
                vero_home=str(self._vero_home),
                task=self.task,
                task_project=str(Path(self.task_project).resolve())
                if self.task_project
                else None,
                task_module=self.task_module,
                subprocess_env_vars=environment_names,
                error_rate_threshold=self.evaluation_parameters.error_rate_threshold,
                use_threading=self.evaluation_parameters.use_threading,
            ),
            subprocess_env_vars=self.subprocess_env_vars,
        )

        budget_path = get_session_dir(
            self.sessions_dir,
            self.session_id,
        ) / "evaluation_budget.json"
        if budget_path.exists():
            ledger = BudgetLedger.load(budget_path)
        else:
            canonical_budgets = []
            for split_budget in self.budget or []:
                evaluation_set = self._dataset_evaluation_set(
                    dataset_id=split_budget.dataset_id,
                    split=split_budget.split,
                )
                canonical_budgets.append(
                    EvaluationBudget(
                        backend_id=backend_id,
                        evaluation_set_key=evaluation_set.budget_key(backend_id),
                        total_runs=split_budget.total_run_budget,
                        remaining_runs=split_budget.remaining_run_budget,
                        total_cases=split_budget.total_sample_budget,
                        remaining_cases=split_budget.remaining_sample_budget,
                        max_cases_per_run=split_budget.max_samples_per_run,
                    )
                )
            ledger = BudgetLedger(canonical_budgets, path=budget_path)
            ledger.save()

        evaluator = ProgramEvaluator(
            workspace=self.session.workspace,
            sessions_dir=self.sessions_dir,
            session_id=self.session_id,
            use_copy=self.use_copy if isinstance(self.use_copy, bool) else False,
        )
        engine = EvaluationEngine(
            evaluator=evaluator,
            backends=BackendRegistry({backend_id: backend}),
            database=database,
            database_path=database_path,
            budget_ledger=ledger,
        )
        objective = self.objective or compatibility_objective()
        self.objective = objective
        compatibility_db = evaluation_database_to_experiment_database(database)

        async def _on_evaluation(record):
            try:
                experiment = evaluation_record_to_experiment(record)
            except ValueError:
                return
            compatibility_db.add_experiment(experiment)
            if self.artifacts:
                vero_dir = f"{self.session.workspace.project_path}/_vero"
                sandbox = self.session.workspace.sandbox
                for artifact in self.artifacts:
                    await artifact.on_experiment(
                        self,
                        experiment,
                        vero_dir,
                        sandbox,
                    )

        engine.listeners.append(_on_evaluation)
        adapter = VeroTaskEvaluatorAdapter(
            engine=engine,
            workspace=self.session.workspace,
            backend_id=backend_id,
            objective=objective,
            task=self.task,
            compatibility_db=compatibility_db,
        )

        self.evaluation_db = database
        self.evaluation_engine = engine
        self.default_backend_id = backend_id
        self.session.evaluator = adapter  # type: ignore[assignment]
        self.session.db = compatibility_db
        self.session.evaluation_engine = engine
        self.session.evaluation_database = database
        self.session.evaluation_backend_id = backend_id
        self.session.evaluation_objective = objective

    @staticmethod
    def _dataset_evaluation_set(
        *,
        dataset_id: str,
        split: str,
        sample_ids: list[int] | None = None,
    ):
        from vero.evaluation import AllCases, CaseIds, EvaluationSet

        selection = (
            CaseIds(ids=[str(sample_id) for sample_id in sample_ids])
            if sample_ids is not None
            else AllCases()
        )
        return EvaluationSet(
            name=dataset_id,
            partition=split,
            selection=selection,
        )

    async def _setup_vero_dir(self) -> None:
        """Set up _vero/ directory in the workspace and exclude from version control."""

        self.initialized(validate=True)

        sandbox = self.session.workspace.sandbox
        workspace = self.session.workspace
        vero_dir = f"{workspace.project_path}/_vero"
        await sandbox.mkdir(vero_dir)

        # Add _vero/ to git exclude so it doesn't show up in git status
        from vero.workspace.git import GitWorkspace

        if isinstance(workspace, GitWorkspace):
            try:
                result = await sandbox.run(
                    ["git", "rev-parse", "--absolute-git-dir"],
                    cwd=workspace.project_path,
                )
                if result.returncode == 0:
                    git_dir = result.stdout.strip()
                    exclude_file = f"{git_dir}/info/exclude"
                    await sandbox.mkdir(f"{git_dir}/info")
                    existing = ""
                    if await sandbox.exists(exclude_file):
                        existing = await sandbox.read_file(exclude_file)
                    if "_vero/" not in existing:
                        await sandbox.write_file(exclude_file, existing + "\n_vero/\n")
            except Exception:
                pass  # Non-critical — git exclude is just a convenience

    def _build_split_budget(self) -> None:
        """Build split budget list from budget fields.

        If `budget` is provided directly, use it (filling in dataset_id where missing).
        Otherwise, build from the convenience fields (train_budget, etc.) — these
        apply to ALL registered datasets.
        """
        from vero.core.dataset import DefaultSplitNames
        from vero.core.dataset.store import list_datasets
        from vero.tools import SplitBudget

        if self.budget is not None:
            # Fill in dataset_id for entries that don't specify one
            dataset_ids = list_datasets(self.sessions_dir, self.session_id)
            for b in self.budget:
                if not b.dataset_id:
                    if len(dataset_ids) == 1:
                        b.dataset_id = dataset_ids[0]
                    else:
                        raise ValueError(
                            f"SplitBudget for split '{b.split}' has no dataset_id, "
                            f"but there are {len(dataset_ids)} datasets. "
                            f"Please specify dataset_id explicitly."
                        )
            return

        # Convenience fields — create budgets for each registered dataset
        dataset_ids = list_datasets(self.sessions_dir, self.session_id)
        budgets = []
        for ds_id in dataset_ids:
            if self.train_budget or self.train_sample_budget:
                budgets.append(
                    SplitBudget(
                        split=DefaultSplitNames.train,
                        dataset_id=ds_id,
                        total_run_budget=self.train_budget,
                        total_sample_budget=self.train_sample_budget,
                    )
                )
            if self.validation_budget or self.validation_sample_budget:
                budgets.append(
                    SplitBudget(
                        split=DefaultSplitNames.validation,
                        dataset_id=ds_id,
                        total_run_budget=self.validation_budget,
                        total_sample_budget=self.validation_sample_budget,
                    )
                )
        self.budget = budgets

    def _validate_budget_splits(self) -> None:
        """Warn if any budget splits don't exist in the dataset."""
        if not self.budget:
            return
        from vero.core.dataset.store import load_dataset

        for b in self.budget:
            ds_id = b.dataset_id or self.session.dataset_id
            try:
                ds = load_dataset(
                    self.sessions_dir, self.dataset_cache, self.session_id, ds_id
                )
                if b.split not in ds:
                    available = list(ds.keys())
                    logger.warning(
                        f"Budget specifies split '{b.split}' for dataset '{ds_id}', "
                        f"but dataset only has splits: {available}. "
                        f"This budget will be unused."
                    )
            except Exception:
                pass

    def _maybe_make_db(self) -> ExperimentDatabase:
        """Create or reconstruct the experiment database.

        If experiments/ exists on disk, reconstruct the DB from it.
        Otherwise create a new empty database.
        """
        self.initialized(validate=True)

        if self.session.db is None:  # type: ignore
            from vero.core.sessions import get_session_experiments_dir

            experiments_dir = get_session_experiments_dir(
                self.sessions_dir, self.session_id
            )
            if experiments_dir.exists() and any(experiments_dir.iterdir()):
                self.session.db = ExperimentDatabase.from_experiments_dir(  # type: ignore
                    experiments_dir, db_id=self.session_id
                )
                logger.info(
                    f"Reconstructed DB from {len(self.session.db.results)} experiments on disk"  # type: ignore
                )
            else:
                self.session.db = ExperimentDatabase(id=self.session_id)  # type: ignore
                logger.debug(f"Created new database with id: {self.session.db.id}")  # type: ignore
        return self.session.db  # type: ignore

    @classmethod
    def resume(
        cls, session_id: str, restore_agent_state: bool = True, **policy_kwargs: Any
    ) -> Policy:
        """Resume a Policy from an existing session.

        Points at the existing session directory, project, and experiments.
        On init(), the DB will be reconstructed from experiments on disk.
        If ``restore_agent_state`` is True and a saved agent state exists,
        it will be deserialized into the agent after init.

        Args:
            session_id: Session ID to resume.
            restore_agent_state: Whether to restore saved agent state (default: True).
            **policy_kwargs: Arguments forwarded to the Policy constructor.
                Must include at least ``agent`` and ``dataset``.

        Returns:
            A new Policy (not yet initialized — use ``async with`` or call ``init()``).
            After init(), agent state will be restored if available.
        """
        from vero.core.sessions import (
            find_project_dir_in_session,
            get_session_state_path,
            get_vero_home_dir,
        )

        vero_home = Path(policy_kwargs.get("vero_home") or get_vero_home_dir())
        sessions_dir = vero_home / "sessions"

        # Find the project directory in the session
        project_dir = find_project_dir_in_session(sessions_dir, session_id)
        if project_dir and project_dir.exists():
            policy_kwargs["project_path"] = project_dir
            policy_kwargs.setdefault("isolate", False)
        elif "project_path" not in policy_kwargs:
            raise ValueError(
                f"No project directory found in session {session_id} "
                "and no project_path provided in kwargs."
            )

        policy_kwargs["session_id"] = session_id
        policy_kwargs.setdefault("vero_home", vero_home)
        policy = cls(**policy_kwargs)

        # Restore agent state after construction (will be applied after init)
        if restore_agent_state:
            state_path = get_session_state_path(sessions_dir, session_id)
            if state_path.exists():
                with open(state_path) as f:
                    saved_state = json.load(f)
                policy.agent.deserialize_state(saved_state)
                logger.info(f"Restored agent state from {state_path}")

        return policy

    @classmethod
    def fork(cls, source_session_id: str, **policy_kwargs: Any) -> Policy:
        """Create a new Policy seeded from an existing session's experiments and project.

        Copies the isolated project repo (with all git branches/commits) and the
        experiments directory into a fresh session, then resumes from the copy.

        Args:
            source_session_id: Session ID to fork from.
            **policy_kwargs: Arguments forwarded to the Policy constructor.
                Must include at least ``agent`` and ``dataset``.

        Returns:
            A new Policy (not yet initialized — use ``async with`` or call ``init()``).
        """
        import shutil

        from vero.core.sessions import (
            create_session_dir,
            find_project_dir_in_session,
            get_session_experiments_dir,
            get_vero_home_dir,
        )

        vero_home = Path(policy_kwargs.get("vero_home") or get_vero_home_dir())
        sessions_dir = vero_home / "sessions"

        # Find source project and experiments
        source_project = find_project_dir_in_session(sessions_dir, source_session_id)
        source_experiments = get_session_experiments_dir(
            sessions_dir, source_session_id
        )

        # Create new session
        new_session_id = str(uuid4())
        new_session_dir = create_session_dir(sessions_dir, new_session_id)

        # Copy project repo (preserves .git/, all branches and commits)
        if source_project and source_project.exists():
            dest_project = new_session_dir / source_project.name
            shutil.copytree(source_project, dest_project)
            logger.info(f"Forked project: {source_project} -> {dest_project}")

        # Copy experiments
        if source_experiments.exists():
            dest_experiments = new_session_dir / "experiments"
            shutil.copytree(source_experiments, dest_experiments)
            logger.info(
                f"Forked {len(list(dest_experiments.iterdir()))} experiments "
                f"from session {source_session_id}"
            )

        # Resume from the new session
        return cls.resume(new_session_id, **policy_kwargs)

    def _load_template(self, template: str | Template) -> Template:
        """Load a Jinja2 template from a file path or vero's built-in templates."""
        from jinja2 import Template

        from vero.jinja import get_stored_jinja_template

        if isinstance(template, Template):
            return template
        return get_stored_jinja_template(template)

    def _render_template(self, template: Template) -> str:
        """Render a Jinja2 template with the standard context."""
        return template.render(
            policy=self,
            ToolRegistry=ToolRegistry,
            **self.prompt_kwargs,
        )

    def _render_instructions(self) -> None:
        """Load and render instructions template into session."""
        if self.instructions_template is not None:
            self.session.instructions = self._render_template(  # type: ignore
                self._load_template(self.instructions_template)
            )

    @property
    def prompt(self) -> str | None:
        """Render prompt from template on demand."""
        if self.prompt_template is None:
            return None
        return self._render_template(self._load_template(self.prompt_template))

    # -------------------------------------------------------------------------
    # Async context manager
    # -------------------------------------------------------------------------

    _background_tasks: list = field(default_factory=list, repr=False)

    async def __aenter__(self) -> Policy:
        await self.init()

        if self.enable_wandb:
            import wandb

            self.initialized(validate=True)

            wandb_name = (
                await self.session.workspace.current_version()
                or self.session.base_branch  # type: ignore
            )
            assert self.wandb_project, "wandb project is not set!"
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=wandb_name,
                job_type="optimization-loop",
            )
            # Store wandb run ID in metadata for later reference
            self.metadata["wandb_run_id"] = self.wandb_run.id
            self.metadata["wandb_run_path"] = self.wandb_run.path

            # Log metadata to wandb
            if self.metadata:
                self.wandb_run.config.update(self.metadata)

            # Register the appropriate durable-record listener.
            if self.is_program_policy and self.evaluation_engine is not None:
                self.evaluation_engine.listeners.append(
                    lambda record: (
                        log_evaluations_to_wandb(self.wandb_run, [record])
                        if self.wandb_run is not None
                        and not self.wandb_run._is_finished
                        else None
                    )
                )
            elif self.session.db:  # type: ignore
                self.session.db.listeners.append(  # type: ignore
                    lambda experiment: (
                        log_experiments_to_wandb(self.wandb_run, [experiment])
                        if self.wandb_run is not None
                        and not self.wandb_run._is_finished
                        else None
                    )
                )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()
        self.finish()

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    async def run(
        self, skip_initial_eval: bool = False, eval_split: str = "test"
    ) -> BestVersion:
        """Run the full optimization loop: init, evaluate, step, get best, final eval, finish.

        Args:
            skip_initial_eval: Skip the initial evaluation.
            eval_split: Split to use for initial and final evaluation.

        Returns:
            BestVersion with the best commit found.
        """
        if self.is_program_policy:
            async with self:
                assert self.program_policy is not None
                self.program_run = await self.program_policy.run(
                    skip_initial_evaluation=skip_initial_eval
                )
                best_record = self.program_run.best
                return BestVersion(
                    commit=best_record.request.candidate.commit
                    if best_record is not None
                    else None,
                    split=self.evaluation_set.name
                    if self.evaluation_set is not None
                    else None,
                    score=best_record.objective.value
                    if best_record is not None and best_record.objective is not None
                    else None,
                    objective_metric=self.objective.selector.metric
                    if self.objective is not None
                    else None,
                    evaluation_id=best_record.id if best_record is not None else None,
                )

        async with self:
            if not skip_initial_eval:
                logger.info(f"Running initial evaluation on {eval_split} split")
                initial = await self.evaluate_version(
                    self.session.base_version,  # type: ignore
                    split=eval_split,
                )
                logger.info(f"Initial score: {initial.result.score()}")

            await self.step()

            best = self.get_best_non_baseline_version(
                splits=[eval_split] if eval_split else None
            )
            if best.commit:
                logger.info(
                    f"Best non-baseline commit: {best.commit} (split: {best.split}, score: {best.score})"
                )
                await self.evaluate_version(
                    best.commit, split=eval_split, log_to_wandb=True
                )
            else:
                logger.info("No non-baseline commit found — agent made no changes")

            overall_best = self.get_best_version()
            if overall_best.commit:
                logger.info(
                    f"Overall best commit: {overall_best.commit} (score: {overall_best.score})"
                )

        return best

    def _fire_event(self, raw_event: Any) -> None:
        """Serialize a raw SDK event and dispatch to all on_event callbacks."""
        if self.agent is None:
            return
        serialized = self.agent.serialize_event(raw_event)
        if serialized is None:
            return
        for callback in self.on_event:
            try:
                callback(serialized)
            except Exception as e:
                logger.warning(f"on_event callback failed: {e}")

    async def step(self, input: str | None = None, max_turns: int | None = None) -> Any:
        """Execute optimization steps via the agent backend."""
        if self.agent is None:
            raise ValueError("step() requires an agent-backed Policy")
        if max_turns is None:
            max_turns = self.max_turns
        if input is None:
            input = self.prompt
        return await self.agent.step(input, max_turns, on_event=self._fire_event)

    async def evaluate_version(
        self,
        commit: str,
        dataset_id: str | None = None,
        split: str = "test",
        sample_ids: list[int] | None = None,
        add_to_db: bool = True,
        log_to_wandb: bool = False,
        use_copy: bool | None = None,
    ) -> Experiment:
        """Evaluate a commit on a dataset split."""

        self.initialized(validate=True)

        if self.is_program_policy:
            assert self.program_policy is not None
            return await self.program_policy.evaluate_version(commit)  # type: ignore[return-value]

        if dataset_id is None:
            dataset_id = self.session.dataset_id  # type: ignore

        assert dataset_id, "Dataset ID not provided and not set!"
        assert self.session.evaluator  # type: ignore

        experiment = await self.session.evaluator.evaluate(  # type: ignore
            commit=commit,
            dataset_id=dataset_id,
            split=split,
            task=self.task,
            sample_ids=sample_ids,
            db=self.session.db if add_to_db else None,  # type: ignore
            evaluation_parameters=self.evaluation_parameters,
            use_copy=use_copy,
            add_to_compatibility_db=add_to_db,
        )

        # Log to wandb (Policy-specific, not in evaluator)
        if (
            log_to_wandb
            and self.wandb_run is not None
            and not self.wandb_run._is_finished
        ):
            log_experiments_to_wandb(self.wandb_run, [experiment])

        return experiment

    async def evaluate_candidate(
        self,
        commit: str,
        *,
        backend_id: str | None = None,
        evaluation_set: EvaluationSet | None = None,
        parameters: dict[str, Any] | None = None,
        limits: EvaluationLimits | None = None,
        objective: ObjectiveSpec | None = None,
    ) -> EvaluationRecord:
        """Evaluate a versioned program through an approved canonical backend."""
        from datetime import datetime

        from vero.core.db.candidate import Candidate
        from vero.evaluation import EvaluationLimits, EvaluationRequest

        self.initialized(validate=True)
        if self.evaluation_engine is None or self.evaluation_db is None:
            raise ValueError("Policy does not have a canonical evaluation engine")
        assert self.session is not None
        assert self.session.workspace is not None

        workspace = self.session.workspace
        if isinstance(workspace, GitWorkspace):
            commit = await workspace.resolve_ref(commit)
        selected_backend = backend_id or self.default_backend_id
        if selected_backend is None:
            raise ValueError("backend_id is required")
        selected_set = evaluation_set or self.evaluation_set
        if selected_set is None:
            raise ValueError("evaluation_set is required")
        selected_objective = objective or self.objective
        if selected_objective is None:
            raise ValueError("objective is required")

        candidate = self.evaluation_db.candidates.get((workspace.name, commit))
        if candidate is None:
            candidate = Candidate(
                commit=commit,
                repo_name=workspace.name,
                created_at=datetime.now(),
            )
        selected_parameters = (
            parameters
            if parameters is not None
            else self.program_parameters
            if self.is_program_policy
            else self.evaluation_parameters.task_params
        )
        selected_limits = limits
        if selected_limits is None:
            if self.is_program_policy:
                selected_limits = self.program_limits or EvaluationLimits()
            else:
                selected_limits = EvaluationLimits(
                    timeout_seconds=self.evaluation_parameters.timeout,
                    case_timeout_seconds=self.evaluation_parameters.sample_timeout,
                    max_concurrency=self.evaluation_parameters.max_concurrency,
                    retry_config=self.evaluation_parameters.retry_config,
                )
        request = EvaluationRequest(
            candidate=candidate,
            evaluation_set=selected_set,
            parameters=selected_parameters,
            limits=selected_limits,
            seed=self.program_seed if self.is_program_policy else None,
        )
        return await self.evaluation_engine.evaluate_record(
            backend_id=selected_backend,
            request=request,
            objective_spec=selected_objective,
        )

    def _get_best_from_db(
        self, splits: list[str], exclude_base: bool = False
    ) -> BestVersion:
        """Query DB for best experiment, optionally excluding base commit."""
        if self.evaluation_db is None or self.objective is None:
            return BestVersion()

        from vero.evaluation import select_best_evaluation

        for split in splits:
            records = [
                record
                for record in self.evaluation_db.evaluations.values()
                if record.backend_id == self.default_backend_id
                and record.objective_spec == self.objective
                and record.request.evaluation_set.partition == split
                and (
                    not exclude_base
                    or record.request.candidate.commit != self.session.base_version  # type: ignore[union-attr]
                )
            ]
            best = select_best_evaluation(records)
            if best is not None:
                logger.info(
                    f"Best {split} evaluation: {best.request.candidate.commit} "
                    f"(objective: {best.objective.value}, exclude_base={exclude_base})"
                )
                return BestVersion(
                    commit=best.request.candidate.commit,
                    split=split,
                    score=best.objective.value,
                    objective_metric=self.objective.selector.metric,
                    evaluation_id=best.id,
                )

        logger.debug(f"No experiments found in splits: {splits}")
        return BestVersion()

    def get_best_version(self, splits: list[str] | None = None) -> BestVersion:
        """Find best experiment overall (including baseline).

        First checks if the agent extracted a best commit (e.g. from structured output).
        Falls back to querying the experiment DB.
        """
        if self.is_program_policy:
            if self.evaluation_db is None or self.objective is None:
                return BestVersion()
            assert self.program_policy is not None
            record = self.program_policy.best_evaluation()
            return BestVersion(
                commit=record.request.candidate.commit if record else None,
                split=record.request.evaluation_set.name if record else None,
                score=record.objective.value
                if record and record.objective is not None
                else None,
                objective_metric=self.objective.selector.metric,
                evaluation_id=record.id if record else None,
            )
        assert self.agent is not None
        agent_best = self.agent.get_best_version()
        if agent_best.commit is not None:
            logger.info(
                f"Best commit from agent: {agent_best.commit} (score: {agent_best.score})"
            )
            return agent_best
        return self._get_best_from_db(
            splits or ["validation", "train"], exclude_base=False
        )

    def get_best_non_baseline_version(
        self, splits: list[str] | None = None
    ) -> BestVersion:
        """Find best experiment excluding the baseline commit — the agent's best effort."""
        if self.is_program_policy:
            if (
                self.evaluation_db is None
                or self.objective is None
                or self.session is None
                or self.session.workspace is None
            ):
                return BestVersion()
            assert self.program_policy is not None
            record = self.program_policy.best_evaluation(
                exclude_candidate=(
                    self.session.workspace.name,
                    self.session.base_version,
                ),
            )
            return BestVersion(
                commit=record.request.candidate.commit if record else None,
                split=record.request.evaluation_set.name if record else None,
                score=record.objective.value
                if record and record.objective is not None
                else None,
                objective_metric=self.objective.selector.metric,
                evaluation_id=record.id if record else None,
            )
        assert self.agent is not None
        agent_best = self.agent.get_best_version()
        if agent_best.commit is not None:
            return agent_best
        return self._get_best_from_db(
            splits or ["validation", "train"], exclude_base=True
        )

    # -------------------------------------------------------------------------
    # Serialization & artifacts
    # -------------------------------------------------------------------------

    def as_dict(self) -> dict:
        """Return a dictionary representation of the policy."""

        assert self.session

        if self.is_program_policy:
            return {
                "schema_version": 2,
                "session_id": self.session_id,
                "base_version": self.session.base_version,
                "current_commit": self.get_best_version().commit,
                "backend_id": self.program_policy.backend_id
                if self.program_policy is not None
                else None,
                "evaluation_set": self.evaluation_set.model_dump(mode="json")
                if self.evaluation_set is not None
                else None,
                "objective": self.objective.model_dump(mode="json")
                if self.objective is not None
                else None,
                "metadata": self.metadata,
            }

        base = {
            "session_id": self.session_id,
            "base_version": self.session.base_version,
            "base_branch": self.session.base_branch,
            "current_commit": self.session.base_version,
            "instructions": self.session.instructions if self.session else None,
            "split_accesses": [
                {"split": str(sa.split), "access": str(sa.access)}
                for sa in self.split_accesses
            ],
            "budget": [recursively_serialize(budget) for budget in self.budget]  # type: ignore
            if self.budget
            else [],
            "metadata": self.metadata,
        }
        base.update(self.agent.dict())
        return base

    def save_session_artifacts(self) -> None:
        """Save session artifacts (database, config, run result) to the session directory."""
        from vero.core.sessions import (
            get_session_config_path,
            get_session_db_path,
            get_session_dir,
            get_session_result_path,
        )

        if self.session_id is None:
            logger.warning("No session ID set, skipping session artifacts save.")
            return

        assert self.session

        if self.is_program_policy:
            config_path = get_session_config_path(self.sessions_dir, self.session_id)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(self.as_dict(), indent=2, default=str))
            return

        # Canonical records are the durable source of truth. The Experiment DB,
        # when present, is only a deprecated in-memory view.
        if self.evaluation_db is not None:
            db_path = get_session_db_path(self.sessions_dir, self.session_id)
            self.evaluation_db.save_to_file(db_path)
            logger.debug(f"Saved database to: {db_path}")
        elif self.session.db is not None:
            db_path = get_session_db_path(self.sessions_dir, self.session_id)
            self.session.db.save_to_file(db_path)
            logger.debug(f"Saved database to: {db_path}")

        # Save config dump
        config_path = get_session_config_path(self.sessions_dir, self.session_id)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.as_dict(), f, indent=2, default=str)
        logger.debug(f"Saved config to: {config_path}")

        # Save run result dump (trace)
        try:
            run_result = self.agent.serialize_trace()
            if run_result is not None:
                result_path = get_session_result_path(
                    self.sessions_dir, self.session_id
                )
                with open(result_path, "w") as f:
                    json.dump(run_result, f, indent=2, default=str)
                logger.debug(f"Saved run result to: {result_path}")
        except Exception as e:
            logger.warning(f"Failed to save run result: {e}")

        # Save agent state (for resumption)
        try:
            from vero.core.sessions import get_session_state_path

            agent_state = self.agent.serialize_state()
            if agent_state is not None:
                state_path = get_session_state_path(self.sessions_dir, self.session_id)
                with open(state_path, "w") as f:
                    json.dump(agent_state, f, indent=2, default=str)
                logger.debug(f"Saved agent state to: {state_path}")
        except Exception as e:
            logger.warning(f"Failed to save agent state: {e}")

        # Save agent-specific artifacts
        session_dir = get_session_dir(self.sessions_dir, self.session_id)
        self.agent.save_artifacts(session_dir)

    def finish(self) -> None:
        """Finish the session and log a summary to wandb if enabled."""
        self.save_session_artifacts()

        if self.is_program_policy:
            if self.wandb_run is not None and not self.wandb_run._is_finished:
                best = (
                    self.program_policy.best_evaluation()
                    if self.program_policy is not None
                    else None
                )
                self.wandb_run.summary.update(
                    {
                        "config": self.as_dict(),
                        "best_evaluation_id": best.id if best else None,
                        "best_commit": best.request.candidate.commit if best else None,
                        "best_objective": best.objective.value
                        if best is not None and best.objective is not None
                        else None,
                        "best_feasible": best.objective.feasible
                        if best is not None and best.objective is not None
                        else None,
                        "usage": self.agent.usage() if self.agent is not None else {},
                    }
                )
                self.wandb_run.finish()
            return

        if self.session_logger is not None:
            self.session_logger.close()

        if self.wandb_run is None or self.wandb_run._is_finished:
            return

        # Get best results per split from experiments DB
        results = {}

        if self.evaluation_db is not None and self.objective is not None:
            from vero.evaluation import select_best_evaluation

            splits = {
                record.request.evaluation_set.partition
                for record in self.evaluation_db.evaluations.values()
                if record.request.evaluation_set.partition is not None
            }
            for split in sorted(splits):
                best = select_best_evaluation(
                    record
                    for record in self.evaluation_db.evaluations.values()
                    if record.backend_id == self.default_backend_id
                    and record.objective_spec == self.objective
                    and record.request.evaluation_set.partition == split
                )
                if best is None:
                    continue
                results[split] = {
                    "commit": best.request.candidate.commit,
                    "score": best.objective.value,
                    "objective_metric": self.objective.selector.metric,
                    "feasible": best.objective.feasible,
                    "status": best.report.status.value,
                    "error_rate": best.report.metrics.get("error_rate"),
                    "num_samples": best.report.metrics.get("num_results"),
                    "evaluation_id": best.id,
                }

        summary = {
            "config": self.as_dict(),
            "best_results": results,
            "usage": self.agent.usage(),
        }
        summary.update(self.agent.summary())

        self.wandb_run.summary.update(summary)
        self.wandb_run.finish()
