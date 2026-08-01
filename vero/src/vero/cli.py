"""Command-line interface for generic program optimization."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import click

from vero.candidate_repository import GitCandidateRepository
from vero.config import load_config
from vero.evaluation import (
    AllCases,
    BudgetLedger,
    CaseIds,
    CaseRange,
    CommandBackend,
    CommandBackendConfig,
    ConstraintOperator,
    DisclosureLevel,
    EvaluationDatabase,
    EvaluationLimits,
    EvaluationPlan,
    EvaluationSet,
    MetricAggregation,
    MetricConstraint,
    MetricSelector,
    ObjectiveSpec,
    RetryPolicy,
    project_evaluation,
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
    SequentialStrategy,
)
from vero.runtime import (
    SessionManifest,
    SessionStatus,
    WandbEventSink,
    create_local_optimization_session,
)


def _default_home() -> Path:
    return Path(os.environ.get("VERO_HOME", "~/.vero")).expanduser().resolve()


_CONFIG_TEMPLATE = '''[target]
root = "./target"
ref = "HEAD"

[backend]
id = "command"
kind = "command"
harness_root = "./harness"
command = ["python3", "evaluate.py", "{workspace}", "{request}", "{report}"]

[[evaluations]]
name = "train"
partition = "train"
agent_can_evaluate = true
agent_visible = true
agent_selection = "arbitrary"
disclosure = "full"
expose_case_resources = true

[[evaluations]]
name = "validation"
partition = "validation"
agent_can_evaluate = true
agent_visible = true
agent_selection = "arbitrary"
disclosure = "aggregate"

[evaluations.agent_budget]
total_runs = 50

[[evaluations]]
name = "test"
partition = "test"
agent_can_evaluate = false
agent_visible = false
agent_selection = "fixed"
disclosure = "none"

[protocol]
selection_evaluation = "validation"
final_evaluation = "test"
max_proposals = 5

[objective]
metric = "score"
direction = "maximize"

[optimizer]
kind = "claude"
instruction = "Improve the program without changing its intended behavior"

# [session]
# Uncommented, `id` turns every later `vero run` over this file into a relaunch
# of one logical run: the session directory becomes $VERO_HOME/sessions/<id>
# instead of a fresh uuid4 per invocation, and the candidates, scores and budget
# already on disk are picked up rather than remade. Commented out because the
# resume is not free: a relaunch skips the baseline evaluation whenever a
# manifest exists, so if the first attempt died *during* its baseline it hands
# the rerun that attempt's unusable baseline record (no objective, no cases) and
# every later comparison is against it, until `vero session clear`. Opt in per
# run, with an id that names the run, and clear the session when the identity of
# what you are optimizing changes.
# id = "my-run-2026-08-01"
'''


def _parse_parameters(values: tuple[str, ...]) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for value in values:
        name, separator, encoded = value.partition("=")
        if not separator or not name.strip():
            raise click.BadParameter(
                "parameters must use NAME=JSON syntax",
                param_hint="--parameter",
            )
        if name in parameters:
            raise click.BadParameter(
                f"duplicate parameter {name!r}",
                param_hint="--parameter",
            )
        try:
            parameters[name] = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise click.BadParameter(
                f"parameter {name!r} is not valid JSON: {error.msg}",
                param_hint="--parameter",
            ) from error
    return parameters


def _command(value: str, option: str) -> list[str]:
    try:
        command = shlex.split(value)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint=option) from error
    if not command:
        raise click.BadParameter("command must not be empty", param_hint=option)
    return command


def _parse_environment(
    values: tuple[str, ...],
    *,
    option: str,
) -> dict[str, str]:
    environment: dict[str, str] = {}
    for value in values:
        name, separator, content = value.partition("=")
        if not separator or not name or "=" in name:
            raise click.BadParameter(
                "values must use NAME=VALUE syntax", param_hint=option
            )
        if name in environment:
            raise click.BadParameter(
                f"duplicate environment variable {name!r}", param_hint=option
            )
        environment[name] = content
    return environment


def _parse_constraints(
    values: tuple[tuple[str, str, str], ...],
) -> list[MetricConstraint]:
    constraints: list[MetricConstraint] = []
    for metric_value, operator_value, target_value in values:
        metric, separator, aggregation_value = metric_value.partition(":")
        if not metric:
            raise click.BadParameter("constraint metric must not be empty")
        try:
            aggregation = (
                MetricAggregation(aggregation_value)
                if separator
                else MetricAggregation.REPORT
            )
            operator = ConstraintOperator(operator_value)
            target = float(target_value)
            constraints.append(
                MetricConstraint(
                    selector=MetricSelector(
                        metric=metric,
                        aggregation=aggregation,
                    ),
                    operator=operator,
                    value=target,
                )
            )
        except (ValueError, TypeError) as error:
            raise click.BadParameter(
                "constraints use METRIC[:AGGREGATION] OP VALUE; "
                "OP is one of ==, !=, <, <=, >, >=",
                param_hint="--constraint",
            ) from error
    return constraints


def _print_result(session, result) -> None:
    click.echo(f"Session: {session.session_dir}")
    click.echo(
        f"Baseline: {result.baseline.request.candidate.id} "
        f"({result.baseline.objective.value if result.baseline.objective else 'n/a'})"
    )
    if result.best is None:
        click.echo("Best: no feasible candidate")
    else:
        click.echo(
            f"Best: {result.best.request.candidate.id} "
            f"({result.best.objective.value if result.best.objective else 'n/a'})"
        )


async def _run_configured(config_path: Path, *, optimize: bool):
    from vero.config import build_configured_runtime, load_config

    runtime = await build_configured_runtime(
        load_config(config_path),
        optimize=optimize,
    )
    result = await runtime.session.run(
        skip_baseline_evaluation=runtime.session.manifest_path.exists(),
        max_proposals=None if optimize else 0,
    )
    return runtime.session, result


@click.group()
def main() -> None:
    """VeRO: a harness for agents to optimize programs."""


@main.command(name="init")
@click.argument(
    "directory",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("."),
)
def initialize_config(directory: Path) -> None:
    """Create a commented-safe train/validation/test vero.toml starter."""

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "vero.toml"
    if destination.exists():
        raise click.ClickException(f"configuration already exists: {destination}")
    destination.write_text(_CONFIG_TEMPLATE, encoding="utf-8")
    click.echo(f"Created {destination}")


@main.command(name="check")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("vero.toml"),
    show_default=True,
)
def check_config(config_path: Path) -> None:
    """Validate configuration, paths, Git state, and evaluation references."""

    try:
        config = load_config(config_path)
        target = Path(config.target.root)
        harness = Path(config.backend.harness_root)
        if not target.is_dir():
            raise ValueError(f"target root does not exist: {target}")
        if not harness.is_dir():
            raise ValueError(f"evaluation harness root does not exist: {harness}")
        for name, path in config.backend.staged_inputs.items():
            if not Path(path).exists():
                raise ValueError(f"staged input {name!r} does not exist: {path}")
        subprocess.run(
            ["git", "rev-parse", "--verify", config.target.ref],
            cwd=target,
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=target,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty.strip():
            raise ValueError("target repository has uncommitted changes")
    except Exception as error:
        raise click.ClickException(str(error) or type(error).__name__) from error
    click.echo(
        f"Configuration is valid: {len(config.evaluations)} evaluations, "
        f"selection={config.protocol.selection_evaluation!r}, "
        f"final={config.protocol.final_evaluation!r}"
    )


@main.command(name="evaluate")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("vero.toml"),
    show_default=True,
)
def evaluate_config(config_path: Path) -> None:
    """Evaluate the configured baseline without producing candidates."""

    try:
        session, result = asyncio.run(_run_configured(config_path, optimize=False))
    except Exception as error:
        raise click.ClickException(str(error) or type(error).__name__) from error
    _print_result(session, result)


@main.command(name="run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("vero.toml"),
    show_default=True,
)
def run_config(config_path: Path) -> None:
    """Run the optimization declared in vero.toml."""

    try:
        session, result = asyncio.run(_run_configured(config_path, optimize=True))
    except Exception as error:
        raise click.ClickException(str(error) or type(error).__name__) from error
    _print_result(session, result)


@main.command(name="report")
@click.argument(
    "session_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Portable HTML path; defaults to SESSION_DIR/experiment.html.",
)
def report_session(session_dir: Path, output: Path | None) -> None:
    """Build a self-contained visual report for an optimization session."""

    from vero.report import generate_experiment_report

    try:
        destination = asyncio.run(generate_experiment_report(session_dir, output))
    except Exception as error:
        raise click.ClickException(str(error) or type(error).__name__) from error
    click.echo(destination)


@main.command()
@click.argument(
    "project_path",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@click.option(
    "--harness-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Trusted directory containing the evaluation harness.",
)
@click.option(
    "--evaluate",
    "evaluation_command",
    required=True,
    help="Evaluation argv with placeholders such as {workspace} and {report}.",
)
@click.option(
    "--producer-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Trusted directory containing an external candidate producer.",
)
@click.option(
    "--produce",
    "producer_command",
    help="Producer argv with placeholders such as {workspace} and {instruction}.",
)
@click.option(
    "--agent",
    type=click.Choice(["claude", "vero"]),
    help="Use a built-in coding-agent producer instead of --produce.",
)
@click.option("--instruction", help="Instruction given to the candidate producer.")
@click.option("--metric", required=True, help="Metric to optimize.")
@click.option(
    "--aggregation",
    type=click.Choice([value.value for value in MetricAggregation]),
    default=MetricAggregation.REPORT.value,
    show_default=True,
)
@click.option(
    "--case-failure-value",
    type=float,
    help="Value assigned to failed or missing cases during case aggregation.",
)
@click.option(
    "--direction",
    type=click.Choice(["maximize", "minimize"]),
    required=True,
)
@click.option("--failure-value", type=float)
@click.option(
    "--constraint",
    type=(str, str, str),
    multiple=True,
    metavar="METRIC[:AGGREGATION] OP VALUE",
    help="Feasibility constraint; repeat for multiple constraints.",
)
@click.option("--evaluation-set", default="default", show_default=True)
@click.option("--partition")
@click.option("--case-id", multiple=True, help="Evaluate only this case; repeatable.")
@click.option("--case-start", default=0, type=click.IntRange(min=0), show_default=True)
@click.option("--case-stop", type=click.IntRange(min=1))
@click.option(
    "--parameter",
    multiple=True,
    help="Evaluation parameter as NAME=JSON; repeat for multiple values.",
)
@click.option(
    "--evaluation-env",
    multiple=True,
    help="Environment variable to pass through to the evaluation harness.",
)
@click.option(
    "--evaluation-variable",
    multiple=True,
    help="Fixed harness environment variable as NAME=VALUE; repeatable.",
)
@click.option(
    "--producer-env",
    multiple=True,
    help="Environment variable to pass through to an external producer.",
)
@click.option(
    "--producer-variable",
    multiple=True,
    help="Fixed producer environment variable as NAME=VALUE; repeatable.",
)
@click.option("--evaluation-working-directory", default=".", show_default=True)
@click.option("--producer-working-directory", default=".", show_default=True)
@click.option(
    "--target-ref",
    default="HEAD",
    show_default=True,
    help="Git ref to use as the immutable baseline.",
)
@click.option(
    "--session-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Durable output directory; defaults to $VERO_HOME/sessions/<id>.",
)
@click.option("--session-id", help="Stable session identity.")
@click.option(
    "--max-proposals", default=1, type=click.IntRange(min=0), show_default=True
)
@click.option(
    "--max-rounds", default=100, type=click.IntRange(min=1), show_default=True
)
@click.option(
    "--max-concurrency", default=1, type=click.IntRange(min=1), show_default=True
)
@click.option("--max-turns", default=200, type=click.IntRange(min=1), show_default=True)
@click.option(
    "--evaluation-timeout",
    "--timeout",
    "evaluation_timeout",
    default=600.0,
    type=click.FloatRange(min=0, min_open=True),
    show_default=True,
    help="Overall timeout for one evaluation. --timeout is a deprecated alias.",
)
@click.option(
    "--producer-timeout",
    default=600.0,
    type=click.FloatRange(min=0, min_open=True),
    show_default=True,
    help="Timeout for one external command-producer attempt.",
)
@click.option(
    "--case-timeout",
    default=180.0,
    type=click.FloatRange(min=0, min_open=True),
    show_default=True,
)
@click.option(
    "--evaluation-concurrency",
    default=100,
    type=click.IntRange(min=1),
    show_default=True,
)
@click.option(
    "--error-rate-threshold",
    default=0.1,
    type=click.FloatRange(min=0, max=1, min_open=True),
    show_default=True,
    help="Fail an evaluation when this fraction of selected cases errors.",
)
@click.option(
    "--retry-max-attempts",
    default=3,
    type=click.IntRange(min=1),
    show_default=True,
    help="Maximum attempts for a transient per-case failure.",
)
@click.option(
    "--retry-initial-delay",
    default=4.0,
    type=click.FloatRange(min=0),
    show_default=True,
)
@click.option(
    "--retry-maximum-delay",
    default=120.0,
    type=click.FloatRange(min=0),
    show_default=True,
)
@click.option(
    "--retry-multiplier",
    default=2.0,
    type=click.FloatRange(min=1),
    show_default=True,
)
@click.option(
    "--retry-on-timeout/--no-retry-on-timeout",
    default=True,
    show_default=True,
)
@click.option("--seed", type=int)
@click.option("--wandb-project", help="Log the session to this W&B project.")
@click.option("--wandb-entity")
@click.option("--wandb-name")
@click.option(
    "--wandb-mode",
    type=click.Choice(["online", "offline", "disabled"]),
)
def optimize(
    project_path: Path,
    harness_root: Path,
    evaluation_command: str,
    producer_root: Path | None,
    producer_command: str | None,
    agent: str | None,
    instruction: str | None,
    metric: str,
    aggregation: str,
    case_failure_value: float | None,
    direction: str,
    failure_value: float | None,
    constraint: tuple[tuple[str, str, str], ...],
    evaluation_set: str,
    partition: str | None,
    case_id: tuple[str, ...],
    case_start: int,
    case_stop: int | None,
    parameter: tuple[str, ...],
    evaluation_env: tuple[str, ...],
    evaluation_variable: tuple[str, ...],
    producer_env: tuple[str, ...],
    producer_variable: tuple[str, ...],
    evaluation_working_directory: str,
    producer_working_directory: str,
    target_ref: str,
    session_dir: Path | None,
    session_id: str | None,
    max_proposals: int,
    max_rounds: int,
    max_concurrency: int,
    max_turns: int,
    evaluation_timeout: float,
    producer_timeout: float,
    case_timeout: float,
    evaluation_concurrency: int,
    error_rate_threshold: float,
    retry_max_attempts: int,
    retry_initial_delay: float,
    retry_maximum_delay: float,
    retry_multiplier: float,
    retry_on_timeout: bool,
    seed: int | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_name: str | None,
    wandb_mode: str | None,
) -> None:
    """Optimize the versioned program at PROJECT_PATH."""

    producer_count = int(producer_command is not None) + int(agent is not None)
    if producer_count > 1 or (max_proposals > 0 and producer_count != 1):
        raise click.UsageError(
            "provide exactly one of --produce or --agent when producing candidates"
        )
    if producer_command is not None and producer_root is None:
        raise click.UsageError("--producer-root is required with --produce")
    if producer_command is None and producer_root is not None:
        raise click.UsageError("--producer-root is only valid with --produce")
    if agent is None and max_turns != 200:
        raise click.UsageError("--max-turns is only valid with --agent")
    if producer_command is None and (
        producer_env
        or producer_variable
        or producer_working_directory != "."
        or producer_timeout != 600.0
    ):
        raise click.UsageError(
            "--producer-env, --producer-variable, --producer-working-directory, "
            "and --producer-timeout are only valid with --produce"
        )
    if wandb_project is None and any(
        value is not None for value in (wandb_entity, wandb_name, wandb_mode)
    ):
        raise click.UsageError(
            "--wandb-entity, --wandb-name, and --wandb-mode require --wandb-project"
        )
    if case_id and case_stop is not None:
        raise click.UsageError("--case-id cannot be combined with --case-stop")
    if case_stop is None and case_start != 0:
        raise click.UsageError("--case-start requires --case-stop")
    if case_stop is not None and case_stop <= case_start:
        raise click.UsageError("--case-stop must be greater than --case-start")

    if case_id:
        selection = CaseIds(ids=list(case_id))
    elif case_stop is not None:
        selection = CaseRange(start=case_start, stop=case_stop)
    else:
        selection = AllCases()

    if session_id is None and session_dir is not None:
        manifest_path = session_dir / "manifest.json"
        if manifest_path.exists():
            session_id = SessionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            ).id
    resolved_session_id = session_id or (
        session_dir.name if session_dir is not None else str(uuid4())
    )
    resolved_session_dir = (
        session_dir.resolve()
        if session_dir is not None
        else _default_home() / "sessions" / resolved_session_id
    )

    async def run():
        backend = CommandBackend(
            CommandBackendConfig(
                harness_root=str(harness_root.resolve()),
                command=_command(evaluation_command, "--evaluate"),
                working_directory=evaluation_working_directory,
                environment=_parse_environment(
                    evaluation_variable, option="--evaluation-variable"
                ),
                passthrough_environment=list(evaluation_env),
            )
        )
        if producer_command is not None:
            assert producer_root is not None
            producer = CommandCandidateProducer(
                CommandCandidateProducerConfig(
                    root=str(producer_root.resolve()),
                    command=_command(producer_command, "--produce"),
                    working_directory=producer_working_directory,
                    environment=_parse_environment(
                        producer_variable, option="--producer-variable"
                    ),
                    passthrough_environment=list(producer_env),
                    timeout_seconds=producer_timeout,
                )
            )
        elif agent is not None:
            from vero.agents import AgentCandidateProducer

            if agent == "claude":
                from vero.agents import ClaudeCodeAgent

                coding_agent = ClaudeCodeAgent()
            else:
                from vero.agents import VeroAgent

                coding_agent = VeroAgent()
            producer = AgentCandidateProducer(
                coding_agent,
                prompt=instruction,
                max_turns=max_turns,
            )
        else:
            producer = None

        session = await create_local_optimization_session(
            project_path=project_path,
            session_dir=resolved_session_dir,
            session_id=resolved_session_id,
            backend_id="command",
            backend=backend,
            objective=ObjectiveSpec(
                selector=MetricSelector(
                    metric=metric,
                    aggregation=MetricAggregation(aggregation),
                    case_failure_value=case_failure_value,
                ),
                direction=direction,
                failure_value=failure_value,
                constraints=_parse_constraints(constraint),
            ),
            evaluation_plan=EvaluationPlan.single(
                EvaluationSet(
                    name=evaluation_set,
                    partition=partition,
                    selection=selection,
                )
            ),
            strategy=SequentialStrategy(instruction=instruction),
            producers={"default": producer} if producer is not None else {},
            parameters=_parse_parameters(parameter),
            limits=EvaluationLimits(
                timeout_seconds=evaluation_timeout,
                case_timeout_seconds=case_timeout,
                max_concurrency=evaluation_concurrency,
                error_rate_threshold=error_rate_threshold,
                retry=RetryPolicy(
                    max_attempts=retry_max_attempts,
                    initial_delay_seconds=retry_initial_delay,
                    maximum_delay_seconds=retry_maximum_delay,
                    multiplier=retry_multiplier,
                    retry_on_timeout=retry_on_timeout,
                ),
            ),
            seed=seed,
            max_proposals=max_proposals,
            max_rounds=max_rounds,
            max_concurrency=max_concurrency,
            base_ref=target_ref,
            metadata={"project_path": str(project_path.resolve())},
        )
        if wandb_project is not None:
            assert session.events is not None
            session.events.sinks.append(
                WandbEventSink(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name,
                    mode=wandb_mode,
                    session_id=session.id,
                    session_dir=session.session_dir,
                    config={
                        "vero/target": str(project_path.resolve()),
                        "vero/evaluation_set": evaluation_set,
                        "vero/objective_metric": metric,
                        "vero/objective_direction": direction,
                    },
                )
            )
        result = await session.run(
            skip_baseline_evaluation=session.manifest_path.exists()
        )
        return session, result

    try:
        session, result = asyncio.run(run())
    except click.ClickException:
        raise
    except Exception as error:
        raise click.ClickException(str(error) or type(error).__name__) from error

    _print_result(session, result)


@main.group()
def session() -> None:
    """Inspect durable optimization sessions."""


@session.command(name="list")
@click.option(
    "--root",
    type=click.Path(path_type=Path, file_okay=False),
    help="Sessions directory; defaults to $VERO_HOME/sessions.",
)
def session_list(root: Path | None) -> None:
    """List session manifests."""

    root = root.resolve() if root is not None else _default_home() / "sessions"
    if not root.exists():
        click.echo("No sessions found.")
        return
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        click.echo("No sessions found.")
        return
    for path in manifests:
        try:
            manifest = SessionManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            click.echo(
                f"{manifest.id}\t{manifest.status.value}\t"
                f"{manifest.best_candidate_id or '-'}\t{path.parent.relative_to(root)}"
            )
        except Exception as error:
            click.echo(f"{path.parent.relative_to(root)}\tinvalid\t{error}")


@session.command(name="inspect")
@click.argument(
    "session_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
def session_inspect(session_dir: Path) -> None:
    """Print a canonical session manifest and evaluation summaries as JSON."""

    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        raise click.ClickException(f"session manifest not found: {manifest_path}")
    try:
        manifest = SessionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise click.ClickException(f"invalid session manifest: {error}") from error
    database_path = session_dir / "database.json"
    try:
        database = (
            EvaluationDatabase.load_from_file(database_path)
            if database_path.exists()
            else EvaluationDatabase.from_evaluations_dir(
                session_dir / "evaluations",
                database_id=manifest.id,
            )
        )
    except Exception as error:
        raise click.ClickException(f"invalid evaluation database: {error}") from error
    evaluations = sorted(
        database.evaluations.values(),
        key=lambda record: (record.completed_at, record.id),
    )
    try:
        if manifest.candidate_repository_family != "git":
            raise ValueError(
                "unsupported candidate repository family: "
                f"{manifest.candidate_repository_family}"
            )
        candidate_repository = asyncio.run(
            GitCandidateRepository.open(session_dir / "candidates")
        )
        candidates = candidate_repository.list()
    except Exception as error:
        raise click.ClickException(f"invalid candidate repository: {error}") from error
    click.echo(
        json.dumps(
            {
                "manifest": manifest.model_dump(mode="json"),
                "candidates": [
                    candidate.model_dump(mode="json") for candidate in candidates
                ],
                "evaluations": [
                    project_evaluation(record, DisclosureLevel.AGGREGATE).model_dump(
                        mode="json"
                    )
                    for record in evaluations
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@session.command(name="export")
@click.argument(
    "session_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Archive path without, or with, a .tar.gz suffix.",
)
def session_export(session_dir: Path, output: Path | None) -> None:
    """Export complete durable session state as a portable tar.gz archive."""

    session_dir = session_dir.resolve()
    if not (session_dir / "manifest.json").is_file():
        raise click.ClickException("session manifest not found")
    destination = (output or session_dir.with_name(f"{session_dir.name}-export"))
    destination = destination.expanduser().resolve()
    archive_base = str(destination)
    if archive_base.endswith(".tar.gz"):
        archive_base = archive_base[: -len(".tar.gz")]
    try:
        archive = shutil.make_archive(
            archive_base,
            "gztar",
            root_dir=session_dir.parent,
            base_dir=session_dir.name,
        )
    except Exception as error:
        raise click.ClickException(str(error) or type(error).__name__) from error
    click.echo(archive)


@session.command(name="fork")
@click.argument(
    "source",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@click.argument(
    "destination",
    type=click.Path(path_type=Path, file_okay=False),
)
@click.option("--session-id", help="New session ID; defaults to destination name.")
@click.option(
    "--max-proposals",
    type=click.IntRange(min=0),
    help="New protocol proposal limit; edit vero.toml to the same value.",
)
@click.option(
    "--reset-budgets",
    is_flag=True,
    help="Restore configured agent/system budgets instead of carrying balances.",
)
def session_fork(
    source: Path,
    destination: Path,
    session_id: str | None,
    max_proposals: int | None,
    reset_budgets: bool,
) -> None:
    """Fork durable candidates and evaluations into a new resumable session."""

    source = source.resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise click.ClickException(f"destination already exists: {destination}")
    if destination.is_relative_to(source):
        raise click.ClickException("fork destination must not be inside the source")
    try:
        manifest = SessionManifest.model_validate_json(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        new_id = session_id or destination.name
        if not new_id.strip():
            raise ValueError("session ID must not be empty")
        shutil.copytree(source, destination)
        run = manifest.run.model_copy(
            update=(
                {"max_proposals": max_proposals}
                if max_proposals is not None
                else {}
            )
        )
        forked_at = datetime.now(UTC)
        forked = manifest.model_copy(
            update={
                "id": new_id,
                "status": SessionStatus.CREATED,
                "run": run,
                "created_at": forked_at,
                "updated_at": forked_at,
                "failure": None,
                "metadata": {
                    **manifest.metadata,
                    "forked_from_session_id": manifest.id,
                },
            }
        )
        (destination / "manifest.json").write_text(
            forked.model_dump_json(indent=2),
            encoding="utf-8",
        )
        database_path = destination / "database.json"
        if database_path.exists():
            database = json.loads(database_path.read_text(encoding="utf-8"))
            database["id"] = new_id
            database_path.write_text(
                json.dumps(database, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        for transient in ("events.jsonl", "agent-context.json"):
            (destination / transient).unlink(missing_ok=True)
        wandb_state = destination / "artifacts" / "wandb"
        if wandb_state.exists():
            shutil.rmtree(wandb_state)
        if reset_budgets:
            budget_path = destination / "budgets.json"
            if forked.evaluation_plan.budgets:
                BudgetLedger(
                    forked.evaluation_plan.budgets,
                    path=budget_path,
                ).save()
            else:
                budget_path.unlink(missing_ok=True)
    except Exception as error:
        if destination.exists():
            shutil.rmtree(destination)
        raise click.ClickException(str(error) or type(error).__name__) from error
    click.echo(f"Forked {manifest.id} to {new_id} at {destination}")


@session.command(name="clear")
@click.argument(
    "session_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
@click.option("--yes", is_flag=True, help="Confirm permanent deletion.")
def session_clear(session_dir: Path, yes: bool) -> None:
    """Permanently delete one session's control-plane state."""

    if not yes:
        raise click.UsageError("session clear requires --yes")
    session_dir = session_dir.resolve()
    if not (session_dir / "manifest.json").is_file():
        raise click.ClickException("refusing to clear a directory without manifest.json")
    shutil.rmtree(session_dir)
    click.echo(f"Deleted {session_dir}")


# harbor subcommand registered here, after `main` is defined
from vero.harbor.cli import harbor as harbor_command  # noqa: E402, I001

main.add_command(harbor_command)


if __name__ == "__main__":
    main()
