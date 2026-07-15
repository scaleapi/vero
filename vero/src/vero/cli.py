"""Command-line interface for generic program optimization."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from uuid import uuid4

import click

from vero.evaluation import (
    CommandBackend,
    CommandBackendConfig,
    EvaluationLimits,
    EvaluationSet,
    MetricSelector,
    ObjectiveSpec,
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
    SequentialStrategy,
)
from vero.runtime import SessionManifest, create_local_optimization_session


def _default_home() -> Path:
    return Path(os.environ.get("VERO_HOME", "~/.vero")).expanduser().resolve()


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


@click.group()
def main() -> None:
    """VeRO: a harness for agents to optimize programs."""


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
    "--direction",
    type=click.Choice(["maximize", "minimize"]),
    required=True,
)
@click.option("--evaluation-set", default="default", show_default=True)
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
    "--producer-env",
    multiple=True,
    help="Environment variable to pass through to an external producer.",
)
@click.option(
    "--session-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Durable output directory; defaults to $VERO_HOME/sessions/<id>.",
)
@click.option("--session-id", help="Stable session identity.")
@click.option("--max-candidates", default=1, type=click.IntRange(min=0), show_default=True)
@click.option("--max-rounds", default=100, type=click.IntRange(min=1), show_default=True)
@click.option("--max-concurrency", default=1, type=click.IntRange(min=1), show_default=True)
@click.option("--max-turns", default=200, type=click.IntRange(min=1), show_default=True)
@click.option("--timeout", default=600.0, type=click.FloatRange(min=0, min_open=True), show_default=True)
@click.option("--seed", type=int)
def optimize(
    project_path: Path,
    harness_root: Path,
    evaluation_command: str,
    producer_root: Path | None,
    producer_command: str | None,
    agent: str | None,
    instruction: str | None,
    metric: str,
    direction: str,
    evaluation_set: str,
    parameter: tuple[str, ...],
    evaluation_env: tuple[str, ...],
    producer_env: tuple[str, ...],
    session_dir: Path | None,
    session_id: str | None,
    max_candidates: int,
    max_rounds: int,
    max_concurrency: int,
    max_turns: int,
    timeout: float,
    seed: int | None,
) -> None:
    """Optimize the versioned program at PROJECT_PATH."""

    if (producer_command is None) == (agent is None):
        raise click.UsageError("provide exactly one of --produce or --agent")
    if producer_command is not None and producer_root is None:
        raise click.UsageError("--producer-root is required with --produce")
    if producer_command is None and producer_root is not None:
        raise click.UsageError("--producer-root is only valid with --produce")

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
                passthrough_environment=list(evaluation_env),
            )
        )
        if producer_command is not None:
            assert producer_root is not None
            producer = CommandCandidateProducer(
                CommandCandidateProducerConfig(
                    root=str(producer_root.resolve()),
                    command=_command(producer_command, "--produce"),
                    passthrough_environment=list(producer_env),
                    timeout_seconds=timeout,
                )
            )
        else:
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

        session = await create_local_optimization_session(
            project_path=project_path,
            session_dir=resolved_session_dir,
            session_id=resolved_session_id,
            backend_id="command",
            backend=backend,
            objective=ObjectiveSpec(
                selector=MetricSelector(metric=metric),
                direction=direction,
            ),
            evaluation_set=EvaluationSet(name=evaluation_set),
            strategy=SequentialStrategy(instruction=instruction),
            producers={"default": producer},
            parameters=_parse_parameters(parameter),
            limits=EvaluationLimits(timeout_seconds=timeout),
            seed=seed,
            max_candidates=max_candidates,
            max_rounds=max_rounds,
            max_concurrency=max_concurrency,
            metadata={"project_path": str(project_path.resolve())},
        )
        if hasattr(producer, "artifacts") and producer.artifacts is None:
            producer.artifacts = session.artifacts
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
    manifests = sorted(root.glob("*/manifest.json"))
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
                f"{manifest.best_candidate_id or '-'}"
            )
        except Exception as error:
            click.echo(f"{path.parent.name}\tinvalid\t{error}")


@session.command(name="inspect")
@click.argument(
    "session_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
)
def session_inspect(session_dir: Path) -> None:
    """Print a canonical session manifest as JSON."""

    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        raise click.ClickException(f"session manifest not found: {manifest_path}")
    try:
        manifest = SessionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise click.ClickException(f"invalid session manifest: {error}") from error
    click.echo(manifest.model_dump_json(indent=2))


from vero.harbor.cli import harbor as harbor_command

main.add_command(harbor_command)


if __name__ == "__main__":
    main()
