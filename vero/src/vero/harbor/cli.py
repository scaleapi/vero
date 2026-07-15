"""CLI clients and server entry point for Harbor sidecar deployments."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import click

from vero.evaluation import CaseIds, CaseRange, EvaluationLimits, EvaluationSet
from vero.harbor.auth import read_admin_token
from vero.harbor.sidecar import SidecarEvaluationRequest


def _base_url() -> str:
    value = os.environ.get("VERO_EVAL_URL")
    if not value:
        raise click.ClickException("VERO_EVAL_URL is not set")
    return value.rstrip("/")


def _request(
    method: str,
    path: str,
    *,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{_base_url()}{path}",
        method=method,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise click.ClickException(
            f"{method} {path} returned {error.code}: {message}"
        ) from error
    except urllib.error.URLError as error:
        raise click.ClickException(f"could not reach evaluation sidecar: {error}") from error


def _parameters(values: tuple[str, ...]) -> dict:
    parameters = {}
    for value in values:
        name, separator, encoded = value.partition("=")
        if not separator or not name.strip():
            raise click.BadParameter("use NAME=JSON", param_hint="--parameter")
        if name in parameters:
            raise click.BadParameter(
                f"duplicate parameter {name!r}",
                param_hint="--parameter",
            )
        try:
            parameters[name] = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise click.BadParameter(
                f"parameter {name!r} is not valid JSON",
                param_hint="--parameter",
            ) from error
    return parameters


@click.group()
def harbor() -> None:
    """Run VeRO across a Harbor sidecar boundary."""


@harbor.command("build")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
)
def build_command(config_path, output):
    """Compile a build YAML into a runnable Harbor task directory."""
    from vero.harbor.build import compile_harbor_task, load_harbor_build_config

    compiled = compile_harbor_task(
        load_harbor_build_config(config_path),
        output,
    )
    click.echo(f"Compiled Harbor task: {compiled}")


@harbor.command(
    "run",
    context_settings={"ignore_unknown_options": True},
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--agent", required=True, help="Harbor optimizer agent.")
@click.option("--model", help="Model used by the optimizer agent.")
@click.option("--environment", default="docker", show_default=True)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def run_command(config_path, agent, model, environment, extra):
    """Compile to a temporary directory and invoke `harbor run`."""
    from vero.harbor.build import compile_harbor_task, load_harbor_build_config

    uvx = shutil.which("uvx")
    if uvx is None:
        raise click.ClickException("uvx is required to run a compiled Harbor task")
    with tempfile.TemporaryDirectory(prefix="vero-harbor-") as temporary:
        task = compile_harbor_task(
            load_harbor_build_config(config_path),
            Path(temporary) / "task",
        )
        command = [
            uvx,
            "harbor",
            "run",
            "-p",
            str(task),
            "-a",
            agent,
            "-e",
            environment,
        ]
        if model is not None:
            command.extend(["-m", model])
        command.extend(extra)
        click.echo(shlex.join(command))
        completed = subprocess.run(command)
        if completed.returncode:
            raise SystemExit(completed.returncode)


@harbor.command("serve")
@click.option("--factory", "factory_path", required=True, help="Trusted module:factory.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--admin-token",
    "admin_token_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, type=click.IntRange(1, 65535), show_default=True)
def serve_command(factory_path, config_path, admin_token_path, host, port):
    """Serve components built by a trusted deployment factory."""
    from vero.harbor.serve import serve

    serve(
        factory_path=factory_path,
        config_path=config_path,
        admin_token_path=admin_token_path,
        host=host,
        port=port,
    )


@harbor.command("eval")
@click.option("--backend", "backend_id", required=True)
@click.option("--evaluation-set", "evaluation_set_name", required=True)
@click.option("--partition")
@click.option("--version", help="Candidate version; defaults to agent repository HEAD.")
@click.option("--case-id", "case_ids", multiple=True)
@click.option("--start", type=click.IntRange(min=0))
@click.option("--stop", type=click.IntRange(min=1))
@click.option("--parameter", multiple=True, help="Evaluation parameter as NAME=JSON.")
@click.option("--timeout", default=600.0, type=click.FloatRange(min=0, min_open=True))
@click.option("--case-timeout", default=180.0, type=click.FloatRange(min=0, min_open=True))
@click.option("--max-concurrency", default=20, type=click.IntRange(min=1))
@click.option("--seed", type=int)
def evaluate_command(
    backend_id,
    evaluation_set_name,
    partition,
    version,
    case_ids,
    start,
    stop,
    parameter,
    timeout,
    case_timeout,
    max_concurrency,
    seed,
):
    """Evaluate a candidate through the metered agent endpoint."""
    if case_ids and (start is not None or stop is not None):
        raise click.UsageError("--case-id cannot be combined with --start/--stop")
    if start is not None and stop is None:
        raise click.UsageError("--start requires --stop")
    selection = None
    if case_ids:
        selection = CaseIds(ids=list(case_ids))
    elif stop is not None:
        selection = CaseRange(start=start or 0, stop=stop)
    evaluation_set = EvaluationSet(
        name=evaluation_set_name,
        partition=partition,
        **({"selection": selection} if selection is not None else {}),
    )
    body = SidecarEvaluationRequest(
        backend_id=backend_id,
        evaluation_set=evaluation_set,
        version=version,
        parameters=_parameters(parameter),
        limits=EvaluationLimits(
            timeout_seconds=timeout,
            case_timeout_seconds=case_timeout,
            max_concurrency=max_concurrency,
        ),
        seed=seed,
    )
    click.echo(
        json.dumps(
            _request("POST", "/eval", payload=body.model_dump(mode="json")),
            indent=2,
        )
    )


@harbor.command("submit")
@click.option("--version", help="Candidate version; defaults to agent repository HEAD.")
def submit_command(version):
    """Nominate a candidate for submit-based finalization."""
    click.echo(json.dumps(_request("POST", "/submit", payload={"version": version}), indent=2))


@harbor.command("status")
def status_command():
    """Show agent-visible evaluation access and remaining budgets."""
    click.echo(json.dumps(_request("GET", "/status"), indent=2))


@harbor.command("finalize")
@click.option(
    "--token-file",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--output",
    default="/logs/verifier/reward.json",
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
def finalize_command(token_file, output):
    """Finalize as the trusted verifier and write Harbor rewards."""
    token = read_admin_token(token_file)
    result = _request(
        "POST",
        "/finalize",
        headers={"Authorization": f"Bearer {token}"},
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result["rewards"], indent=2) + "\n",
        encoding="utf-8",
    )
    click.echo(json.dumps(result, indent=2))
