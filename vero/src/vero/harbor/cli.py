"""CLI clients and server entry point for Harbor sidecar deployments."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import click

from vero.evaluation import (
    CaseIds,
    CaseRange,
    EvaluationLimits,
    EvaluationSet,
    RetryPolicy,
)
from vero.layout import LAYOUT
from vero.sidecar.auth import read_admin_token
from vero.sidecar.session import (
    create_harbor_session_archive,
    extract_harbor_session_archive,
    file_sha256,
)
from vero.sidecar.sidecar import SidecarEvaluationRequest


def _base_url() -> str:
    value = os.environ.get(LAYOUT.eval_url_env)
    if not value:
        raise click.ClickException(f"{LAYOUT.eval_url_env} is not set")
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
        raise click.ClickException(
            f"could not reach evaluation sidecar: {error}"
        ) from error


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download(
    path: str,
    destination: Path,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    request = urllib.request.Request(
        f"{_base_url()}{path}",
        method="GET",
        headers=headers or {},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            try:
                with urllib.request.urlopen(request) as response:
                    shutil.copyfileobj(response, file, length=1024 * 1024)
            except urllib.error.HTTPError as error:
                message = error.read().decode("utf-8", errors="replace")
                raise click.ClickException(
                    f"GET {path} returned {error.code}: {message}"
                ) from error
            except urllib.error.URLError as error:
                raise click.ClickException(
                    f"could not reach evaluation sidecar: {error}"
                ) from error
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _redact_trace_text(value: str) -> str:
    value = re.sub(
        r"(?m)^([A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)=).*$",
        r"\1[REDACTED]",
        value,
    )
    value = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~-]{16,}",
        r"\1[REDACTED]",
        value,
    )
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", value)


def _redact_trace_value(value: object) -> object:
    if isinstance(value, str):
        return _redact_trace_text(value)
    if isinstance(value, list):
        return [_redact_trace_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_trace_value(item) for key, item in value.items()}
    return value


def _load_agent_trace(path: Path) -> object:
    text = _redact_trace_text(path.read_text(encoding="utf-8", errors="replace"))
    try:
        return _redact_trace_value(json.loads(text))
    except json.JSONDecodeError:
        values = []
        for line in text.splitlines():
            try:
                values.append(_redact_trace_value(json.loads(line)))
            except json.JSONDecodeError:
                continue
        if not values:
            return [{"role": "assistant", "content": text}]

    entries: list[dict] = []
    for value in values:
        if not isinstance(value, dict):
            entries.append({"type": "event", "value": value})
            continue
        item = value.get("item")
        if value.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "agent_message":
            entries.append({"role": "assistant", "content": item.get("text", "")})
        elif item_type == "command_execution":
            entries.extend(
                [
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": item.get("command", ""),
                    },
                    {
                        "type": "function_call_output",
                        "output": item.get("aggregated_output", ""),
                    },
                ]
            )
        elif item_type == "error":
            entries.append(
                {"type": "error", "message": item.get("message", "unknown error")}
            )
    return entries or values


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv-style ``NAME=VALUE`` secrets file.

    Blank lines and ``#`` comments are ignored; a leading ``export`` is
    tolerated and surrounding single/double quotes are stripped. Values are
    kept verbatim otherwise (no shell/variable expansion) so a secret cannot
    be silently mangled.
    """
    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise click.ClickException(
                f"{path}:{lineno}: expected NAME=VALUE with a valid NAME, got {raw!r}"
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def _compiled_run_environment(
    task: Path, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Separate provider credentials from the environment seen by Harbor agents.

    ``overrides`` (e.g. a loaded ``--env-file``) take precedence over the
    ambient environment so a run is reproducible from an explicit secrets file
    regardless of what happens to be exported in the shell.
    """
    environment = os.environ.copy()
    if overrides:
        environment.update(overrides)
    path = task / "environment/gateway/launch.json"
    if not path.exists():
        return environment
    try:
        launch = json.loads(path.read_text(encoding="utf-8"))
        api_source = launch["upstream_api_key_source"]
        api_target = launch["upstream_api_key_target"]
        producer_api_key = launch["producer_api_key"]
        producer_base_url = launch["producer_base_url"]
        base_source = launch.get("upstream_base_url_source")
        base_target = launch["upstream_base_url_target"]
    except (KeyError, json.JSONDecodeError, OSError, TypeError) as error:
        raise click.ClickException(
            f"invalid compiled gateway launch config: {error}"
        ) from error
    for name, value in (
        ("upstream_api_key_source", api_source),
        ("upstream_api_key_target", api_target),
        ("producer_api_key", producer_api_key),
        ("producer_base_url", producer_base_url),
        ("upstream_base_url_target", base_target),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(f"invalid compiled gateway field {name}")
    upstream_api_key = environment.get(api_source)
    if not upstream_api_key:
        raise click.ClickException(
            f"upstream inference credential {api_source} is missing"
        )
    environment[api_target] = upstream_api_key
    if base_source is not None:
        if not isinstance(base_source, str) or not base_source:
            raise click.ClickException(
                "invalid compiled gateway field upstream_base_url_source"
            )
        upstream_base_url = environment.get(base_source)
        if not upstream_base_url:
            raise click.ClickException(
                f"upstream inference base URL {base_source} is missing"
            )
        environment[base_target] = upstream_base_url
    environment["OPENAI_API_KEY"] = producer_api_key
    environment["OPENAI_BASE_URL"] = producer_base_url
    # Claude Code (harbor -a claude) reads the Anthropic surface from the host
    # env; point it at the same producer scope. The Anthropic SDK re-appends
    # "/v1/messages", so hand it the scope root without the trailing "/v1".
    # Set unconditionally: codex ignores ANTHROPIC_*, claude ignores OPENAI_*, so
    # one compiled task serves either optimizer via `--agent`.
    environment["ANTHROPIC_API_KEY"] = producer_api_key
    environment["ANTHROPIC_BASE_URL"] = producer_base_url[: -len("/v1")] if (
        producer_base_url.endswith("/v1")
    ) else producer_base_url
    return environment


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


def _parse_build_params(values: tuple[str, ...]) -> dict[str, str]:
    """Parse repeatable ``--param NAME=VALUE`` into a substitution context."""
    params: dict[str, str] = {}
    for item in values:
        name, separator, value = item.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise click.ClickException(
                f"--param must be NAME=VALUE with a valid NAME: {item!r}"
            )
        params[name] = value
    return params


_PARAM_OPTION = click.option(
    "--param",
    "params",
    multiple=True,
    metavar="NAME=VALUE",
    help="Substitute ${NAME} in the build YAML; repeatable. Overrides the environment.",
)


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
@_PARAM_OPTION
def build_command(config_path, output, params):
    """Compile a build YAML into a runnable Harbor task directory."""
    from vero.harbor.build import compile_harbor_task, load_harbor_build_config

    compiled = compile_harbor_task(
        load_harbor_build_config(config_path, params=_parse_build_params(params)),
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
@click.option("--environment", default="modal", show_default=True)
@click.option(
    "--env-file",
    "env_file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help=(
        "Load NAME=VALUE run secrets (e.g. the upstream inference key and "
        "MODAL_TOKEN_ID/SECRET) from a dotenv file. File values take precedence "
        "over the ambient environment and never appear on the command line."
    ),
)
@_PARAM_OPTION
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def run_command(config_path, agent, model, environment, params, env_file, extra):
    """Compile to a temporary directory and invoke `harbor run`."""
    from vero.harbor.build import compile_harbor_task, load_harbor_build_config

    uvx = shutil.which("uvx")
    if uvx is None:
        raise click.ClickException("uvx is required to run a compiled Harbor task")
    overrides = _load_env_file(env_file) if env_file else None
    if overrides:
        click.echo(f"Loaded {len(overrides)} secret(s) from {env_file}")
        # Apply before compiling: the build's declared-credential check and the
        # ${NAME} param resolution both read os.environ, so the env-file must
        # satisfy them too — not just the launched subprocess.
        os.environ.update(overrides)
    resolved = _parse_build_params(params)
    # The optimizer model is both codex's -m and the producer scope's allow-list;
    # expose it as the reserved ${optimizer_model} so a build that references it
    # keeps the two in lockstep (no 403 model mismatch).
    if model is not None:
        resolved.setdefault("optimizer_model", model)
    config = load_harbor_build_config(config_path, params=resolved)
    with tempfile.TemporaryDirectory(prefix="vero-harbor-") as temporary:
        task = compile_harbor_task(
            config,
            Path(temporary) / "task",
        )
        command = [
            uvx,
            "--python",
            sys.executable,
            "--from",
            config.harbor_requirement,
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
        # Forward the build's declared agent env to the optimizer agent's shell.
        # Harbor's `--ae KEY=VALUE` populates the agent's extra_env, which harbor
        # injects into the agent's setup/install exec (scoped_exec_env). Sorted
        # for a deterministic command line.
        for key in sorted(config.agent_env):
            command.extend(["--ae", f"{key}={config.agent_env[key]}"])
        command.extend(extra)
        click.echo(shlex.join(command))
        completed = subprocess.run(
            command,
            env=_compiled_run_environment(task, overrides),
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)


@harbor.command("serve")
@click.option(
    "--factory", "factory_path", required=True, help="Trusted module:factory."
)
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
    from vero.sidecar.serve import serve

    serve(
        factory_path=factory_path,
        config_path=config_path,
        admin_token_path=admin_token_path,
        host=host,
        port=port,
    )


@harbor.command("inference-gateway")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8001, type=click.IntRange(1, 65535), show_default=True)
def inference_gateway_command(config_path, host, port):
    """Serve the credential-isolating, budgeted inference proxy."""
    from vero.gateway.inference import serve_inference_gateway

    serve_inference_gateway(config_path=config_path, host=host, port=port)


@harbor.command("eval")
@click.option("--backend", "backend_id", required=True)
@click.option("--evaluation-set", "evaluation_set_name", required=True)
@click.option("--partition")
@click.option("--version", help="Candidate version; defaults to agent repository HEAD.")
@click.option("--case-id", "case_ids", multiple=True)
@click.option("--start", type=click.IntRange(min=0))
@click.option("--stop", type=click.IntRange(min=1))
@click.option("--parameter", multiple=True, help="Evaluation parameter as NAME=JSON.")
@click.option("--timeout", type=click.FloatRange(min=0, min_open=True))
@click.option("--case-timeout", type=click.FloatRange(min=0, min_open=True))
@click.option("--max-concurrency", type=click.IntRange(min=1))
@click.option(
    "--error-rate-threshold",
    type=click.FloatRange(min=0, max=1, min_open=True),
)
@click.option("--retry-max-attempts", type=click.IntRange(min=1))
@click.option("--retry-initial-delay", type=click.FloatRange(min=0))
@click.option("--retry-maximum-delay", type=click.FloatRange(min=0))
@click.option("--retry-multiplier", type=click.FloatRange(min=1))
@click.option(
    "--retry-on-timeout/--no-retry-on-timeout",
    default=None,
)
@click.option("--seed", type=int)
@click.option(
    "--detach",
    is_flag=True,
    help="Start a durable evaluation job and return without waiting for its result.",
)
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
    error_rate_threshold,
    retry_max_attempts,
    retry_initial_delay,
    retry_maximum_delay,
    retry_multiplier,
    retry_on_timeout,
    seed,
    detach,
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
    retry_values = {
        name: value
        for name, value in {
            "max_attempts": retry_max_attempts,
            "initial_delay_seconds": retry_initial_delay,
            "maximum_delay_seconds": retry_maximum_delay,
            "multiplier": retry_multiplier,
            "retry_on_timeout": retry_on_timeout,
        }.items()
        if value is not None
    }
    limit_values = {
        name: value
        for name, value in {
            "timeout_seconds": timeout,
            "case_timeout_seconds": case_timeout,
            "max_concurrency": max_concurrency,
            "error_rate_threshold": error_rate_threshold,
        }.items()
        if value is not None
    }
    if retry_values:
        limit_values["retry"] = RetryPolicy(**retry_values)
    body = SidecarEvaluationRequest(
        backend_id=backend_id,
        evaluation_set=evaluation_set,
        version=version,
        parameters=_parameters(parameter),
        limits=EvaluationLimits(**limit_values) if limit_values else None,
        seed=seed,
    )
    click.echo(
        json.dumps(
            _request(
                "POST",
                "/eval/jobs" if detach else "/eval",
                payload=body.model_dump(mode="json"),
            ),
            indent=2,
        )
    )


@harbor.command("eval-status")
@click.argument("job_id")
def evaluation_status_command(job_id):
    """Inspect a detached evaluation job."""
    click.echo(json.dumps(_request("GET", f"/eval/jobs/{job_id}"), indent=2))


@harbor.command("eval-result")
@click.argument("job_id")
def evaluation_result_command(job_id):
    """Retrieve a detached evaluation result when it is available."""
    click.echo(json.dumps(_request("GET", f"/eval/jobs/{job_id}/result"), indent=2))


@harbor.command("submit")
@click.option("--version", help="Candidate version; defaults to agent repository HEAD.")
def submit_command(version):
    """Nominate a candidate for submit-based finalization."""
    click.echo(
        json.dumps(_request("POST", "/submit", payload={"version": version}), indent=2)
    )


@harbor.command("status")
def status_command():
    """Show agent-visible evaluation access and remaining budgets."""
    click.echo(json.dumps(_request("GET", "/status"), indent=2))


@harbor.command("score-baseline")
@click.option(
    "--token-file",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--replicates", default=1, show_default=True, type=click.IntRange(min=1))
def score_baseline_command(token_file, replicates):
    """Admin-score the fixed seed N times to produce a pinnable baseline number."""
    token = read_admin_token(token_file)
    result = _request(
        "POST",
        "/score/baseline",
        payload={"replicates": replicates},
        headers={"Authorization": f"Bearer {token}"},
    )
    click.echo(json.dumps(result, indent=2))


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
    # reward.json keeps only the reward map Harbor consumes.
    destination.write_text(
        json.dumps(result["rewards"], indent=2) + "\n",
        encoding="utf-8",
    )
    # Persist the full verification result (the shipped flag, verifier errors,
    # baseline rewards) alongside it, so "did anything ship, and if not why" is
    # answerable without re-running — reward.json alone drops those signals.
    (destination.parent / "finalization.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    click.echo(json.dumps(result, indent=2))


@harbor.command("export-session")
@click.option(
    "--token-file",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--output",
    default="/logs/verifier/session.tar.gz",
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
@click.option(
    "--report-output",
    default="/logs/verifier/experiment.html",
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
@click.option(
    "--status-output",
    default="/logs/verifier/status.json",
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
@click.option(
    "--finalization-output",
    default="/logs/verifier/finalization.json",
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
@click.option(
    "--agent-trace",
    default="/logs/agent/trajectory.json",
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
def export_session_command(
    token_file,
    output,
    report_output,
    status_output,
    finalization_output,
    agent_trace,
):
    """Persist the complete sidecar session and portable experiment report."""
    from vero.report import generate_experiment_report

    token = read_admin_token(token_file)
    headers = {"Authorization": f"Bearer {token}"}
    finalization = _request("POST", "/finalize", headers=headers)
    status = _request("GET", "/status")
    output = Path(output).expanduser().resolve()
    report_output = Path(report_output).expanduser().resolve()
    status_output = Path(status_output).expanduser().resolve()
    finalization_output = Path(finalization_output).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="vero-harbor-session-") as directory:
        temporary = Path(directory)
        downloaded = temporary / "sidecar-session.tar.gz"
        _download("/session/export", downloaded, headers=headers)
        session = extract_harbor_session_archive(downloaded, temporary / "extracted")
        encoded_finalization = (
            json.dumps(finalization, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        encoded_status = (
            json.dumps(status, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        _atomic_write_bytes(session / "harbor-finalization.json", encoded_finalization)
        _atomic_write_bytes(session / "harbor-status.json", encoded_status)

        requested_trace = Path(agent_trace).expanduser()
        trace_path = next(
            (
                path
                for path in (
                    requested_trace,
                    Path("/logs/agent/trajectory.json"),
                    Path("/logs/agent/codex.txt"),
                )
                if path.is_file() and not path.is_symlink()
            ),
            None,
        )
        if trace_path is not None:
            trace = _load_agent_trace(trace_path)
            trace_destination = (
                session / "artifacts" / "agents" / "harbor-producer" / "trace.json"
            )
            _atomic_write_bytes(
                trace_destination,
                (json.dumps(trace, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )

        generated_report = temporary / "experiment.html"
        asyncio.run(generate_experiment_report(session, generated_report))
        create_harbor_session_archive(session, output)
        digest = file_sha256(output)
        _atomic_write_bytes(report_output, generated_report.read_bytes())
        _atomic_write_bytes(status_output, encoded_status)
        _atomic_write_bytes(finalization_output, encoded_finalization)
        _atomic_write_bytes(
            output.with_name(f"{output.name}.sha256"),
            f"{digest}  {output.name}\n".encode("ascii"),
        )
    click.echo(
        json.dumps(
            {
                "session": str(output),
                "sha256": digest,
                "report": str(report_output),
            },
            indent=2,
        )
    )
