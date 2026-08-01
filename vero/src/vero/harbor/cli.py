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


# opencode's own default is 100 agentic iterations, after which it forces a
# text-only response and stops. That is well inside a real optimization run:
# gaia's optimizer used all 100 and never reached `evals submit`. Set high
# enough that the harness never truncates the search -- the case budget and
# the gateway token cap are the intended limits.
OPENCODE_STEP_LIMIT = 1000

# Harnesses that drive the model through litellm rather than a provider SDK.
# litellm reads the base URL as <PROVIDER>_API_BASE; the SDKs read
# <PROVIDER>_BASE_URL. vero sets the SDK names, so a litellm-based harness sees no
# override and calls the provider's public endpoint instead of the gateway.
_LITELLM_AGENTS = frozenset({"mini-swe-agent", "swe-agent"})

# How long a dropped Modal stdio stream keeps trying to reconnect before the
# trial dies, and how often.
#
# Harbor reads a whole agent phase through one Modal stdio stream, and Modal
# budgets reconnects per *stream*: `stream_stdio_max_retries` is 10 for the life
# of the stream and is never replenished, because a successful chunk resets only
# the backoff delay. Shipped as 0.01s with a doubling factor, those ten attempts
# are spent in 10.23 seconds, so a multi-hour run is protected against ten
# seconds of network trouble and the next drop is fatal whenever it lands. Two
# cells died that way on 2026-07-31 and 2026-08-01.
#
# A flat delay rather than a doubling one, because the count is not the useful
# knob: with a factor of 2 the sleeps outgrow the run by roughly the seventeenth
# attempt, so raising the count alone converts a crash into a hang. Flat keeps
# every gap short and makes the tolerated outage the simple product below.
#
# MODAL_STREAM_RECONNECT_WINDOW_SECONDS is capped by how much stdout the worker
# keeps available for a reconnect at a byte offset. Past that the reconnect does
# not fail cleanly, it resumes past the retained span, so output goes missing
# instead of the run dying. Waiting longer would therefore be worse than dying,
# which is what bounds this and not the length of a typical outage.
#
# The retained span is measured in BYTES, not seconds, so the safe window in
# seconds scales inversely with how chatty the harness is. Probed against live
# sandboxes: 6 KB over 150s and 240 KB over 120s both reconnected contiguously,
# 3 MB over 30s came back empty. A measured opencode optimizer transcript runs
# ~350 B/s, so 120s is ~42 KB, inside the verified range with roughly 6x margin
# on the rate. Re-measure before raising this, and treat a much noisier harness
# as a reason to lower it.
#
# Note what this value is NOT: evidence that 120s covers real outages. Nothing
# here measures how long a drop actually lasts. It is the largest window that is
# safe, and the outer-trial retry remains the backstop past it.
MODAL_STREAM_RECONNECT_DELAY_SECONDS = 2.0
MODAL_STREAM_RECONNECT_WINDOW_SECONDS = 120
MODAL_STREAM_RECONNECT_FACTOR = 1.0

# Directory holding the `sitecustomize` that applies the above inside the harbor
# subprocess. Modal exposes these three only as constructor keywords that
# `TaskCommandRouterClient._connect` never forwards, and `modal/config.py` has no
# entry for them, so there is no supported path: not an argument, not an
# environment variable. PYTHONPATH plus `sitecustomize` is the seam that reaches
# a dependency's defaults without vendoring it.
_STREAM_PATCH_DIRECTORY = Path(__file__).parent / "_stream_patch"


def _litellm_base_url_args(agent: str, task: Path) -> list[str]:
    """Give litellm-based harnesses the gateway URL under the name they read.

    mini-swe-agent installs `litellm[proxy]` and resolves `anthropic/<model>`
    through it. Without ANTHROPIC_API_BASE it reaches api.anthropic.com holding
    only a scoped gateway token, and fails with
    ``AuthenticationError: invalid x-api-key`` -- fails closed, so nothing leaks,
    but the harness cannot run at all. Set both provider aliases from the
    compiled producer scope so whichever provider the model names is covered.
    """

    if agent not in _LITELLM_AGENTS:
        return []
    path = task / "environment/gateway/launch.json"
    if not path.exists():
        return []
    try:
        base_url = json.loads(path.read_text(encoding="utf-8"))["producer_base_url"]
    except (OSError, json.JSONDecodeError, KeyError):
        return []
    # litellm appends its own route to whatever base it is given, and the two
    # providers want different bases. Its openai path adds `/chat/completions`,
    # so that one takes the `/v1` producer URL as-is. Its anthropic path adds
    # `/v1/messages` unless the base already ends in exactly that (main.py
    # `anthropic_chat_completions`), so the same URL there yields a doubled
    # `/v1/v1/messages`, which the gateway forwards upstream as a route nobody
    # serves -- a 403 that reads like an auth failure. Hand anthropic the fully
    # qualified messages path, which litellm leaves alone either way.
    root = base_url.rstrip("/")
    for suffix in ("/v1/messages", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return [
        "--ae",
        f"OPENAI_API_BASE={base_url}",
        "--ae",
        f"ANTHROPIC_API_BASE={root}/v1/messages",
    ]


def _kimi_gateway_args(agent: str, task: Path) -> list[str]:
    """Point kimi-cli's provider at the gateway instead of api.openai.com.

    kimi-cli only accepts a model whose provider half is in its own table, and
    ``fireworks_ai`` is not; prefixing with ``openai/`` selects its
    ``openai_legacy`` provider and leaves the rest of the string as the model.
    That provider's base URL defaults to ``https://api.openai.com/v1``, so
    without an override the harness sends its scoped producer token to OpenAI
    and gets a 401 -- failing closed, so nothing leaks and the upstream key is
    never involved, but unable to run at all.

    The override goes through ``--ak base_url``, which harbor passes to the
    adapter's constructor and which lands in the provider block of the config
    file kimi-cli loads. kimi-cli *also* reads ``OPENAI_BASE_URL`` from its own
    process environment, but that never arrived: the variable is set on the
    container and passed as an agent env var, and neither reaches the harness
    here. Writing the config directly does not depend on any of that. The env
    pair is still set, harmlessly, for the key.
    """

    if agent != "kimi-cli":
        return []
    path = task / "environment/gateway/launch.json"
    if not path.exists():
        return []
    try:
        launch = json.loads(path.read_text(encoding="utf-8"))
        base_url = launch["producer_base_url"]
        api_key = launch["producer_api_key"]
    except (OSError, json.JSONDecodeError, KeyError):
        return []
    return [
        "--ak",
        f"base_url={base_url}",
        "--ae",
        f"OPENAI_BASE_URL={base_url}",
        "--ae",
        f"OPENAI_API_KEY={api_key}",
    ]


def _opencode_gateway_args(agent: str, model: str | None, task: Path) -> list[str]:
    """Route opencode's non-openai providers through the gateway.

    Harbor's opencode adapter writes a provider ``baseURL`` into
    ``opencode.json`` only when the provider half of ``provider/model`` is
    ``openai`` (``agents/installed/opencode.py``). For any other provider it
    writes none, so opencode calls that provider's public endpoint. That fails
    closed rather than leaking -- the optimizer only ever holds a scoped gateway
    token, and Anthropic answers ``401 invalid x-api-key`` -- but it does leave
    ``openai/`` as the only usable form, which forces the Responses API. Driving
    a Claude model that way breaks: litellm's Anthropic-to-Responses translation
    emits mixed id namespaces in one stream (a ``resp_`` id, Anthropic ``msg_``/
    ``toolu_`` item ids, and a stray ``chatcmpl-`` id), and opencode dies
    resolving a text part under an id it never registered.

    Supplying the baseURL ourselves keeps the traffic on the provider's own API
    -- Messages for Anthropic, which is the path claude-code already proves --
    and metered. The adapter deep-merges job kwargs last, so this wins.
    """

    if agent != "opencode" or not model:
        return []

    # opencode caps agentic iterations at 100 by default and then "forces a
    # text-only response" (its own config schema's words for `steps`). A gaia
    # optimizer hit that cap after ~2h, and the forced final message reads like a
    # considered wrap-up, so the truncation is invisible unless you notice the
    # step count is exactly 100 -- it never reached `evals submit`. claude-code
    # takes harbor's --max-turns instead, so leaving this at the default makes
    # the two harnesses incomparable. `build` is opencode's primary agent.
    payload: dict[str, object] = {
        "agent": {"build": {"steps": OPENCODE_STEP_LIMIT}}
    }

    provider, _, _ = model.partition("/")
    if "/" in model and provider != "openai":
        # The adapter injects a baseURL only for the openai provider; for any
        # other it writes none and opencode calls that provider's public
        # endpoint. Supply it ourselves so the traffic stays metered.
        path = task / "environment/gateway/launch.json"
        if path.exists():
            try:
                base_url = json.loads(path.read_text(encoding="utf-8"))[
                    "producer_base_url"
                ]
            except (OSError, json.JSONDecodeError, KeyError):
                base_url = None
            if base_url:
                # producer_base_url ends in /v1 and provider SDKs append their
                # own route (/messages), matching ANTHROPIC_BASE_URL for
                # claude-code.
                payload["provider"] = {provider: {"options": {"baseURL": base_url}}}
    return ["--ak", f"opencode_config={json.dumps(payload, separators=(',', ':'))}"]


def _outer_app_name_args(
    environment: str, config_name: str, extra: tuple[str, ...]
) -> list[str]:
    """Name the outer trial's Modal app so its sandbox can be found.

    Inner evaluation sandboxes are already grouped by an explicit
    ``--ek app_name=...`` in each build's ``extra_harbor_args``, but the outer
    trial had none and so landed in Modal's default ``__harbor__`` app. That
    costs twice: the workspace holds thousands of containers, so an unnamed outer
    sandbox is effectively unfindable in the UI, and recovering the session from
    a run that must be killed begins with identifying its container.

    Derived from the build name (``vero/optimize-gaia-baseline`` ->
    ``vero-optimize-gaia-baseline``) so outer trials group per benchmark. A
    caller passing its own ``--ek app_name=`` wins.
    """

    if environment != "modal":
        return []
    if any("app_name=" in argument for argument in extra):
        return []
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config_name).strip("-")
    return ["--ek", f"app_name={slug or 'vero'}"]


def _agent_environment_blanks(task: Path) -> list[str]:
    """`--ae NAME=` for every declared credential, so the optimizer cannot read it.

    The gateway service takes the real upstream key from the environment this
    module builds for the harbor subprocess, and harbor's *agent* inherits that
    same environment. Two optimizers ran `env` and their transcripts captured the
    live upstream key and base URL in plaintext. With both, an optimizer can call
    the provider directly -- unmetered, past the per-scope model allow-list and
    past its budget -- so the fixed-target guarantee is unenforceable while they
    are readable. That is a property of the plumbing, not of any particular
    credential, so it survives rotating the keys.

    Harbor merges exec environments as persistent < per-exec < scoped, and wraps
    the agent's setup and run phases in ``scoped_exec_env(agent.extra_env)``,
    which is what ``--ae`` populates. An explicit empty value therefore overrides
    the inherited one for the agent while leaving the gateway service, which
    reads the same names through compose interpolation, untouched.

    Every declared secret is blanked, not just the upstream inference credential.
    The compose template already blanks all of them on the candidate's main
    service and routes them to the trusted sidecar, which is their only legitimate
    consumer -- WANDB_API_KEY reaches the sidecar's own container, and inner
    evaluations shell out to harbor from inside the sidecar, so MODAL_TOKEN_* is
    needed there rather than in the agent. The agent exec was the one surface that
    promise did not cover. The list is computed at compile time and published to
    ``environment/agent-env-blanks.json`` so both surfaces blank the same names
    and cannot drift.

    A missing blank-list file yields the upstream names alone, so a task compiled
    before this existed still gets the credential that motivated it hidden.
    """
    names: list[str] = []
    blanks = task / "environment/agent-env-blanks.json"
    if blanks.exists():
        try:
            names.extend(json.loads(blanks.read_text(encoding="utf-8"))["names"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass  # fall back to the gateway names below rather than run unblanked
    path = task / "environment/gateway/launch.json"
    if path.exists():
        try:
            launch = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            launch = {}  # _compiled_run_environment reports this properly
        names.extend(
            [
                launch.get("upstream_api_key_target"),
                launch.get("upstream_base_url_target"),
                # The source names hold the same secret under the caller's own
                # spelling.
                launch.get("upstream_api_key_source"),
                launch.get("upstream_base_url_source"),
            ]
        )
    arguments: list[str] = []
    for name in sorted({n for n in names if isinstance(n, str) and n}):
        # Never blank a name the producer credential is delivered under, or the
        # optimizer loses its own inference.
        if name in LAYOUT.routed_credential_envs:
            continue
        arguments.extend(["--ae", f"{name}="])
    return arguments


def _modal_stream_patch_environment(base: dict[str, str]) -> dict[str, str]:
    """Put the reconnect-budget `sitecustomize` on the harbor subprocess's path.

    Prepended to any inherited PYTHONPATH rather than replacing it, and the
    `sitecustomize` chains to whichever module it shadows, so a caller who
    already relies on one keeps it.

    A caller who sets the three values explicitly keeps them: this fills in the
    vero defaults and never overrides an explicit choice, which is what makes the
    window tunable per run without a code change.
    """

    patched = {"PYTHONPATH": str(_STREAM_PATCH_DIRECTORY)}
    inherited = base.get("PYTHONPATH")
    if inherited:
        patched["PYTHONPATH"] = os.pathsep.join([patched["PYTHONPATH"], inherited])

    retries = int(
        MODAL_STREAM_RECONNECT_WINDOW_SECONDS / MODAL_STREAM_RECONNECT_DELAY_SECONDS
    )
    for name, value in (
        ("VERO_MODAL_STREAM_RETRY_DELAY_SECS", MODAL_STREAM_RECONNECT_DELAY_SECONDS),
        ("VERO_MODAL_STREAM_RETRY_FACTOR", MODAL_STREAM_RECONNECT_FACTOR),
        ("VERO_MODAL_STREAM_MAX_RETRIES", retries),
    ):
        if not base.get(name):
            patched[name] = str(value)
    return patched


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
    environment.update(_modal_stream_patch_environment(environment))
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


#: The two request shapes an OpenAI-compatible upstream may accept. Agents in
#: this repo use both (gaia is on Responses, the rest are on Chat Completions),
#: and an upstream is free to implement only one, so a 404 from the first is
#: not evidence about the model until the second has also been tried.
_PROBE_ROUTES: tuple[tuple[str, str], ...] = (
    ("/responses", "input"),
    ("/chat/completions", "messages"),
)


def _model_is_missing(body: str) -> bool:
    """True when a 404 body blames the model rather than the route.

    A route-level 404 (the upstream does not implement this path) and a
    model-level 404 (the deployment does not exist) share a status code and
    mean opposite things, so the body is the only thing that separates them.
    Matched on the providers' own error codes, and on the Azure and OpenAI
    sentences, never on a bare "does not exist".
    """
    lowered = body.lower()
    if "deploymentnotfound" in lowered or "model_not_found" in lowered:
        return True
    return "model" in lowered and "does not exist" in lowered


def _probe_route(
    base_url: str, api_key: str, model: str, route: str, input_key: str
) -> tuple[int | None, str]:
    """Ask one route for one token from `model`. Returns (status, body)."""
    payload: dict[str, object] = {"model": model}
    if input_key == "input":
        payload["input"] = "ok"
        payload["max_output_tokens"] = 16
    else:
        payload["messages"] = [{"role": "user", "content": "ok"}]
        payload["max_tokens"] = 16
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{route}",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Azure OpenAI authenticates on api-key; OpenAI ignores it.
            "api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, ""
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:400]
    except Exception as error:  # network/DNS/timeout: inconclusive, not fatal
        return None, str(error)


def _probe_model(base_url: str, api_key: str, model: str) -> tuple[int | None, str]:
    """Ask the upstream for one token from `model`. Returns (status, body).

    Tries each route until one is conclusive. A 404 is only returned when the
    body names the model; a route-level 404 falls through to the next route,
    and if every route 404s on the route rather than the model the result is
    reported as inconclusive so the run proceeds.
    """
    last: tuple[int | None, str] = (None, "no route was reachable")
    for route, input_key in _PROBE_ROUTES:
        status, body = _probe_route(base_url, api_key, model, route, input_key)
        if status == 404 and not _model_is_missing(body):
            # This upstream does not serve this route. Says nothing about the
            # model; keep the result only as a fallback and try the next one.
            last = (None, f"{route} is not served by this upstream")
            continue
        return status, body
    return last


def _preflight_models(config) -> None:
    """Refuse to launch when a configured model is not deployed upstream.

    A missing deployment is invisible until the run is over: the gateway
    forwards the request, the upstream 404s, the agent makes no progress, and
    every case is scored 0.0 as an honest-looking task failure. That costs a
    full optimizer trial to discover. This costs one token per model.

    Only a definitive 404 blocks. Anything else (a timeout, a rate limit, a
    503) is inconclusive and must not stop a run that would have succeeded.
    """
    gateway = getattr(config, "inference_gateway", None)
    if gateway is None:
        return
    api_key = os.environ.get(gateway.upstream_api_key_env)
    if not api_key:
        return
    base_url = gateway.default_upstream_base_url
    if gateway.upstream_base_url_env:
        base_url = os.environ.get(gateway.upstream_base_url_env) or base_url

    scopes: dict[str, str] = {}
    for scope_name in ("producer", "evaluation", "finalization"):
        scope = getattr(gateway, scope_name, None)
        if scope is None:
            continue
        for name in scope.allowed_models:
            scopes.setdefault(name, scope_name)

    missing: list[str] = []
    for name, scope_name in scopes.items():
        # A provider prefix is meaningful to a routing proxy and meaningless to
        # a single-provider endpoint, so try the configured name first and only
        # fall back to the bare one. Reporting a model missing on the strength
        # of one spelling would block a run that would have worked.
        candidates = [name]
        if "/" in name:
            candidates.append(name.split("/", 1)[1])
        for candidate in candidates:
            status, body = _probe_model(base_url, api_key, candidate)
            if status != 404:
                break
        if status == 404:
            missing.append(f"{name} ({scope_name} scope): {body.strip()}")
        elif status is not None and status >= 400:
            click.echo(
                f"Preflight: {name} returned HTTP {status}; continuing "
                f"(only a 404 is treated as fatal)"
            )
    if missing:
        raise click.ClickException(
            "these models are not deployed on "
            f"{base_url} and every request to them would fail:\n  - "
            + "\n  - ".join(missing)
            + "\nNote a model can appear in GET /models (the catalogue) and "
            "still have no deployment."
        )


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
    _preflight_models(config)
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
        command.extend(_agent_environment_blanks(task))
        command.extend(_opencode_gateway_args(agent, model, task))
        command.extend(_litellm_base_url_args(agent, task))
        command.extend(_kimi_gateway_args(agent, task))
        # Build-declared outer-trial flags first, so a command-line arg can still
        # override them (harbor's `--ek` takes the last value for a key).
        command.extend(config.optimizer_harbor_args)
        # The derived app name defers to an explicit one from *either* source: a
        # build may declare `--ek app_name=` in optimizer_harbor_args just as a
        # caller may pass it on the command line, and appending ours after the
        # build's would silently win on harbor's last-value-per-key rule.
        command.extend(
            _outer_app_name_args(
                environment,
                config.name,
                (*config.optimizer_harbor_args, *extra),
            )
        )
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
@click.option(
    "--backend", "backend_id", required=True,
    help=(
        "Evaluation backend to score against. Must match --partition: each "
        "partition is served by exactly one backend and asking a different one "
        "is denied. `evals plan` lists the pair for every partition you may run."
    ),
)
@click.option(
    "--evaluation-set", "evaluation_set_name", required=True,
    help="Name of the evaluation set to score on.",
)
@click.option(
    "--partition",
    help="Partition within the evaluation set (e.g. development or validation).",
)
@click.option("--version", help="Candidate version to score; defaults to the agent repository's HEAD commit.")
@click.option(
    "--case-id", "case_ids", multiple=True,
    help="Score only these case ids (repeatable) — a cheap subset for fast iteration. Cannot combine with --start/--stop.",
)
@click.option(
    "--start", type=click.IntRange(min=0),
    help="Subset start index (with --stop): score cases [start, stop) for cheap iteration on a slice.",
)
@click.option("--stop", type=click.IntRange(min=1), help="Subset stop index (exclusive); requires --start.")
@click.option("--parameter", multiple=True, help="Evaluation parameter as NAME=JSON (repeatable).")
@click.option(
    "--timeout", type=click.FloatRange(min=0, min_open=True),
    help="Override the whole-evaluation wall timeout, in seconds.",
)
@click.option(
    "--case-timeout", type=click.FloatRange(min=0, min_open=True),
    help="Override the per-case wall budget for THIS run, in seconds. A case that exceeds it is stopped and scores the failure value; the final held-out evaluation always uses the configured budget, so keep search runs comparable.",
)
@click.option("--max-concurrency", type=click.IntRange(min=1), help="Override how many cases run in parallel.")
@click.option(
    "--error-rate-threshold",
    type=click.FloatRange(min=0, max=1, min_open=True),
    help="Abort the evaluation if the fraction of errored cases exceeds this.",
)
@click.option("--retry-max-attempts", type=click.IntRange(min=1), help="Max attempts per case on transient failure.")
@click.option("--retry-initial-delay", type=click.FloatRange(min=0), help="Initial retry backoff delay, in seconds.")
@click.option("--retry-maximum-delay", type=click.FloatRange(min=0), help="Maximum retry backoff delay, in seconds.")
@click.option("--retry-multiplier", type=click.FloatRange(min=1), help="Retry backoff growth multiplier.")
@click.option(
    "--retry-on-timeout/--no-retry-on-timeout",
    default=None,
    help="Whether a per-case timeout counts as a retryable failure.",
)
@click.option(
    "--seed", type=int,
    help=(
        "Seed for case sampling / evaluation. Backend-dependent: a backend that "
        "fixes its own sampling rejects this with 'invalid evaluation request'. "
        "To replicate elsewhere, re-run the identical case selection."
    ),
)
@click.option(
    "--detach",
    is_flag=True,
    help="Run several evaluations at once: start a durable job and return a job_id immediately instead of blocking. Poll `evals status JOB` until complete, then read `evals result JOB`. Omit to block and get the result in one call.",
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
@click.option(
    "--version",
    help="Candidate version to nominate; defaults to the agent repository's HEAD commit.",
)
def submit_command(version):
    """Nominate your best candidate as the one to ship.

    This is the deliberate selection that finalization scores on the held-out
    set. Submit only a candidate you have confirmed beats the baseline on the
    same cases. If you never submit, VeRO falls back to auto-best over
    validation and then your last commit, so an unvetted edit can ship by
    default -- submit on purpose.
    """
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


@harbor.command("archive-session")
@click.option(
    "--session-dir",
    default=LAYOUT.session_dir,
    show_default=True,
    type=click.Path(path_type=Path, file_okay=False),
)
@click.option(
    "--output",
    default=LAYOUT.session_rescue_archive,
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
def archive_session_command(session_dir, output):
    """Snapshot the session to a tar.gz in place, without finalizing.

    The rescue half of `export-session`. It reads the session directory off the
    admin volume and writes an archive beside it: no admin token, no HTTP call,
    and above all no `/finalize`, so it cannot spend the finalization budget or
    take the 28 minutes the verifier phase took on 2026-07-31. Measured at 3.2s
    on a real 63M / ~2300-file session.

    That cheapness is the point. This runs from a `[[verifier.collect]]` hook,
    which Harbor invokes on *every* terminal outcome (`Trial._recover_outputs`
    runs it even when the trial raised), whereas `export-session` runs only from
    the verifier phase. Harbor swallows just `AgentTimeoutError` and
    `NonZeroAgentExitCodeError` out of the agent phase; anything else skips the
    verifier entirely. Measured: two outer trials died the same night, and the
    one that raised `NonZeroAgentExitCodeError` reached the verifier and left an
    8.8M `session.tar.gz`, while the one that raised a Modal
    `grpclib.StreamTerminatedError` at 71 minutes left nothing at all, losing a
    candidate that had already scored 0.1224 on 49 validation cases.

    The archive is the same format `export-session` produces, minus the
    finalization/status/report files that only exist after a finalize. It still
    carries `candidates/repository.git` (every candidate commit) and
    `database.json` (every evaluation and score).
    """
    archive = create_harbor_session_archive(session_dir, output)
    click.echo(
        json.dumps(
            {"session": str(archive), "sha256": file_sha256(archive)},
            indent=2,
        )
    )


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
