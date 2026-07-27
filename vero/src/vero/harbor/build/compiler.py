"""Compile a trusted VeRO configuration into a runnable Harbor task."""

from __future__ import annotations

import importlib.resources
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath

from vero.evaluation import (
    EvaluationBudget,
    EvaluationLimits,
    EvaluationSet,
    RetryPolicy,
)
from vero.gateway.inference import generate_inference_token, token_digest
from vero.harbor.build.config import HarborBuildConfig
from vero.harbor.build.specs import WorkspaceOverlaySpec
from vero.layout import LAYOUT

logger = logging.getLogger(__name__)

_TEMPLATES = Path(__file__).parent / "templates"
_VERO_COPY = ("pyproject.toml", "README.md", "uv.lock", "src")

SESSION_ID = "trial"
UPSTREAM_API_KEY_ENV = LAYOUT.gateway_upstream_api_key_env
UPSTREAM_BASE_URL_ENV = LAYOUT.gateway_upstream_base_url_env
PRODUCER_BASE_URL = LAYOUT.scope_url("producer", LAYOUT.optimizer_attribution)

# Credentials the compose template routes through the gateway by setting them
# explicitly, instead of blanking them like every other declared secret. The two
# halves are one invariant: a name here must be set below, and a name set below
# must be here, or the rendered compose emits the key twice.
GATEWAY_ROUTED_CREDENTIALS = frozenset(LAYOUT.routed_credential_envs)

# Container paths and service identities come from the layout, never from a
# literal here: the templates read the same object, so the two cannot drift.
VERO_DIR = LAYOUT.vero
TRUSTED_REPO = LAYOUT.trusted_repo
AGENT_REPO = LAYOUT.target_repo
CASES_DIR = LAYOUT.cases
TASK_SOURCE_DIR = LAYOUT.task_source
HARNESS_DIR = LAYOUT.harness
SERVE_CONFIG = LAYOUT.serve_config
AGENT_VOLUME = LAYOUT.agent_volume
ADMIN_VOLUME = LAYOUT.admin_volume
SESSION_DIR = LAYOUT.session_dir
TOKEN_PATH = LAYOUT.token_path
INFERENCE_STATE = LAYOUT.inference_state
INFERENCE_REQUEST_LOG_DIR = LAYOUT.inference_request_log_dir
INFERENCE_GATEWAY_URL = LAYOUT.gateway_url

# How long finalization waits for already-running agent evaluations before
# cancelling them. A grace period, not a ceiling: expiry is graceful (the
# evaluator's cancellation path persists terminal records and refunds budgets),
# so waiting longer buys nothing and only delays the held-out score. Matches
# harbor/deployment.py's own default.
DEFAULT_EVALUATION_DRAIN_SECONDS = 600.0


def _backend_id(partition: str) -> str:
    return f"harbor-{partition}"


def _is_vero_source(vero_root: Path) -> bool:
    return (vero_root / "pyproject.toml").is_file() and (
        vero_root / "src/vero"
    ).is_dir()


def _copy_vero_source(vero_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in _VERO_COPY:
        source = vero_root / name
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination / name)


def _rewrite_vero_path(pyproject: Path) -> None:
    if not pyproject.exists():
        return
    original = pyproject.read_text(encoding="utf-8")
    rewritten = re.sub(
        r'(scale-vero\s*=\s*\{[^}]*?path\s*=\s*")[^"]*(")',
        rf"\g<1>{VERO_DIR}\g<2>",
        original,
    )
    if rewritten != original:
        pyproject.write_text(rewritten, encoding="utf-8")


def _safe_extract_tar(payload: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe path in Git archive: {member.name!r}")
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                if link.is_absolute() or ".." in link.parts:
                    raise ValueError(f"unsafe link in Git archive: {member.linkname!r}")
        # filter="data" strips device files / setuid bits and neutralizes unsafe
        # links, matching extract_harbor_session_archive's defensive posture.
        archive.extractall(destination, filter="data")


def _prepare_baseline_repo(
    source: Path,
    destination: Path,
    *,
    rewrite_vero_path: bool,
) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    repository = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if repository.returncode == 0:
        root = Path(repository.stdout.strip())
        relative = source.relative_to(root)
        treeish = "HEAD" if str(relative) == "." else f"HEAD:{relative.as_posix()}"
        archived = subprocess.run(
            ["git", "-C", str(root), "archive", "--format=tar", treeish],
            capture_output=True,
        )
        if archived.returncode != 0:
            raise RuntimeError(
                "git archive failed: "
                + archived.stderr.decode("utf-8", errors="replace").strip()
            )
        _safe_extract_tar(archived.stdout, destination)
    else:
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git"),
        )
    if rewrite_vero_path:
        _rewrite_vero_path(destination / "pyproject.toml")
    if (destination / ".evals").exists():
        raise ValueError("agent baseline contains reserved path '.evals'")

    def git(*arguments: str) -> str:
        result = subprocess.run(
            [
                "git",
                "-c",
                "user.name=vero",
                "-c",
                "user.email=vero@localhost",
                "-C",
                str(destination),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "baseline")
    return git("rev-parse", "HEAD")


def _local_result_task_name(task_source: Path, selector: str) -> str:
    root = task_source.resolve()
    task_dir = (root / selector).resolve()
    if task_dir.parent != root:
        raise ValueError(
            f"local Harbor task selector {selector!r} must name a direct child directory"
        )
    config_path = task_dir / "task.toml"
    if not config_path.is_file():
        raise ValueError(f"local Harbor task selector {selector!r} has no task.toml")
    try:
        value = tomllib.loads(config_path.read_text(encoding="utf-8"))
        task_name = value["task"]["name"]
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            f"local Harbor task selector {selector!r} has no canonical task.name"
        ) from error
    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError(
            f"local Harbor task selector {selector!r} has no canonical task.name"
        )
    return task_name


def _write_cases(config: HarborBuildConfig, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    # Only a harbor backend names a task_source, and only then does a case id
    # correspond to a local Harbor task directory with a canonical name. A
    # command backend's ids name whatever its harness understands.
    task_source = Path(config.task_source) if config.task_source else None
    local = task_source is not None and task_source.exists()
    for partition, tasks in config.partitions.items():
        path = destination / f"{partition}.jsonl"
        lines = [
            json.dumps(
                {
                    "id": task,
                    "task_name": task,
                    **(
                        {
                            "result_task_name": _local_result_task_name(
                                task_source,
                                task,
                            )
                        }
                        if local
                        else {}
                    ),
                },
                ensure_ascii=False,
            )
            for task in tasks
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _deployment_config(
    config: HarborBuildConfig,
    *,
    baseline_version: str,
    local_task_source: bool,
    evaluation_inference_token: str | None,
    finalization_inference_token: str | None,
) -> dict:
    task_source = TASK_SOURCE_DIR if local_task_source else config.task_source
    backends = {}
    if config.command_backend is not None:
        # A command backend scores by running a program, so it needs none of the
        # nested-`harbor run` plumbing: no task source, no agent import path, no
        # model. The harness is baked into the sidecar at HARNESS_DIR, and case
        # enumeration is the harness's job — it reads the partition's case file
        # from VERO_CASES_DIR, since the backend itself derives a case count from
        # the requested selection rather than from a cases file.
        specification = config.command_backend
        for partition in config.partitions:
            backends[_backend_id(partition)] = {
                "type": "command",
                "harness_root": HARNESS_DIR,
                "command": list(specification.command),
                "working_directory": specification.working_directory,
                "environment": {
                    "VERO_CASES_DIR": CASES_DIR,
                    **specification.environment,
                },
                "passthrough_environment": list(
                    dict.fromkeys(
                        [*specification.passthrough_environment, *config.secrets]
                    )
                ),
                "staged_inputs": dict(specification.staged_inputs),
                "agent_context_inputs": {
                    name: list(paths)
                    for name, paths in specification.agent_context_inputs.items()
                },
            }
    else:
        # A target may override attempts-per-case on its own (held-out) partition
        # without touching search/selection. Backends are per-partition, so the
        # override is injected into that partition's backend at compile time; a
        # config validator forbids overriding an agent-evaluable partition.
        target_n_attempts = {
            target.partition: target.n_attempts
            for target in config.targets
            if target.n_attempts is not None
        }
        target_aggregate = {
            target.partition: target.aggregate_attempts
            for target in config.targets
            if target.aggregate_attempts is not None
        }
        for partition in config.partitions:
            backends[_backend_id(partition)] = {
                "type": "harbor",
                "task_source": task_source,
                "agent_import_path": config.agent_import_path,
                "cases_path": f"{CASES_DIR}/{partition}.jsonl",
                "harbor_requirement": config.harbor_requirement,
                "evaluation_set_name": config.evaluation_set_name,
                "partition": partition,
                "model": config.model,
                "environment_name": config.environment_name,
                "python_version": config.harbor_python_version,
                "case_timeout_seconds": config.case_timeout_seconds,
                "task_agent_timeout_seconds": config.task_agent_timeout_seconds,
                "default_index": config.default_index,
                "n_attempts": target_n_attempts.get(partition, config.n_attempts),
                "max_retries": config.max_retries,
                "retry_wait_multiplier": config.retry_wait_multiplier,
                "retry_min_wait_seconds": config.retry_min_wait_seconds,
                "retry_max_wait_seconds": config.retry_max_wait_seconds,
                "infrastructure_max_attempts": config.infrastructure_max_attempts,
                "infrastructure_retry_delay_seconds": (
                    config.infrastructure_retry_delay_seconds
                ),
                "reward_key": config.reward_key,
                "aggregate_attempts": target_aggregate.get(
                    partition, config.aggregate_attempts
                ),
                "feedback_transcripts": config.feedback_transcripts,
                "feedback_max_bytes": config.feedback_max_bytes,
                "expose_attempt_detail": config.expose_attempt_detail,
                "passthrough_environment": config.secrets,
                "environment": config.task_environment,
                "inference_gateway_url": (
                    INFERENCE_GATEWAY_URL
                    if config.inference_gateway is not None
                    else None
                ),
                "inference_gateway_token": evaluation_inference_token,
                "inference_gateway_finalization_token": finalization_inference_token,
                "harness_user": config.harness_user,
                "task_services_use_upstream": config.task_services_use_upstream,
                "upstream_api_key_env": (
                    UPSTREAM_API_KEY_ENV
                    if config.inference_gateway is not None
                    else None
                ),
                "upstream_base_url_env": (
                    UPSTREAM_BASE_URL_ENV
                    if config.inference_gateway is not None
                    and config.inference_gateway.upstream_base_url_env is not None
                    else None
                ),
                "case_resources_cache_path": (
                    f"{LAYOUT.case_resources_dir}/{partition}"
                ),
                "inference_usage_path": (
                    INFERENCE_STATE if config.inference_gateway is not None else None
                ),
                "extra_args": config.extra_harbor_args,
            }

    limits = EvaluationLimits(
        timeout_seconds=config.timeout_seconds,
        case_timeout_seconds=config.case_timeout_seconds,
        max_concurrency=config.max_concurrency,
        error_rate_threshold=config.error_rate_threshold,
        retry=RetryPolicy.disabled(),
    )
    policies = []
    budgets = []
    for access in config.agent_access:
        backend_id = _backend_id(access.partition)
        evaluation_set = EvaluationSet(
            name=config.evaluation_set_name,
            partition=access.partition,
        )
        policies.append(
            {
                "backend_id": backend_id,
                "evaluation_set_name": config.evaluation_set_name,
                "partition": access.partition,
                "objective": config.objective.model_dump(mode="json"),
                "access": access.to_access_policy().model_dump(mode="json"),
                "parameters": {},
                "allowed_parameters": [],
                "limits": limits.model_dump(mode="json"),
            }
        )
        if access.total_runs is not None or access.total_cases is not None:
            budgets.append(
                EvaluationBudget(
                    backend_id=backend_id,
                    evaluation_set_key=evaluation_set.budget_key(backend_id),
                    total_runs=access.total_runs,
                    total_cases=access.total_cases,
                ).model_dump(mode="json")
            )

    selection_backend = _backend_id(config.selection_partition)
    selection_set = EvaluationSet(
        name=config.evaluation_set_name,
        partition=config.selection_partition,
    )
    targets = []
    for target in config.targets:
        parameters = dict(target.parameters)
        if target.model is not None:
            parameters["harbor_model_override"] = target.model
        targets.append(
            {
                "reward_key": target.reward_key,
                "backend_id": _backend_id(target.partition),
                "evaluation_set": EvaluationSet(
                    name=config.evaluation_set_name,
                    partition=target.partition,
                ).model_dump(mode="json"),
                "objective": config.objective.model_dump(mode="json"),
                "parameters": parameters,
                "limits": limits.model_dump(mode="json"),
                "failure_value": target.failure_value,
                "reward_scale": (
                    1.0 if config.objective.direction == "maximize" else -1.0
                ),
                "baseline_reward": target.baseline_reward,
                "max_attempts": target.max_attempts,
            }
        )
    return {
        "task_name": config.name,
        "task_description": config.description,
        "repo_path": TRUSTED_REPO,
        "agent_repo_path": AGENT_REPO,
        "session_dir": SESSION_DIR,
        "session_id": SESSION_ID,
        "backends": backends,
        "access_policies": policies,
        "budgets": budgets,
        "selection": {
            "mode": config.reward_mode,
            # Always populated so auto_best can serve as the fallback when a
            # submit-mode run has no submission.
            "backend_id": selection_backend,
            "evaluation_set": selection_set.model_dump(mode="json"),
            "objective": config.objective.model_dump(mode="json"),
            "baseline_version": baseline_version,
            "parameters": {},
            "limits": limits.model_dump(mode="json"),
            "rescore_top_k": config.rescore_top_k,
            "rescore_attempts": config.rescore_attempts,
            "baseline_floor": config.baseline_floor,
            "baseline_selection_score": config.baseline_selection_score,
            "selection_coverage_threshold": config.selection_coverage_threshold,
        },
        "targets": targets,
        "agent_volume": AGENT_VOLUME,
        "admin_volume": ADMIN_VOLUME,
        "submit_enabled": config.reward_mode == "submit",
        "disclose_budget": config.disclose_budget,
        "score_baseline": config.score_baseline,
        "wandb": (
            config.wandb.model_dump(mode="json") if config.wandb is not None else None
        ),
        # Must NOT inherit timeout_seconds: that is deliberately sized to be
        # unreachable (every trial hitting its per-case cap), so inheriting it
        # turns one hung sub-run into a finalization stall of the same order.
        # officeqa run #4 sat 6h in exactly that state -- its agent evaluation
        # had finished writing results two hours earlier, but the subprocess
        # never exited and the drain was waiting out an inherited 21600s.
        "evaluation_drain_timeout_seconds": (
            config.evaluation_drain_timeout_seconds
            or DEFAULT_EVALUATION_DRAIN_SECONDS
        ),
        "inference_usage_path": (
            INFERENCE_STATE if config.inference_gateway is not None else None
        ),
        "inference_request_log_dir": (
            INFERENCE_REQUEST_LOG_DIR
            if config.inference_gateway is not None
            and config.inference_gateway.log_requests
            else None
        ),
        "inference_limits": (
            {
                "producer": config.inference_gateway.producer.model_dump(mode="json"),
                "evaluation": config.inference_gateway.evaluation.model_dump(
                    mode="json"
                ),
            }
            if config.inference_gateway is not None
            else {}
        ),
    }


def _render(template: str, destination: Path, **context) -> None:
    """Render one template. Every template gets the layout, so no template needs
    to spell out a container path, service name, or port of its own."""
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as error:
        raise RuntimeError(
            "install scale-vero[harbor] to compile Harbor tasks"
        ) from error
    environment = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        environment.get_template(template).render(layout=LAYOUT, **context),
        encoding="utf-8",
    )


def compile_harbor_task(
    config: HarborBuildConfig,
    output_dir: Path | str,
    *,
    vero_root: Path | None = None,
) -> Path:
    """Emit a self-contained Harbor task directory from validated config."""
    output = Path(output_dir).expanduser().resolve()
    source_root = (vero_root or Path(__file__).parents[4]).resolve()
    use_local_vero = _is_vero_source(source_root)
    if vero_root is not None and not use_local_vero:
        raise ValueError(f"vero_root {source_root} is not a scale-vero source checkout")
    protected = [Path(config.agent_repo).resolve()]
    if use_local_vero:
        protected.append(source_root)
    if config.task_source is not None:
        task_source_path = Path(config.task_source)
        if task_source_path.exists():
            protected.append(task_source_path.resolve())
    for path in protected:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(
                f"output directory {output} overlaps protected source {path}"
            )
    # Imported here, not at module scope: deployment pulls in the whole runtime
    # stack, and harbor/__init__ imports this package before it, so a top-level
    # import would only work by accident of partial-initialization ordering.
    from vero.harbor.deployment import FACTORY_PATH

    gateway_environment: list[str] = []
    credential_sources: list[str] = []
    if config.inference_gateway is not None:
        gateway_environment.append(UPSTREAM_API_KEY_ENV)
        credential_sources.append(config.inference_gateway.upstream_api_key_env)
        if config.inference_gateway.upstream_base_url_env is not None:
            gateway_environment.append(UPSTREAM_BASE_URL_ENV)
            credential_sources.append(config.inference_gateway.upstream_base_url_env)
    task_environment = list(dict.fromkeys([*config.secrets, *gateway_environment]))
    if os.environ.get("VERO_SKIP_SECRET_CHECK") is None:
        required_sources = list(dict.fromkeys([*config.secrets, *credential_sources]))
        missing = [name for name in required_sources if not os.environ.get(name)]
        if missing:
            raise ValueError(
                "declared task credentials are missing: " + ", ".join(missing)
            )
    if output.exists():
        shutil.rmtree(output)
    environment_dir = output / "environment"
    sidecar_dir = environment_dir / "sidecar"
    gateway_dir = environment_dir / "gateway"
    environment_dir.mkdir(parents=True)
    if use_local_vero:
        _copy_vero_source(source_root, environment_dir / "vero")

    baseline = _prepare_baseline_repo(
        Path(config.agent_repo),
        environment_dir / "agent-baseline",
        rewrite_vero_path=use_local_vero,
    )
    shutil.copytree(
        environment_dir / "agent-baseline",
        environment_dir / "agent-seed",
    )
    # General filesystem overlay: bake arbitrary host files/dirs into the agent
    # workspace at build time. A source dir's *contents* land under dest; a source
    # file lands as dest/<name>. dest="." is the workspace root.
    overlays = list(config.workspace_overlays)
    if config.include_evals_skill:
        # vero's own packaged skill; resolved from the installed package so the
        # baked copy always matches the vero (and `evals` CLI) in the image.
        skill_source = importlib.resources.files("vero") / "skills" / "evals"
        overlays.append(
            WorkspaceOverlaySpec(source=str(skill_source), dest="skills/evals")
        )
    overlay_present = bool(overlays)
    overlay_excludes: list[str] = []
    if overlay_present:
        overlay_root = environment_dir / "overlay"
        for spec in overlays:
            source = Path(spec.source)
            target = overlay_root if spec.dest == "." else overlay_root / spec.dest
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target / source.name)
            if spec.dest != ".":
                top = spec.dest.split("/", 1)[0]
                if top not in overlay_excludes:
                    overlay_excludes.append(top)
    _write_cases(config, sidecar_dir / "cases")
    if config.command_backend is not None:
        # Bake the scoring program into the trusted sidecar, alongside the cases
        # it enumerates. Never reachable from the agent workspace.
        shutil.copytree(
            Path(config.command_backend.harness_source),
            sidecar_dir / "harness",
        )
    local_task_source = (
        config.task_source is not None and Path(config.task_source).exists()
    )
    if local_task_source:
        shutil.copytree(
            Path(config.task_source),
            sidecar_dir / "task-source",
        )
    producer_inference_token = (
        generate_inference_token() if config.inference_gateway is not None else None
    )
    evaluation_inference_token = (
        generate_inference_token() if config.inference_gateway is not None else None
    )
    finalization_inference_token = (
        generate_inference_token() if config.inference_gateway is not None else None
    )
    deployment = _deployment_config(
        config,
        baseline_version=baseline,
        local_task_source=local_task_source,
        evaluation_inference_token=evaluation_inference_token,
        finalization_inference_token=finalization_inference_token,
    )
    (sidecar_dir / "serve.json").write_text(
        json.dumps(deployment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if config.inference_gateway is not None:
        assert producer_inference_token is not None
        assert evaluation_inference_token is not None
        assert finalization_inference_token is not None
        # Finalization reserves its own budget; default to the evaluation policy.
        finalization_spec = (
            config.inference_gateway.finalization or config.inference_gateway.evaluation
        )
        gateway_dir.mkdir(parents=True, exist_ok=True)
        gateway_config = {
            "upstream_api_key_env": UPSTREAM_API_KEY_ENV,
            "upstream_base_url_env": (
                UPSTREAM_BASE_URL_ENV
                if config.inference_gateway.upstream_base_url_env is not None
                else None
            ),
            "default_upstream_base_url": (
                config.inference_gateway.default_upstream_base_url
            ),
            "state_path": INFERENCE_STATE,
            "request_log": (
                {
                    "directory": INFERENCE_REQUEST_LOG_DIR,
                    "body_bytes": config.inference_gateway.request_log_body_bytes,
                    "attribution": config.inference_gateway.request_log_attribution,
                }
                if config.inference_gateway.log_requests
                else None
            ),
            "scopes": {
                "producer": {
                    "token_sha256": token_digest(producer_inference_token),
                    **config.inference_gateway.producer.model_dump(mode="json"),
                },
                "evaluation": {
                    "token_sha256": token_digest(evaluation_inference_token),
                    **config.inference_gateway.evaluation.model_dump(mode="json"),
                },
                "finalization": {
                    "token_sha256": token_digest(finalization_inference_token),
                    **finalization_spec.model_dump(mode="json"),
                },
            },
        }
        (gateway_dir / "config.json").write_text(
            json.dumps(gateway_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (gateway_dir / "launch.json").write_text(
            json.dumps(
                {
                    "upstream_api_key_source": (
                        config.inference_gateway.upstream_api_key_env
                    ),
                    "upstream_api_key_target": UPSTREAM_API_KEY_ENV,
                    "upstream_base_url_source": (
                        config.inference_gateway.upstream_base_url_env
                    ),
                    "upstream_base_url_target": UPSTREAM_BASE_URL_ENV,
                    "producer_api_key": producer_inference_token,
                    "producer_base_url": (PRODUCER_BASE_URL),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    selection_access = next(
        access
        for access in config.agent_access
        if access.partition == config.selection_partition
    )
    context = {
        "name_toml": json.dumps(config.name, ensure_ascii=False),
        "description_toml": json.dumps(config.description, ensure_ascii=False),
        "description": config.description,
        "base_image_main": config.base_image_main,
        "base_image_sidecar": config.base_image_sidecar,
        "use_local_vero": use_local_vero,
        # The trusted sidecar needs the wandb extra when W&B reporting is on.
        "sidecar_extras": "harbor,wandb" if config.wandb is not None else "harbor",
        "vero_requirement": (
            None
            if use_local_vero
            else (
                "scale-vero["
                + ("harbor,wandb" if config.wandb is not None else "harbor")
                + f"]=={distribution_version('scale-vero')}"
            )
        ),
        "harbor_requirement": config.harbor_requirement,
        "secrets": task_environment,
        "sidecar_secrets": config.secrets,
        "inference_gateway": config.inference_gateway,
        "gateway_environment": gateway_environment,
        # Names the main container blanks. Default-deny: every declared secret is
        # blanked for the optimizer unless it appears in GATEWAY_ROUTED_CREDENTIALS,
        # which the compose template sets explicitly just below the blanking loop
        # (to the producer token and the gateway URL). The exclusion exists to
        # avoid emitting the same key twice, not to permit anything: adding a new
        # credential to `secrets` gets it blanked automatically.
        "scrubbed_main_environment": [
            name for name in task_environment if name not in GATEWAY_ROUTED_CREDENTIALS
        ],
        "producer_inference_token": producer_inference_token,
        "evaluation_inference_token": evaluation_inference_token,
        "inference_gateway_url": INFERENCE_GATEWAY_URL,
        "read_only_paths": config.read_only_paths,
        "local_task_source": local_task_source,
        "sidecar_factory": FACTORY_PATH,
        "producer_base_url": PRODUCER_BASE_URL,
        "command_harness": config.command_backend is not None,
        # The Harbor backend hard-rejects request.seed, so only advertise the
        # flag when the build evaluates through a command backend.
        "seed_supported": config.command_backend is not None,
        "selection_backend": _backend_id(config.selection_partition),
        "evaluation_set_name": config.evaluation_set_name,
        "selection_partition": config.selection_partition,
        "submit_enabled": config.reward_mode == "submit",
        "task_services_use_upstream": config.task_services_use_upstream,
        "multifidelity": config.instruct_multifidelity,
        "minimum_subset_cases": (
            selection_access.min_aggregate_cases
            if selection_access.disclosure == "aggregate"
            else 1
        ),
        "exposed_partitions": [
            access.partition
            for access in config.agent_access
            if access.expose_case_resources
        ],
        "exhaust_budget": config.instruct_exhaust_budget,
        "disclose_budget": config.disclose_budget,
        "build_timeout": config.build_timeout_seconds,
        "verifier_timeout": (
            config.verifier_timeout_seconds or max(1, int(config.timeout_seconds))
        ),
        "overlay_present": overlay_present,
        "overlay_excludes": overlay_excludes,
    }
    _render("task.toml.j2", output / "task.toml", **context)
    _render("instruction.md.j2", output / "instruction.md", **context)
    _render("Dockerfile.main.j2", environment_dir / "Dockerfile", **context)
    _render(
        "Dockerfile.sidecar.j2",
        sidecar_dir / "Dockerfile",
        **context,
    )
    if config.inference_gateway is not None:
        _render(
            "Dockerfile.gateway.j2",
            gateway_dir / "Dockerfile",
            **context,
        )
    _render(
        "docker-compose.yaml.j2",
        environment_dir / "docker-compose.yaml",
        **context,
    )
    _render("seed.sh.j2", environment_dir / "main/seed.sh", **context)
    _render("test.sh.j2", output / "tests/test.sh", **context)
    _render("solve.sh.j2", output / "solution/solve.sh", **context)
    for script in (
        environment_dir / "main/seed.sh",
        output / "tests/test.sh",
        output / "solution/solve.sh",
    ):
        script.chmod(0o755)
    logger.info("Compiled Harbor task at %s from baseline %s", output, baseline)
    return output
