from __future__ import annotations

import json
import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vero.harbor import (
    AgentAccessSpec,
    CommandBackendSpec,
    HarborBuildConfig,
    HarborDeploymentConfig,
    InferenceBudgetSpec,
    InferenceGatewaySpec,
    VerificationTargetSpec,
    WandbSpec,
    WorkspaceOverlaySpec,
    compile_harbor_task,
    load_harbor_build_config,
)
from vero.harbor.build.compiler import GATEWAY_ROUTED_CREDENTIALS
from vero.harbor.build.config import (
    _HARBOR_ONLY_FIELDS,
    _AgentWorkspaceFields,
    _EvaluationLimitFields,
    _HarborEvaluationFields,
    _SearchAndSelectionFields,
    _TaskEnvironmentFields,
    _TaskIdentityFields,
)
from vero.harbor.deployment import FACTORY_PATH
from vero.layout import LAYOUT
from vero.sidecar.serve import load_factory

_OMIT = object()


def test_task_layout_values_are_pinned():
    """The one place the layout's literal values are written down twice.

    Every other test references LAYOUT, so without this they would all be
    tautological: changing a path would silently keep them green. These values
    are a contract with the benchmarks' read_only_paths and with the compiled
    task directories that are checked in, so a deliberate change should have to
    edit this list too.
    """
    assert LAYOUT.target_repo == "/work/agent"
    assert LAYOUT.trusted_repo == "/opt/agent-baseline"
    assert LAYOUT.seed_repo == "/opt/agent-seed"
    assert LAYOUT.vero == "/opt/vero"
    assert LAYOUT.cases == "/opt/cases"
    assert LAYOUT.task_source == "/opt/task-source"
    assert LAYOUT.harness == "/opt/harness"
    assert LAYOUT.overlay == "/opt/overlay"
    assert LAYOUT.serve_config == "/opt/serve.json"
    assert LAYOUT.seed_script == "/opt/seed.sh"
    assert LAYOUT.inference_config == "/opt/inference.json"
    assert LAYOUT.agent_volume == "/state/agent-context"
    assert LAYOUT.admin_volume == "/state/admin"
    assert LAYOUT.token_dir == "/state/token"
    assert LAYOUT.inference_dir == "/state/inference"
    assert LAYOUT.sidecar_host == "eval-sidecar"
    assert LAYOUT.sidecar_port == 8000
    assert LAYOUT.gateway_host == "inference-gateway"
    assert LAYOUT.gateway_port == 8001
    # Derived paths, so a base and its children cannot drift apart.
    assert LAYOUT.session_dir == "/state/admin/session"
    assert LAYOUT.case_resources_dir == "/state/admin/case-resources"
    assert LAYOUT.token_path == "/state/token/admin.token"
    assert LAYOUT.inference_state == "/state/inference/usage.json"
    assert LAYOUT.inference_request_log_dir == "/state/inference/requests"
    assert LAYOUT.target_git == "/work/agent/.git"
    assert LAYOUT.target_git_exclude == "/work/agent/.git/info/exclude"
    assert LAYOUT.target_evals == "/work/agent/.evals"
    assert LAYOUT.sidecar_url == "http://eval-sidecar:8000"
    assert LAYOUT.gateway_url == "http://inference-gateway:8001"


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_build_param_substitution_semantics():
    from vero.harbor.build.loader import _substitute_build_param as sub

    context = {"A": "x"}
    assert sub("${A}", context) == "x"
    assert sub("${MISSING:-fallback}", context) == "fallback"
    assert sub("pre-${A}-post", context) == "pre-x-post"
    with pytest.raises(ValueError, match="required build parameter 'MISSING'"):
        sub("${MISSING:?please set it}", context)
    with pytest.raises(ValueError, match="build parameter 'MISSING' is unset"):
        sub("${MISSING}", context)


def _target_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "VeRO Test")
    _git(path, "config", "user.email", "vero@example.test")
    (path / "README.md").write_text("# Target\n", encoding="utf-8")
    (path / "pyproject.toml").write_text(
        '[project]\nname="target"\nversion="0.1.0"\n',
        encoding="utf-8",
    )
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "target baseline")
    return path


def _config(tmp_path: Path, **updates) -> HarborBuildConfig:
    target = _target_repo(tmp_path / "target")
    task_source = tmp_path / "protected-tasks"
    task_source.mkdir()
    task_names = ["task-a", "task-b", "task-c", "task-d", "task-e", "task-hidden"]
    for task_name in task_names:
        task = task_source / task_name
        task.mkdir()
        (task / "task.toml").write_text(
            f'[task]\nname="org/{task_name}"\n',
            encoding="utf-8",
        )
    values = {
        "name": 'org/optimize-"program"',
        "description": "Improve the program",
        "agent_repo": str(target),
        "task_source": str(task_source),
        "agent_import_path": "target.agent:Agent",
        "harbor_requirement": "harbor==0.1.17",
        "partitions": {
            "validation": ["task-a", "task-b", "task-c", "task-d", "task-e"],
            "test": ["task-hidden"],
        },
        "agent_access": [
            AgentAccessSpec(
                partition="validation",
                expose_case_resources=True,
                total_runs=5,
                total_cases=25,
            )
        ],
        "selection_partition": "validation",
        "targets": [VerificationTargetSpec(partition="test")],
    }
    values.update(updates)
    # Pass _OMIT to leave a key out entirely, which is different from passing
    # None: the build config reports Harbor-only keys that were set at all.
    return HarborBuildConfig(**{k: v for k, v in values.items() if v is not _OMIT})


def test_compiler_bakes_workspace_overlay_into_agent_environment(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / ".claude" / "agents").mkdir(parents=True)
    (bundle / ".claude" / "agents" / "insights.md").write_text(
        "# a\n", encoding="utf-8"
    )
    (bundle / "skills" / "insights").mkdir(parents=True)
    (bundle / "skills" / "insights" / "SKILL.md").write_text("# s\n", encoding="utf-8")

    config = _config(
        tmp_path,
        workspace_overlays=[
            WorkspaceOverlaySpec(source=str(bundle / ".claude"), dest=".claude"),
            WorkspaceOverlaySpec(source=str(bundle / "skills"), dest="skills"),
        ],
    )
    output = compile_harbor_task(config, tmp_path / "compiled")
    env = output / "environment"

    # staged into the build context under environment/overlay/<dest>
    assert (env / "overlay" / ".claude" / "agents" / "insights.md").is_file()
    assert (env / "overlay" / "skills" / "insights" / "SKILL.md").is_file()
    # Dockerfile copies it, seed applies it, injected tooling is git-excluded
    assert f"COPY overlay {LAYOUT.overlay}" in (env / "Dockerfile").read_text()
    seed = (env / "main" / "seed.sh").read_text()
    assert f"cp -a {LAYOUT.overlay}/. {LAYOUT.target_repo}/" in seed
    assert "/.claude/" in seed and "/skills/" in seed


def test_compiler_plumbs_wandb_into_serve_config(tmp_path):
    output = compile_harbor_task(
        _config(tmp_path, wandb=WandbSpec(project="vero-bench", group="gaia")),
        tmp_path / "compiled",
    )
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    assert serve["wandb"]["project"] == "vero-bench"
    assert serve["wandb"]["group"] == "gaia"


def test_compiler_omits_wandb_when_unset(tmp_path):
    output = compile_harbor_task(_config(tmp_path), tmp_path / "compiled")
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    assert serve["wandb"] is None


def test_compiler_omits_overlay_when_unset(tmp_path):
    output = compile_harbor_task(
        _config(tmp_path, include_evals_skill=False), tmp_path / "compiled"
    )
    env = output / "environment"
    assert not (env / "overlay").exists()
    assert "COPY overlay" not in (env / "Dockerfile").read_text()
    assert LAYOUT.overlay not in (env / "main" / "seed.sh").read_text()


def test_compiler_bakes_evals_skill_by_default(tmp_path):
    output = compile_harbor_task(_config(tmp_path), tmp_path / "compiled")
    env = output / "environment"
    skill = env / "overlay" / "skills" / "evals" / "SKILL.md"
    assert skill.is_file()
    assert "evals run" in skill.read_text(encoding="utf-8")
    assert f"'/skills/' >> {LAYOUT.target_git_exclude}" in (
        env / "main" / "seed.sh"
    ).read_text(encoding="utf-8")


def test_compiler_budget_disclosure_toggle(tmp_path):
    disclosed = compile_harbor_task(_config(tmp_path / "on"), tmp_path / "on/out")
    instruction_on = (disclosed / "instruction.md").read_text(encoding="utf-8")
    serve_on = json.loads(
        (disclosed / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    assert "budget" in instruction_on.lower()
    assert serve_on["disclose_budget"] is True

    blind = compile_harbor_task(
        _config(tmp_path / "off", disclose_budget=False), tmp_path / "off/out"
    )
    instruction_off = (blind / "instruction.md").read_text(encoding="utf-8")
    serve_off = json.loads(
        (blind / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    assert "budget" not in instruction_off.lower()
    assert serve_off["disclose_budget"] is False


def test_compiler_plumbs_task_services_use_upstream(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_UPSTREAM_KEY", "real-provider-secret")
    monkeypatch.setenv("TEST_UPSTREAM_URL", "https://provider.example/v1")
    monkeypatch.setenv("TEST_MODAL_TOKEN", "modal-secret")
    config = _config(
        tmp_path,
        secrets=["TEST_MODAL_TOKEN"],
        task_services_use_upstream=True,
        # Task-owned upstream services are incompatible with harness isolation
        # (the key would reach the harness env), so this build opts out.
        harness_user=None,
        task_environment={"TAU2_USER_MODEL": "openai/gpt-target"},
        inference_gateway=InferenceGatewaySpec(
            upstream_api_key_env="TEST_UPSTREAM_KEY",
            upstream_base_url_env="TEST_UPSTREAM_URL",
            producer=InferenceBudgetSpec(allowed_models=["gpt-producer"]),
            evaluation=InferenceBudgetSpec(allowed_models=["gpt-target"]),
        ),
    )
    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    backend = next(iter(serve["backends"].values()))
    assert backend["task_services_use_upstream"] is True
    assert backend["upstream_api_key_env"] == "VERO_INFERENCE_UPSTREAM_API_KEY"
    assert backend["upstream_base_url_env"] == "VERO_INFERENCE_UPSTREAM_BASE_URL"
    assert backend["environment"] == {"TAU2_USER_MODEL": "openai/gpt-target"}
    # validates against the deployment schema (backend config accepts the flags)
    for partition, b in serve["backends"].items():
        b["cases_path"] = str(
            output
            / f"environment/sidecar/cases/{partition.removeprefix('harbor-')}.jsonl"
        )
    HarborDeploymentConfig.model_validate(serve)
    # the trusted eval-sidecar receives the real upstream (main keeps it scrubbed)
    compose = yaml.safe_load((output / "environment/docker-compose.yaml").read_text())
    sidecar_env = compose["services"]["eval-sidecar"]["environment"]
    assert "VERO_INFERENCE_UPSTREAM_API_KEY" in sidecar_env
    assert "VERO_INFERENCE_UPSTREAM_BASE_URL" in sidecar_env
    assert (
        compose["services"]["main"]["environment"]["VERO_INFERENCE_UPSTREAM_API_KEY"]
        == ""
    )


def test_compiler_reserves_finalization_scope_defaulting_to_evaluation(tmp_path):
    config = _config(
        tmp_path,
        inference_gateway=InferenceGatewaySpec(
            producer=InferenceBudgetSpec(allowed_models=["gpt-producer"]),
            evaluation=InferenceBudgetSpec(
                allowed_models=["gpt-target"], max_requests=15000, max_tokens=100000000
            ),
        ),
    )
    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )
    gateway = json.loads(
        (output / "environment/gateway/config.json").read_text(encoding="utf-8")
    )
    scopes = gateway["scopes"]
    # a reserved finalization scope exists, defaulting to the evaluation policy
    assert "finalization" in scopes
    assert scopes["finalization"]["allowed_models"] == ["gpt-target"]
    assert scopes["finalization"]["max_tokens"] == 100000000
    # its token is distinct from the evaluation token (optimizer can't drain it)
    assert (
        scopes["finalization"]["token_sha256"] != scopes["evaluation"]["token_sha256"]
    )
    # and the sidecar backend carries a finalization token distinct from the eval one
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    backend = next(iter(serve["backends"].values()))
    assert backend["inference_gateway_finalization_token"] is not None
    assert (
        backend["inference_gateway_finalization_token"]
        != backend["inference_gateway_token"]
    )
    # Per-request logging is on by default: the gateway captures every
    # request/response on its state volume and the sidecar mirrors it. The
    # experimental thread-attribution stamping stays off unless opted into.
    assert gateway["request_log"] == {
        "directory": LAYOUT.inference_request_log_dir,
        "body_bytes": 16384,
        "attribution": False,
    }
    assert serve["inference_request_log_dir"] == LAYOUT.inference_request_log_dir


def test_compiler_emits_request_log_attribution_when_enabled(tmp_path):
    config = _config(
        tmp_path,
        inference_gateway=InferenceGatewaySpec(
            producer=InferenceBudgetSpec(allowed_models=["gpt-producer"]),
            evaluation=InferenceBudgetSpec(allowed_models=["gpt-target"]),
            request_log_attribution=True,
        ),
    )
    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )
    gateway = json.loads(
        (output / "environment/gateway/config.json").read_text(encoding="utf-8")
    )
    assert gateway["request_log"]["attribution"] is True


def test_compiler_omits_request_log_when_disabled(tmp_path):
    config = _config(
        tmp_path,
        inference_gateway=InferenceGatewaySpec(
            producer=InferenceBudgetSpec(allowed_models=["gpt-producer"]),
            evaluation=InferenceBudgetSpec(allowed_models=["gpt-target"]),
            log_requests=False,
        ),
    )
    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )
    gateway = json.loads(
        (output / "environment/gateway/config.json").read_text(encoding="utf-8")
    )
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    assert gateway["request_log"] is None
    assert serve["inference_request_log_dir"] is None


def test_workspace_overlay_rejects_unsafe_dest_and_missing_source(tmp_path):
    present = tmp_path / "present"
    present.mkdir()
    with pytest.raises(ValidationError):
        WorkspaceOverlaySpec(source=str(present), dest="../escape")
    with pytest.raises(ValidationError):
        WorkspaceOverlaySpec(source=str(tmp_path / "missing"), dest="ok")


def test_build_config_requires_pins_and_valid_partition_references(tmp_path):
    assert (
        _config(
            tmp_path / "modal",
            harbor_requirement="harbor[modal]==0.20.0",
        ).harbor_requirement
        == "harbor[modal]==0.20.0"
    )
    with pytest.raises(ValidationError, match="pin an exact version"):
        _config(tmp_path / "unpinned", harbor_requirement="harbor>=0.1")
    with pytest.raises(ValidationError, match="selection_partition"):
        _config(tmp_path / "unknown", selection_partition="missing")
    with pytest.raises(ValidationError, match="controlled flags"):
        _config(tmp_path / "flags", extra_harbor_args=["--jobs-dir=/forged"])
    with pytest.raises(ValidationError, match="controlled flags"):
        _config(tmp_path / "outer-flags", optimizer_harbor_args=["-a", "forged"])
    assert _config(
        tmp_path / "outer-ek",
        optimizer_harbor_args=["--ek", "modal_vm_runtime=true"],
    ).optimizer_harbor_args == ["--ek", "modal_vm_runtime=true"]
    # Long forms are the dangerous half: harbor takes the last value for a key
    # and these are appended after vero's own flags, so a build declaring
    # `--agent other` silently replaced the agent the caller asked for. Note
    # harbor spells the long form of `-e` as `--env`, not `--environment`.
    for forged in (
        ["--agent", "forged"],
        ["--agent=forged"],
        ["--model", "forged"],
        ["--env", "docker"],
        ["--path", "/forged"],
    ):
        with pytest.raises(ValidationError, match="controlled flags"):
            _config(tmp_path / f"outer-long-{forged[0].strip('-')}", optimizer_harbor_args=forged)
    # `-o` pairs with `--jobs-dir`, which was already reserved for a nested run.
    with pytest.raises(ValidationError, match="controlled flags"):
        _config(tmp_path / "eval-o", extra_harbor_args=["-o", "/forged"])
    for forged in (["--dataset", "x"], ["--n-concurrent", "8"]):
        with pytest.raises(ValidationError, match="controlled flags"):
            _config(tmp_path / f"eval-{forged[0].strip('-')}", extra_harbor_args=forged)
    # Matching is exact, so these legitimate near-misses must still pass.
    for allowed in (
        ["--agent-timeout-multiplier", "4"],
        ["--agent-kwarg", "base_url=http://gw/v1"],
        ["--environment-build-timeout-multiplier", "2"],
    ):
        assert (
            _config(
                tmp_path / f"outer-ok-{allowed[0].strip('-')}",
                optimizer_harbor_args=allowed,
            ).optimizer_harbor_args
            == allowed
        )
    with pytest.raises(ValidationError, match="explicit version"):
        _config(tmp_path / "source", task_source="org/unversioned")


def test_build_config_requires_requested_models_to_be_allowed_by_their_scope(tmp_path):
    """A model the gateway would refuse must fail the build, not the run.

    The verification case is the costly one: finalization runs the target agent
    too, so a target scoring with a model its scope disallows only 403s after
    search has already spent its whole budget.
    """

    def gateway(**updates) -> InferenceGatewaySpec:
        values = {
            "producer": InferenceBudgetSpec(allowed_models=["gpt-producer"]),
            "evaluation": InferenceBudgetSpec(allowed_models=["gpt-target"]),
        }
        values.update(updates)
        return InferenceGatewaySpec(**values)

    # The matching case builds, and finalization inherits the evaluation policy.
    assert (
        _config(tmp_path / "ok", model="gpt-target", inference_gateway=gateway()).model
        == "gpt-target"
    )

    with pytest.raises(ValidationError, match="evaluation allowed_models"):
        _config(tmp_path / "search", model="gpt-typo", inference_gateway=gateway())

    # An explicit finalization scope that forgets the task model starves the
    # target agent during verification.
    with pytest.raises(ValidationError, match="finalization allowed_models"):
        _config(
            tmp_path / "narrowed",
            model="gpt-target",
            inference_gateway=gateway(
                finalization=InferenceBudgetSpec(allowed_models=["gpt-judge"])
            ),
        )

    # A per-target override is checked against finalization, not evaluation.
    with pytest.raises(ValidationError, match="finalization allowed_models"):
        _config(
            tmp_path / "override",
            model="gpt-target",
            targets=[VerificationTargetSpec(partition="test", model="gpt-other")],
            inference_gateway=gateway(),
        )

    # No gateway means no metering, so there is nothing to check against.
    assert _config(tmp_path / "ungated", model="gpt-anything").model == "gpt-anything"


def _command_backend(tmp_path: Path) -> CommandBackendSpec:
    harness = tmp_path / "harness"
    harness.mkdir(parents=True)
    (harness / "score.py").write_text("print('score')\n", encoding="utf-8")
    return CommandBackendSpec(
        harness_source=str(harness),
        command=["python", "{harness}/score.py", "--report", "{report}"],
    )


def test_command_backend_compiles_a_task_with_no_target_agent(tmp_path):
    """The outer loop stays a Harbor agent while the target is scored by a program.

    This is the Family B shape: a solver or index build has no agent to drive, so
    there is nothing for a nested `harbor run` to do.
    """
    config = _config(
        tmp_path,
        evaluation_backend="command",
        command_backend=_command_backend(tmp_path / "cmd"),
        agent_import_path=_OMIT,
        task_source=_OMIT,
    )
    assert config.agent_import_path is None

    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    backend = serve["backends"]["harbor-validation"]
    assert backend["type"] == "command"
    assert backend["harness_root"] == LAYOUT.harness
    # Case enumeration is the harness's job, so it is told where the case files are.
    assert backend["environment"]["VERO_CASES_DIR"] == LAYOUT.cases
    # None of the nested-harbor plumbing leaks into a command backend.
    assert "agent_import_path" not in backend
    assert "task_source" not in backend

    # The harness is baked into the trusted sidecar, never the agent workspace.
    assert (output / "environment/sidecar/harness/score.py").is_file()
    dockerfile = (output / "environment/sidecar/Dockerfile").read_text(encoding="utf-8")
    assert f"COPY sidecar/harness {LAYOUT.harness}" in dockerfile

    # And the trusted side still parses what the compiler emitted.
    assert HarborDeploymentConfig.model_validate(serve).backends["harbor-validation"]


def test_deployment_treats_a_backend_without_a_type_as_harbor(tmp_path):
    """serve.json written before the backend union existed must still load.

    A discriminated union needs the tag in the input, so without a default the
    union would break any run resuming from an older compiled task.
    """
    from vero.harbor.backend import HarborBackendConfig

    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id": "a", "task_name": "a"}\n', encoding="utf-8")
    backend = HarborBackendConfig(
        task_source=str(tmp_path),
        agent_import_path="target.agent:Agent",
        cases_path=str(cases),
        harbor_requirement="harbor==0.20.0",
    )
    legacy = {
        "repo_path": LAYOUT.trusted_repo,
        "agent_repo_path": LAYOUT.target_repo,
        "session_dir": LAYOUT.session_dir,
        "admin_volume": LAYOUT.admin_volume,
        "access_policies": [],
        "targets": [],
        "selection": {"mode": "submit"},
        "submit_enabled": True,
        # The tag the older compiler never wrote.
        "backends": {
            "harbor-validation": backend.model_dump(mode="json", exclude={"type"})
        },
    }
    parsed = HarborDeploymentConfig.model_validate(legacy)
    assert parsed.backends["harbor-validation"].type == "harbor"


def test_build_config_keeps_the_two_backend_kinds_apart(tmp_path):
    with pytest.raises(ValidationError, match="requires a command_backend"):
        _config(
            tmp_path / "missing",
            evaluation_backend="command",
            agent_import_path=_OMIT,
            task_source=_OMIT,
        )
    with pytest.raises(ValidationError, match="requires evaluation_backend: command"):
        _config(tmp_path / "stray", command_backend=_command_backend(tmp_path / "s"))
    with pytest.raises(ValidationError, match="requires: agent_import_path"):
        _config(tmp_path / "noagent", agent_import_path=None)
    with pytest.raises(ValidationError, match="only apply to a harbor"):
        _config(
            tmp_path / "tasksource",
            evaluation_backend="command",
            command_backend=_command_backend(tmp_path / "t"),
            agent_import_path=_OMIT,
        )
    # Harbor-only knobs are reported rather than silently ignored.
    with pytest.raises(ValidationError, match="only apply to a harbor"):
        _config(
            tmp_path / "ignored",
            evaluation_backend="command",
            command_backend=_command_backend(tmp_path / "i"),
            agent_import_path=_OMIT,
            task_source=_OMIT,
            max_retries=5,
            feedback_transcripts=True,
        )


def test_every_harbor_only_field_is_rejected_by_a_command_build(tmp_path):
    """The rejection list is derived, so this sweep needs no maintenance.

    The previous hand-written frozenset had already drifted: reward_key is
    Harbor-only in fact -- a command harness reports its own reward -- but was
    missing from the list, so setting it on a command build passed validation and
    did nothing. Looping over the group means a field added to
    _HarborEvaluationFields is covered the day it lands.
    """
    assert _HARBOR_ONLY_FIELDS == frozenset(_HarborEvaluationFields.model_fields)

    for index, name in enumerate(sorted(_HARBOR_ONLY_FIELDS)):
        # Its own default is enough: the check reads model_fields_set, so setting
        # a field at all is the mistake, whatever the value.
        default = HarborBuildConfig.model_fields[name].get_default(
            call_default_factory=True
        )
        # Listed last so it overrides the _OMIT when it is one of these two.
        overrides = {
            "evaluation_backend": "command",
            "command_backend": _command_backend(tmp_path / f"h{index}"),
            "agent_import_path": _OMIT,
            "task_source": _OMIT,
            name: default,
        }
        with pytest.raises(ValidationError, match="only apply to a harbor") as caught:
            _config(tmp_path / f"harbor-only-{index}", **overrides)
        assert name in str(caught.value)


def test_field_groups_are_inherited_not_nested():
    """Grouping must not change the wire format.

    The groups exist so the schema can be read one concern at a time, but a
    build.yaml is flat and the checked-in benchmark configs depend on that.
    Turning a group into a sub-object would break every one of them silently, so
    assert the keys stay top-level -- and that every field belongs to a group,
    which is what keeps each one documented somewhere.
    """
    groups = (
        _TaskIdentityFields,
        _HarborEvaluationFields,
        _SearchAndSelectionFields,
        _EvaluationLimitFields,
        _TaskEnvironmentFields,
        _AgentWorkspaceFields,
    )
    top_level = set(HarborBuildConfig.model_fields)
    for group in groups:
        assert set(group.model_fields) <= top_level

    grouped = {name for group in groups for name in group.model_fields}
    # Only the backend choice itself is declared on HarborBuildConfig.
    assert grouped | {"evaluation_backend", "command_backend"} == top_level


def test_compiled_task_factory_path_resolves(tmp_path):
    """The factory named in the compiled compose file must actually import.

    A stale dotted path fails when the sidecar container starts, not at import,
    so it is invisible to the type checker and to every other test. Asserting on
    the rendered artifact rather than on FACTORY_PATH alone means this still
    holds if someone puts a literal back in the template.
    """
    output = compile_harbor_task(_config(tmp_path), tmp_path / "compiled")
    compose = yaml.safe_load(
        (output / "environment/docker-compose.yaml").read_text(encoding="utf-8")
    )
    command = compose["services"][LAYOUT.sidecar_host]["command"]
    factory = command[command.index("--factory") + 1]

    assert factory == FACTORY_PATH
    assert callable(load_factory(factory))


def test_compiled_producer_base_url_matches_the_gateway_route(tmp_path):
    """Every artifact naming the producer scope must agree with the served route.

    The URL used to be rebuilt by string concatenation in four places while the
    route was declared in a fifth. A mismatch 404s or 403s inside the container
    at run time, so asserting on the rendered artifacts is the only check that
    would catch a template going back to concatenation.
    """
    expected = LAYOUT.scope_url("producer", LAYOUT.optimizer_attribution)
    config = _config(
        tmp_path,
        inference_gateway=InferenceGatewaySpec(
            producer=InferenceBudgetSpec(allowed_models=["gpt-producer"]),
            evaluation=InferenceBudgetSpec(allowed_models=["gpt-target"]),
        ),
        model="gpt-target",
    )
    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )

    compose = yaml.safe_load(
        (output / "environment/docker-compose.yaml").read_text(encoding="utf-8")
    )
    assert compose["services"]["main"]["environment"]["OPENAI_BASE_URL"] == expected
    launch = json.loads(
        (output / "environment/gateway/launch.json").read_text(encoding="utf-8")
    )
    assert launch["producer_base_url"] == expected
    seed = (output / "environment/main/seed.sh").read_text(encoding="utf-8")
    assert f'base_url = "{expected}"' in seed

    # And the path the gateway actually serves is the same string, not a copy.
    assert LAYOUT.scope_route.startswith(LAYOUT.scope_route_base)
    assert expected.endswith(
        LAYOUT.scope_path("producer", LAYOUT.optimizer_attribution)
    )


def test_routed_credentials_are_set_not_merely_left_unblanked(tmp_path):
    """Every name excluded from blanking must be positively set instead.

    The compiler blanks each declared secret in the main container, skipping
    GATEWAY_ROUTED_CREDENTIALS because the compose template assigns those
    itself. That exclusion is only safe while the two agree. A name dropped
    from the blanking loop but never assigned would be neither blanked nor
    set, so whatever the host exported would survive into the optimizer's
    container -- which is the one thing the scrub exists to prevent.
    """
    config = _config(
        tmp_path,
        inference_gateway=InferenceGatewaySpec(
            producer=InferenceBudgetSpec(allowed_models=["gpt-producer"]),
            evaluation=InferenceBudgetSpec(allowed_models=["gpt-target"]),
        ),
    )
    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )
    compose = yaml.safe_load(
        (output / "environment/docker-compose.yaml").read_text(encoding="utf-8")
    )
    environment = compose["services"]["main"]["environment"]

    assert set(GATEWAY_ROUTED_CREDENTIALS) == set(LAYOUT.routed_credential_envs)
    for name in GATEWAY_ROUTED_CREDENTIALS:
        assert name in environment, f"{name} is skipped by the scrub but never set"
        assert environment[name], f"{name} is set to an empty value"

    # And they carry the scoped values, not the upstream ones.
    assert environment[LAYOUT.producer_api_key_env] != ""
    assert environment[LAYOUT.producer_base_url_env] == LAYOUT.scope_url(
        "producer", LAYOUT.optimizer_attribution
    )
    # The raw upstream is blanked in the same container.
    assert environment[LAYOUT.gateway_upstream_api_key_env] == ""


def test_load_build_config_resolves_relative_local_paths(tmp_path):
    target = _target_repo(tmp_path / "target")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    config_path = tmp_path / "build.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: org/task",
                "agent_repo: target",
                "task_source: tasks",
                "agent_import_path: target.agent:Agent",
                "harbor_requirement: harbor==0.1.17",
                "partitions:",
                "  validation: [org/a]",
                "  test: [org/b]",
                "agent_access:",
                "  - partition: validation",
                "selection_partition: validation",
                "targets:",
                "  - partition: test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_harbor_build_config(config_path)

    assert loaded.agent_repo == str(target)
    assert loaded.task_source == str(tasks)


def test_load_build_config_resolves_a_relative_harness_source(tmp_path):
    """A command build's harness is relocatable like every other path field."""
    target = _target_repo(tmp_path / "target")
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "score.py").write_text("print('score')\n", encoding="utf-8")
    config_path = tmp_path / "build.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: org/solve",
                f"agent_repo: {target.name}",
                "harbor_requirement: harbor==0.1.17",
                "partitions:",
                "  validation: [s0]",
                "  test: [s1]",
                "agent_access:",
                "  - partition: validation",
                "    min_aggregate_cases: 1",
                "selection_partition: validation",
                "targets:",
                "  - partition: test",
                "evaluation_backend: command",
                "command_backend:",
                f"  harness_source: {harness.name}",
                '  command: [python, "{harness}/score.py"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_harbor_build_config(config_path)

    assert loaded.command_backend is not None
    assert loaded.command_backend.harness_source == str(harness)


def test_load_build_config_supports_partition_files_and_validates_manifest(tmp_path):
    target = _target_repo(tmp_path / "target")
    partitions = tmp_path / "partitions"
    partitions.mkdir()
    (partitions / "validation.json").write_text(
        '["task-a", "task-b"]\n', encoding="utf-8"
    )
    (partitions / "test.json").write_text('["task-c"]\n', encoding="utf-8")
    source = "org/benchmark@sha256:abc"
    manifest = partitions / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task_source": source,
                "tasks": [
                    {"name": "task-a", "ref": "sha256:a"},
                    {"name": "task-b", "ref": "sha256:b"},
                    {"name": "task-c", "ref": "sha256:c"},
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "build.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: org/task",
                "agent_repo: target",
                f"task_source: {source}",
                "task_manifest: partitions/manifest.json",
                "agent_import_path: target.agent:Agent",
                "harbor_requirement: harbor==0.20.0",
                "partition_files:",
                "  validation: partitions/validation.json",
                "  test: partitions/test.json",
                "agent_access:",
                "  - partition: validation",
                "selection_partition: validation",
                "targets:",
                "  - partition: test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_harbor_build_config(config_path)

    assert loaded.agent_repo == str(target)
    assert loaded.partitions == {
        "validation": ["task-a", "task-b"],
        "test": ["task-c"],
    }
    assert loaded.task_manifest == str(manifest)

    (partitions / "test.json").write_text('["task-missing"]\n', encoding="utf-8")
    with pytest.raises(ValidationError, match="absent from task_manifest"):
        load_harbor_build_config(config_path)


def test_load_build_config_matches_vendored_local_task_source(tmp_path):
    # A local (vendored) task_source is resolved to an absolute path by the
    # loader once the directory exists, while the committed manifest records it
    # relative to itself; the two must still be recognized as the same source.
    _target_repo(tmp_path / "target")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    partitions = tmp_path / "partitions"
    partitions.mkdir()
    (partitions / "validation.json").write_text('["task-a"]\n', encoding="utf-8")
    (partitions / "test.json").write_text('["task-a"]\n', encoding="utf-8")
    (partitions / "manifest.json").write_text(
        json.dumps(
            {
                "task_source": "../tasks",
                "tasks": [{"name": "task-a", "ref": "sha256:a"}],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "build.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: org/task",
                "agent_repo: target",
                "task_source: tasks",
                "task_manifest: partitions/manifest.json",
                "agent_import_path: target.agent:Agent",
                "harbor_requirement: harbor==0.20.0",
                "partition_files:",
                "  validation: partitions/validation.json",
                "  test: partitions/test.json",
                "agent_access:",
                "  - partition: validation",
                "selection_partition: validation",
                "targets:",
                "  - partition: test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_harbor_build_config(config_path)
    assert loaded.task_source == str(tasks.resolve())

    # A genuinely different source still fails.
    (partitions / "manifest.json").write_text(
        json.dumps(
            {
                "task_source": "../elsewhere",
                "tasks": [{"name": "task-a", "ref": "sha256:a"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="does not match build task_source"):
        load_harbor_build_config(config_path)


def test_compiler_emits_isolated_canonical_harbor_task(tmp_path):
    config = _config(tmp_path)
    output = compile_harbor_task(
        config,
        tmp_path / "compiled",
        vero_root=Path(__file__).parents[1],
    )

    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    assert set(serve["backends"]) == {"harbor-validation", "harbor-test"}
    access = serve["access_policies"][0]["access"]
    assert access["disclosure"] == "aggregate"
    assert access["expose_case_resources"] is True
    assert access["min_aggregate_cases"] == 5  # safe floor survives compilation
    assert serve["budgets"][0]["total_runs"] == 5
    assert serve["selection"]["backend_id"] == "harbor-validation"
    assert serve["targets"][0]["backend_id"] == "harbor-test"
    assert serve["targets"][0]["reward_scale"] == 1.0
    # A grace period, not a ceiling — and explicitly NOT timeout_seconds, which is
    # sized to be unreachable. Inheriting it stalled officeqa run #4's
    # finalization for hours behind a sub-run that had already finished its work.
    assert serve["evaluation_drain_timeout_seconds"] == 600.0
    assert serve["evaluation_drain_timeout_seconds"] != config.timeout_seconds
    assert serve["backends"]["harbor-test"]["task_source"] == LAYOUT.task_source
    assert serve["backends"]["harbor-test"]["python_version"] == "3.12"
    assert serve["backends"]["harbor-test"]["case_timeout_seconds"] == 180.0
    assert serve["backends"]["harbor-test"]["task_agent_timeout_seconds"] == 600.0
    assert (
        serve["backends"]["harbor-validation"]["case_resources_cache_path"]
        == f"{LAYOUT.case_resources_dir}/validation"
    )
    assert serve["access_policies"][0]["limits"]["retry"]["max_attempts"] == 1
    assert serve["access_policies"][0]["limits"]["case_timeout_seconds"] == 180.0
    assert "use_evaluation_copies" not in serve
    instruction = (output / "instruction.md").read_text()
    assert "--detach" in instruction
    assert "evals status JOB_ID" in instruction
    assert "randomness remains" in instruction
    for partition, backend in serve["backends"].items():
        partition_name = partition.removeprefix("harbor-")
        backend["cases_path"] = str(
            output / f"environment/sidecar/cases/{partition_name}.jsonl"
        )
    HarborDeploymentConfig.model_validate(serve)
    test_case = json.loads(
        (output / "environment/sidecar/cases/test.jsonl").read_text(encoding="utf-8")
    )
    assert test_case == {
        "id": "task-hidden",
        "task_name": "task-hidden",
        "result_task_name": "org/task-hidden",
    }
    assert (output / "environment/sidecar/task-source/task-hidden/task.toml").is_file()
    assert not (output / "environment/agent-seed/protected-tasks").exists()
    instruction = (output / "instruction.md").read_text(encoding="utf-8")
    assert "## Objective\n\nImprove the program" in instruction
    assert "--backend harbor-validation" in instruction
    # k-anonymity floor (default 5) is disclosed to the agent
    assert "must include at least 5 cases" in instruction
    assert "Complete task resources" in instruction
    assert ".evals/results/" in instruction
    task_toml = (output / "task.toml").read_text(encoding="utf-8")
    assert 'name = "org/optimize-\\"program\\""' in task_toml
    assert tomllib.loads(task_toml)["task"]["name"] == 'org/optimize-"program"'
    compose = (output / "environment/docker-compose.yaml").read_text()
    assert "vero.harbor.deployment:build_harbor_components" in compose
    assert f"admin_state:{LAYOUT.admin_volume}" in compose
    assert f"agent_context:{LAYOUT.target_evals}:ro" in compose
    assert f"agent_context:{LAYOUT.agent_volume}" in compose
    assert set(yaml.safe_load(compose)["services"]) == {"main", "eval-sidecar"}
    sidecar_dockerfile = (output / "environment/sidecar/Dockerfile").read_text()
    assert 'uv pip install --system "harbor==0.1.17"' in sidecar_dockerfile
    # the unprivileged harness user exists and gets a pre-warmed uv cache, so
    # `uv run` as harness can resolve the candidate package offline
    assert "useradd -m -u 1002 harness" in sidecar_dockerfile
    assert "chown -R harness:harness /home/harness/.cache" in sidecar_dockerfile
    seed = (output / "environment/main/seed.sh").read_text()
    assert f"-path {LAYOUT.target_evals} -prune" in seed
    assert f"'/.evals/' >> {LAYOUT.target_git_exclude}" in seed
    test_script = output / "tests/test.sh"
    assert test_script.stat().st_mode & 0o111
    assert "vero harbor export-session" in test_script.read_text()


def test_compiler_checks_secrets_before_writing_and_rejects_source_overlap(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, secrets=["MISSING_TEST_SECRET"])
    output = tmp_path / "compiled"
    monkeypatch.delenv("MISSING_TEST_SECRET", raising=False)

    with pytest.raises(ValueError, match="MISSING_TEST_SECRET"):
        compile_harbor_task(
            config,
            output,
            vero_root=Path(__file__).parents[1],
        )
    assert not output.exists()

    monkeypatch.setenv("MISSING_TEST_SECRET", "configured")
    compile_harbor_task(
        config,
        output,
        vero_root=Path(__file__).parents[1],
    )
    task = tomllib.loads((output / "task.toml").read_text(encoding="utf-8"))
    assert task["environment"]["env"] == {
        "MISSING_TEST_SECRET": "${MISSING_TEST_SECRET}"
    }

    safe = config.model_copy(update={"secrets": []})
    with pytest.raises(ValueError, match="overlaps protected source"):
        compile_harbor_task(
            safe,
            safe.agent_repo,
            vero_root=Path(__file__).parents[1],
        )

    (Path(safe.agent_repo) / ".evals").mkdir()
    (Path(safe.agent_repo) / ".evals" / "context.json").write_text("{}\n")
    _git(Path(safe.agent_repo), "add", "-f", ".evals/context.json")
    _git(Path(safe.agent_repo), "commit", "-q", "-m", "reserved context")
    with pytest.raises(ValueError, match="reserved path"):
        compile_harbor_task(
            safe,
            tmp_path / "reserved-context",
            vero_root=Path(__file__).parents[1],
        )


def test_compiler_isolates_upstream_inference_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_UPSTREAM_KEY", "real-provider-secret")
    monkeypatch.setenv("TEST_UPSTREAM_URL", "https://provider.example/v1")
    monkeypatch.setenv("TEST_MODAL_TOKEN", "modal-secret")
    config = _config(
        tmp_path,
        secrets=["TEST_MODAL_TOKEN"],
        inference_gateway=InferenceGatewaySpec(
            upstream_api_key_env="TEST_UPSTREAM_KEY",
            upstream_base_url_env="TEST_UPSTREAM_URL",
            producer=InferenceBudgetSpec(
                allowed_models=["gpt-producer"],
                max_requests=10,
                max_tokens=1000,
            ),
            evaluation=InferenceBudgetSpec(
                allowed_models=["gpt-target"],
                max_requests=20,
                max_tokens=2000,
            ),
        ),
    )

    output = compile_harbor_task(
        config,
        tmp_path / "compiled",
        vero_root=Path(__file__).parents[1],
    )

    task = tomllib.loads((output / "task.toml").read_text())
    assert task["environment"]["env"] == {
        "TEST_MODAL_TOKEN": "${TEST_MODAL_TOKEN}",
        "VERO_INFERENCE_UPSTREAM_API_KEY": "${VERO_INFERENCE_UPSTREAM_API_KEY}",
        "VERO_INFERENCE_UPSTREAM_BASE_URL": "${VERO_INFERENCE_UPSTREAM_BASE_URL}",
    }
    compose = yaml.safe_load((output / "environment/docker-compose.yaml").read_text())
    assert set(compose["services"]) == {
        "main",
        "eval-sidecar",
        "inference-gateway",
    }
    main_environment = compose["services"]["main"]["environment"]
    assert main_environment["TEST_MODAL_TOKEN"] == ""
    assert main_environment["VERO_INFERENCE_UPSTREAM_API_KEY"] == ""
    assert main_environment["OPENAI_API_KEY"]
    assert main_environment["OPENAI_API_KEY"] != "real-provider-secret"
    assert main_environment["OPENAI_BASE_URL"].endswith("/scopes/producer/optimizer/v1")
    assert compose["services"]["eval-sidecar"]["environment"] == {
        "TEST_MODAL_TOKEN": "${TEST_MODAL_TOKEN:?TEST_MODAL_TOKEN must be set for the eval sidecar}"
    }
    assert compose["services"]["inference-gateway"]["environment"] == {
        "VERO_INFERENCE_UPSTREAM_API_KEY": "${VERO_INFERENCE_UPSTREAM_API_KEY:?VERO_INFERENCE_UPSTREAM_API_KEY must be set for the inference gateway}",
        "VERO_INFERENCE_UPSTREAM_BASE_URL": "${VERO_INFERENCE_UPSTREAM_BASE_URL:?VERO_INFERENCE_UPSTREAM_BASE_URL must be set for the inference gateway}",
    }
    gateway = json.loads((output / "environment/gateway/config.json").read_text())
    assert "real-provider-secret" not in json.dumps(gateway)
    assert gateway["scopes"]["producer"]["token_sha256"]
    launch = json.loads((output / "environment/gateway/launch.json").read_text())
    assert launch["upstream_api_key_source"] == "TEST_UPSTREAM_KEY"
    assert launch["upstream_api_key_target"] == "VERO_INFERENCE_UPSTREAM_API_KEY"
    assert launch["producer_api_key"] == main_environment["OPENAI_API_KEY"]
    seed = (output / "environment/main/seed.sh").read_text()
    assert 'model_provider = "vero_gateway"' in seed
    assert "supports_websockets = false" in seed
    serve = json.loads((output / "environment/sidecar/serve.json").read_text())
    backend = serve["backends"]["harbor-validation"]
    assert backend["passthrough_environment"] == ["TEST_MODAL_TOKEN"]
    assert backend["inference_gateway_token"]
    assert backend["inference_gateway_token"] != "real-provider-secret"
    assert "real-provider-secret" not in json.dumps(serve)
    assert (output / "environment/gateway/Dockerfile").is_file()


def test_compiler_uses_published_version_outside_a_source_checkout(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    from vero.harbor.build import compiler

    monkeypatch.setattr(
        compiler,
        "__file__",
        "/installed/site-packages/vero/harbor/build/compiler.py",
    )
    monkeypatch.setattr(compiler, "distribution_version", lambda _name: "0.5.0")

    output = compiler.compile_harbor_task(config, tmp_path / "compiled")

    assert not (output / "environment/vero").exists()
    dockerfile = (output / "environment/Dockerfile").read_text(encoding="utf-8")
    assert "scale-vero[harbor]==0.5.0" in dockerfile


def test_agent_access_defaults_to_safe_k_anonymity_floor():
    # Omitting min_aggregate_cases must yield a real floor (5), not a no-op (1).
    assert AgentAccessSpec(partition="validation").min_aggregate_cases == 5


def test_run_command_forwards_agent_env_as_harbor_ae(tmp_path, monkeypatch):
    # `config.agent_env` must be rendered as repeated `--ae KEY=VALUE` on the
    # optimizer's `harbor run` so it reaches the agent's setup/install exec
    # (harbor injects --ae into the agent's extra_env / scoped_exec_env). This is
    # the only supported channel for e.g. UV_TOOL_BIN_DIR on a non-root sandbox;
    # extra_harbor_args only flows into the eval sub-run, not this command.
    from click.testing import CliRunner

    from vero.harbor import build as harbor_build
    from vero.harbor import cli as harbor_cli

    config = _config(
        tmp_path,
        agent_env={"UV_TOOL_BIN_DIR": "/home/agent/.local/bin", "FOO": "bar"},
    )
    config_path = tmp_path / "build.yaml"
    config_path.write_text("name: placeholder\n", encoding="utf-8")

    captured: dict[str, list[str]] = {}

    def fake_run(command, *args, **kwargs):
        captured["command"] = list(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(harbor_cli.shutil, "which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr(harbor_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        harbor_build, "load_harbor_build_config", lambda *a, **k: config
    )
    monkeypatch.setattr(harbor_build, "compile_harbor_task", lambda cfg, out: out)

    result = CliRunner().invoke(
        harbor_cli.harbor,
        [
            "run",
            "--config",
            str(config_path),
            "--agent",
            "codex",
            "--model",
            "gpt-5.4",
            "--environment",
            "modal",
        ],
    )

    assert result.exit_code == 0, result.output
    command = captured["command"]
    assert "--ae" in command
    assert "--ae UV_TOOL_BIN_DIR=/home/agent/.local/bin" in shlex.join(command)
    assert "--ae FOO=bar" in shlex.join(command)
    # Deterministic (sorted-by-key) ordering: FOO before UV_TOOL_BIN_DIR.
    joined = shlex.join(command)
    assert joined.index("--ae FOO=bar") < joined.index("--ae UV_TOOL_BIN_DIR=")


def _serve_backends(tmp_path: Path, **updates) -> dict:
    config = _config(tmp_path, **updates)
    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    return serve["backends"]


def test_verification_target_spec_defaults_n_attempts_none():
    target = VerificationTargetSpec(partition="test")
    assert target.n_attempts is None
    assert target.aggregate_attempts is None


def test_compiler_applies_per_target_n_attempts_override(tmp_path):
    # Only the held-out test backend gets 3x/mean; validation (search/selection)
    # keeps the global n_attempts=1.
    backends = _serve_backends(
        tmp_path,
        n_attempts=1,
        targets=[
            VerificationTargetSpec(
                partition="test", n_attempts=3, aggregate_attempts="mean"
            )
        ],
    )
    assert backends["harbor-test"]["n_attempts"] == 3
    assert backends["harbor-test"]["aggregate_attempts"] == "mean"
    assert backends["harbor-validation"]["n_attempts"] == 1


def test_compiler_defaults_target_n_attempts_to_global(tmp_path):
    backends = _serve_backends(tmp_path, n_attempts=2)
    # No per-target override -> every backend inherits the global (backward-compat).
    assert backends["harbor-test"]["n_attempts"] == 2
    assert backends["harbor-validation"]["n_attempts"] == 2


def test_build_rejects_n_attempts_override_on_agent_partition(tmp_path):
    # validation is agent-evaluable + the selection partition; its backend is
    # shared with search, so an override there must be rejected.
    with pytest.raises(ValidationError, match="agent-evaluable partition"):
        _config(
            tmp_path,
            targets=[VerificationTargetSpec(partition="validation", n_attempts=3)],
        )


def test_instruction_omits_retired_insights_paragraph(tmp_path):
    # The insights-generator subagent / skills/insights overlay was retired; the
    # rendered optimizer instruction must not advertise it (it did whenever the
    # always-baked evals skill made overlay_present true).
    config = _config(tmp_path)
    output = compile_harbor_task(
        config, tmp_path / "compiled", vero_root=Path(__file__).parents[1]
    )
    instruction = (output / "instruction.md").read_text(encoding="utf-8")
    assert "insights-generator" not in instruction
    assert "skills/insights/" not in instruction


def test_instruction_advertises_seed_only_where_the_backend_accepts_it(tmp_path):
    # HarborBackend.validate_request rejects request.seed outright, so telling a
    # Harbor-scored optimizer to "pass --seed N to reproduce a comparison" sends
    # it into a guaranteed 400. Observed live: the optimizer followed that advice
    # and burned a round trip on `invalid evaluation request`.
    harbor = compile_harbor_task(
        _config(tmp_path), tmp_path / "harbor", vero_root=Path(__file__).parents[1]
    )
    instruction = (harbor / "instruction.md").read_text(encoding="utf-8")
    assert "--seed" in instruction  # still named, so the rejection is not a surprise
    assert "rejects `--seed`" in instruction
    assert "re-run the identical selection" in instruction

    # A command backend does its own sampling and accepts the seed.
    command = compile_harbor_task(
        _config(
            tmp_path / "cmd-config",
            evaluation_backend="command",
            command_backend=_command_backend(tmp_path / "cmd"),
            agent_import_path=_OMIT,
            task_source=_OMIT,
        ),
        tmp_path / "command",
        vero_root=Path(__file__).parents[1],
    )
    instruction = (command / "instruction.md").read_text(encoding="utf-8")
    assert "reproduce a noisy comparison exactly" in instruction
    assert "rejects `--seed`" not in instruction


def test_drain_timeout_is_independent_of_the_unreachable_eval_ceiling(tmp_path):
    # timeout_seconds is deliberately set above ceil(trials/concurrency) x
    # case_timeout so it can never fire. The drain is the opposite kind of clock:
    # it decides how long finalization waits on already-running agent evaluations
    # before cancelling them, and cancellation is graceful. Tying the two makes a
    # hung sub-run cost hours of held-out scoring, which is what happened.
    default = compile_harbor_task(
        _config(tmp_path / "cfg-default", timeout_seconds=90000.0),
        tmp_path / "default",
        vero_root=Path(__file__).parents[1],
    )
    serve = json.loads(
        (default / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    assert serve["evaluation_drain_timeout_seconds"] == 600.0

    # An explicit value still wins, for a benchmark that genuinely wants to wait.
    explicit = compile_harbor_task(
        _config(
            tmp_path / "cfg-explicit",
            timeout_seconds=90000.0,
            evaluation_drain_timeout_seconds=1800.0,
        ),
        tmp_path / "explicit",
        vero_root=Path(__file__).parents[1],
    )
    serve = json.loads(
        (explicit / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    assert serve["evaluation_drain_timeout_seconds"] == 1800.0


def test_unresolvable_path_task_source_names_the_missing_data(tmp_path):
    """A missing tasks/ directory must not be reported as a missing version pin.

    Vendored task data is gitignored, so a fresh checkout has no tasks/ at all.
    The loader leaves an unresolvable relative path untouched and this validator
    then sees a bare name, so the old message -- "registry task_source must
    include an explicit version" -- sent the reader looking for a version to add
    when the real problem was unfetched data. That cost real debugging time.
    """
    # _config provisions a target repo under the directory it is given, so each
    # case needs its own.
    def case(name: str, **updates):
        root = tmp_path / name
        root.mkdir()
        return _config(root, **updates)

    with pytest.raises(ValidationError, match="looks like a path but does not exist"):
        case("relative", task_source="../tasks")
    with pytest.raises(ValidationError, match="vendor_tasks.sh"):
        case("absolute", task_source="/nowhere/officeqa/tasks")

    # A registry reference still gets the pin message, and a pinned one is fine
    # even though it does not resolve on this filesystem -- note it contains "/",
    # so the path-shape test must not run ahead of the version check.
    with pytest.raises(ValidationError, match="explicit version"):
        case("unpinned", task_source="gaia/gaia")
    case("pinned", task_source="gaia/gaia@sha256:abc123")


def test_model_alias_tolerates_keys_a_given_launch_never_requests():
    """One build config serves every cell of a grid, so the alias map must
    tolerate keys the current launch cannot request.

    `allowed_models` is templated per launch (``${optimizer_model}``), so a map
    that pins the right deployment for one optimizer necessarily carries keys the
    other launches never name. Those entries are inert -- an alias can only fire
    for a model the allow-list already admitted -- so rejecting them would make
    the field unusable in exactly the shared-config case it exists for.
    """
    spec = InferenceBudgetSpec(
        allowed_models=["claude-opus-5"],
        model_aliases={"gpt-5.6-sol": "azure_ai/gpt-5.6-sol"},
    )
    assert spec.model_aliases == {"gpt-5.6-sol": "azure_ai/gpt-5.6-sol"}

    # A self-alias is a no-op and is dropped rather than rejected, so a template
    # that resolves to `{X: X}` when no override is passed still validates.
    assert (
        InferenceBudgetSpec(
            allowed_models=["gpt-producer"],
            model_aliases={"gpt-producer": "gpt-producer"},
        ).model_aliases
        == {}
    )

    with pytest.raises(ValidationError, match="must not be empty"):
        InferenceBudgetSpec(
            allowed_models=["gpt-producer"], model_aliases={"gpt-producer": "  "}
        )

    # Reaches the gateway via model_dump, which is how the compiler lowers it.
    assert InferenceBudgetSpec(
        allowed_models=["gpt-producer"],
        model_aliases={"gpt-producer": "vendor_x/gpt-producer"},
    ).model_dump(mode="json")["model_aliases"] == {
        "gpt-producer": "vendor_x/gpt-producer"
    }
