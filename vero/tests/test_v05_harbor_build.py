from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vero.harbor import (
    AgentAccessSpec,
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

BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "harness-engineering-bench"

# harness-engineering-bench is a separate branch/PR in the stacked split, so
# skip the benchmark-config tests when it isn't checked out here.
_requires_benchmarks = pytest.mark.skipif(
    not BENCHMARK_ROOT.exists(),
    reason="harness-engineering-bench is not present on this branch",
)


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


@_requires_benchmarks
@pytest.mark.parametrize(
    ("benchmark", "producer_models"),
    [
        # gaia parametrizes the producer model via ${optimizer_model}; the
        # default (no --param) resolves to gpt-5.4.
        ("gaia", ["gpt-5.4"]),
        ("swe-atlas-qna", ["gpt-5.4"]),
        ("tau3", ["gpt-5.4"]),
    ],
)
def test_canonical_benchmarks_isolate_upstream_inference_credentials(
    benchmark, producer_models
):
    # All benchmarks live at the top level of harness-engineering-bench.
    config = load_harbor_build_config(
        BENCHMARK_ROOT / benchmark / "baseline" / "build.yaml"
    )

    assert config.inference_gateway is not None
    assert not {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    }.intersection(config.secrets)
    assert config.inference_gateway.upstream_api_key_env == "OPENAI_API_KEY"
    assert config.inference_gateway.upstream_base_url_env == "OPENAI_BASE_URL"
    assert config.inference_gateway.producer.allowed_models == producer_models
    assert config.inference_gateway.producer.max_requests is None
    assert config.inference_gateway.producer.max_tokens is None
    assert config.inference_gateway.evaluation.allowed_models == [
        "gpt-5.4-mini-2026-03-17"
    ]
    assert config.inference_gateway.evaluation.max_requests == 15000
    assert config.inference_gateway.evaluation.max_tokens == 100000000


@_requires_benchmarks
def test_build_params_override_run_time_knobs_without_rebuild():
    path = BENCHMARK_ROOT / "gaia" / "baseline" / "build.yaml"

    default = load_harbor_build_config(path)
    assert default.environment_name == "modal"
    assert default.inference_gateway.producer.allowed_models == ["gpt-5.4"]

    overridden = load_harbor_build_config(
        path, params={"optimizer_model": "gpt-5.5", "inner_env": "docker"}
    )
    assert overridden.environment_name == "docker"
    assert overridden.inference_gateway.producer.allowed_models == ["gpt-5.5"]
    # The rest of the measurement substrate is untemplated and stays fixed.
    assert overridden.model == default.model
    assert overridden.task_source == default.task_source


def test_build_param_substitution_semantics():
    from vero.harbor.build.config import _substitute_build_param as sub

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
    return HarborBuildConfig(**values)


def test_compiler_bakes_workspace_overlay_into_agent_environment(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / ".claude" / "agents").mkdir(parents=True)
    (bundle / ".claude" / "agents" / "insights.md").write_text("# a\n", encoding="utf-8")
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
    assert "COPY overlay /opt/overlay" in (env / "Dockerfile").read_text()
    seed = (env / "main" / "seed.sh").read_text()
    assert "cp -a /opt/overlay/. /work/agent/" in seed
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
    assert "/opt/overlay" not in (env / "main" / "seed.sh").read_text()


def test_compiler_bakes_evals_skill_by_default(tmp_path):
    output = compile_harbor_task(_config(tmp_path), tmp_path / "compiled")
    env = output / "environment"
    skill = env / "overlay" / "skills" / "evals" / "SKILL.md"
    assert skill.is_file()
    assert "evals run" in skill.read_text(encoding="utf-8")
    assert "'/skills/' >> /work/agent/.git/info/exclude" in (
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
    assert compose["services"]["main"]["environment"][
        "VERO_INFERENCE_UPSTREAM_API_KEY"
    ] == ""


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
    assert scopes["finalization"]["token_sha256"] != scopes["evaluation"]["token_sha256"]
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
    # request/response on its state volume and the sidecar mirrors it.
    assert gateway["request_log"] == {
        "directory": "/state/inference/requests",
        "body_bytes": 16384,
    }
    assert serve["inference_request_log_dir"] == "/state/inference/requests"


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
    with pytest.raises(ValidationError, match="explicit version"):
        _config(tmp_path / "source", task_source="org/unversioned")


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
    assert serve["evaluation_drain_timeout_seconds"] == config.timeout_seconds
    assert serve["backends"]["harbor-test"]["task_source"] == "/opt/task-source"
    assert serve["backends"]["harbor-test"]["python_version"] == "3.12"
    assert serve["backends"]["harbor-test"]["case_timeout_seconds"] == 180.0
    assert serve["backends"]["harbor-test"]["task_agent_timeout_seconds"] == 600.0
    assert (
        serve["backends"]["harbor-validation"]["case_resources_cache_path"]
        == "/state/admin/case-resources/validation"
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
    assert "admin_state:/state/admin" in compose
    assert "agent_context:/work/agent/.evals:ro" in compose
    assert "agent_context:/state/agent-context" in compose
    assert set(yaml.safe_load(compose)["services"]) == {"main", "eval-sidecar"}
    sidecar_dockerfile = (output / "environment/sidecar/Dockerfile").read_text()
    assert 'uv pip install --system "harbor==0.1.17"' in sidecar_dockerfile
    # the unprivileged harness user exists and gets a pre-warmed uv cache, so
    # `uv run` as harness can resolve the candidate package offline
    assert "useradd -m -u 1002 harness" in sidecar_dockerfile
    assert "chown -R harness:harness /home/harness/.cache" in sidecar_dockerfile
    seed = (output / "environment/main/seed.sh").read_text()
    assert "-path /work/agent/.evals -prune" in seed
    assert "'/.evals/' >> /work/agent/.git/info/exclude" in seed
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
