from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from vero.harbor import (
    AgentAccessSpec,
    HarborBuildConfig,
    VerificationTargetSpec,
    compile_harbor_task,
    load_harbor_build_config,
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
    (task_source / "task.toml").write_text('[task]\nname="inner"\n')
    values = {
        "name": 'org/optimize-"program"',
        "description": "Improve the program",
        "agent_repo": str(target),
        "task_source": str(task_source),
        "agent_import_path": "target.agent:Agent",
        "harbor_requirement": "harbor==0.1.17",
        "partitions": {
            "validation": ["org/task-a", "org/task-b", "org/task-c", "org/task-d", "org/task-e"],
            "test": ["org/task-hidden"],
        },
        "agent_access": [
            AgentAccessSpec(
                partition="validation",
                total_runs=5,
                total_cases=25,
                max_cases_per_run=5,
            )
        ],
        "selection_partition": "validation",
        "targets": [VerificationTargetSpec(partition="test")],
    }
    values.update(updates)
    return HarborBuildConfig(**values)


def test_build_config_requires_pins_and_valid_partition_references(tmp_path):
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
    assert serve["access_policies"][0]["disclosure"] == "aggregate"
    assert serve["budgets"][0]["total_runs"] == 5
    assert serve["selection"]["backend_id"] == "harbor-validation"
    assert serve["targets"][0]["backend_id"] == "harbor-test"
    assert serve["targets"][0]["reward_scale"] == 1.0
    assert serve["backends"]["harbor-test"]["task_source"] == "/opt/task-source"
    assert (output / "environment/sidecar/task-source/task.toml").is_file()
    assert not (output / "environment/agent-seed/protected-tasks").exists()
    instruction = (output / "instruction.md").read_text(encoding="utf-8")
    assert "--backend harbor-validation" in instruction
    assert "at least 5 cases" in instruction
    task_toml = (output / "task.toml").read_text(encoding="utf-8")
    assert 'name = "org/optimize-\\"program\\""' in task_toml
    assert tomllib.loads(task_toml)["task"]["name"] == 'org/optimize-"program"'
    compose = (output / "environment/docker-compose.yaml").read_text()
    assert "vero.harbor.deployment:build_harbor_components" in compose
    assert "admin_state:/state/admin" in compose
    assert set(yaml.safe_load(compose)["services"]) == {"main", "eval-sidecar"}
    assert (output / "tests/test.sh").stat().st_mode & 0o111


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

    safe = config.model_copy(update={"secrets": []})
    with pytest.raises(ValueError, match="overlaps protected source"):
        compile_harbor_task(
            safe,
            safe.agent_repo,
            vero_root=Path(__file__).parents[1],
        )


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
    assert 'scale-vero[harbor]==0.5.0' in dockerfile
