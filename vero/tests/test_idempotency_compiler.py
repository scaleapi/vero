"""Recompiling the same build has to be able to rejoin a preserved session.

Every test here defends one half of that: the compiled tree's identities (the
baseline commit sha, the three gateway tokens) must not drift when nothing about
the build changed, and a compile that dies must not overwrite the tree that a
resume would otherwise still be able to use.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from vero.harbor import (
    AgentAccessSpec,
    HarborBuildConfig,
    InferenceBudgetSpec,
    InferenceGatewaySpec,
    VerificationTargetSpec,
)
from vero.harbor.build import compiler

_VERO_ROOT = Path(__file__).parents[1]


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


def _task_source(path: Path, names: list[str]) -> Path:
    for name in names:
        task = path / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            f'[task]\nname="org/{name}"\n', encoding="utf-8"
        )
    return path


def _config(root: Path, **updates) -> HarborBuildConfig:
    """The smallest build that still exercises a harbor backend per partition."""
    values = {
        "name": "org/optimize-program",
        "description": "Improve the program",
        "agent_repo": str(_target_repo(root / "target")),
        "task_source": str(
            _task_source(root / "protected-tasks", ["task-a", "task-b", "task-hidden"])
        ),
        "agent_import_path": "target.agent:Agent",
        "harbor_requirement": "harbor==0.1.17",
        "partitions": {"validation": ["task-a", "task-b"], "test": ["task-hidden"]},
        "agent_access": [AgentAccessSpec(partition="validation", total_runs=5)],
        "selection_partition": "validation",
        "targets": [VerificationTargetSpec(partition="test")],
    }
    values.update(updates)
    return HarborBuildConfig(**values)


def _gateway_config(root: Path, **updates) -> HarborBuildConfig:
    return _config(
        root,
        inference_gateway=InferenceGatewaySpec(
            producer=InferenceBudgetSpec(allowed_models=["gpt-producer"]),
            evaluation=InferenceBudgetSpec(allowed_models=["gpt-target"]),
        ),
        **updates,
    )


def _baseline_version(output: Path) -> str:
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    return serve["selection"]["baseline_version"]


def _tokens(output: Path) -> tuple[str, str, str]:
    launch = json.loads(
        (output / "environment/gateway/launch.json").read_text(encoding="utf-8")
    )
    serve = json.loads(
        (output / "environment/sidecar/serve.json").read_text(encoding="utf-8")
    )
    backend = next(iter(serve["backends"].values()))
    return (
        launch["producer_api_key"],
        backend["inference_gateway_token"],
        backend["inference_gateway_finalization_token"],
    )


def test_baseline_commit_dates_are_pinned_so_the_sha_is_content_addressed(
    tmp_path,
    monkeypatch,
):
    """The baseline sha is selection.baseline_version, so it has to be a function
    of the tree alone. Ambient GIT_*_DATE is set to two different values across
    the two compiles here precisely because the compiler used to inherit it (and
    otherwise the wall clock), which is how identical content produced two
    different shas and locked a recompile out of its preserved session.
    """
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2021-01-01T00:00:00+00:00")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2021-01-01T00:00:00+00:00")
    first = compiler.compile_harbor_task(
        _config(tmp_path / "a"), tmp_path / "a/out", vero_root=_VERO_ROOT
    )
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2023-06-06T06:06:06+00:00")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2023-06-06T06:06:06+00:00")
    second = compiler.compile_harbor_task(
        _config(tmp_path / "b"), tmp_path / "b/out", vero_root=_VERO_ROOT
    )

    assert _baseline_version(first) == _baseline_version(second)
    # And the pin is the documented constant, not just "the same twice".
    pinned = datetime.fromisoformat(compiler.BASELINE_COMMIT_DATE)
    expected = str(int(pinned.timestamp()))
    stamps = _git(
        first / "environment/agent-baseline", "show", "-s", "--format=%at %ct", "HEAD"
    )
    assert stamps == f"{expected} {expected}"


def test_recompile_into_the_same_directory_reuses_the_gateway_tokens(tmp_path):
    """Fresh tokens move the backend config digest the session manifest pins, so
    a recompile that re-mints them can never be brought up against durable state
    even though nothing about the evaluation changed.
    """
    output = tmp_path / "compiled"
    first = compiler.compile_harbor_task(
        _gateway_config(tmp_path), output, vero_root=_VERO_ROOT
    )
    before = _tokens(first)
    digests_before = json.loads(
        (first / "environment/gateway/config.json").read_text(encoding="utf-8")
    )["scopes"]
    serve_before = (first / "environment/sidecar/serve.json").read_text(
        encoding="utf-8"
    )

    second = compiler.compile_harbor_task(
        _gateway_config(tmp_path / "again"), output, vero_root=_VERO_ROOT
    )

    assert _tokens(second) == before
    # The three stay distinct, so the optimizer still cannot spend finalization's
    # reserved budget; reuse is not collapsing them into one token.
    assert len(set(before)) == 3
    digests_after = json.loads(
        (second / "environment/gateway/config.json").read_text(encoding="utf-8")
    )["scopes"]
    for scope in ("producer", "evaluation", "finalization"):
        assert (
            digests_after[scope]["token_sha256"]
            == digests_before[scope]["token_sha256"]
        )
    # With the baseline sha pinned too, the whole sidecar config is now
    # byte-identical across a recompile, which is what the manifest check wants.
    serve_after = (second / "environment/sidecar/serve.json").read_text(
        encoding="utf-8"
    )
    assert serve_after == serve_before


def test_a_different_output_directory_still_mints_its_own_tokens(tmp_path):
    """Reuse is keyed on the output directory, not global: two concurrent runs
    must not end up sharing one gateway credential.
    """
    first = compiler.compile_harbor_task(
        _gateway_config(tmp_path / "a"), tmp_path / "a/out", vero_root=_VERO_ROOT
    )
    second = compiler.compile_harbor_task(
        _gateway_config(tmp_path / "b"), tmp_path / "b/out", vero_root=_VERO_ROOT
    )
    assert set(_tokens(first)).isdisjoint(set(_tokens(second)))


def test_a_dead_compile_leaves_the_previous_tree_in_place(tmp_path, monkeypatch):
    """The failure that motivated staging: a compile dies partway, and what is
    left behind in the output directory has enough of a shape that the next step
    treats it as a finished task and fails much later, somewhere unrelated.
    """
    output = tmp_path / "compiled"
    good = compiler.compile_harbor_task(
        _config(tmp_path / "a"), output, vero_root=_VERO_ROOT
    )
    task_toml = (good / "task.toml").read_text(encoding="utf-8")
    # A successful compile leaves no staging directory behind.
    assert not (tmp_path / "compiled.partial").exists()

    def die(*arguments, **keywords):
        raise RuntimeError("compile died halfway")

    monkeypatch.setattr(compiler, "_write_cases", die)
    with pytest.raises(RuntimeError, match="compile died halfway"):
        compiler.compile_harbor_task(
            _config(tmp_path / "b"), output, vero_root=_VERO_ROOT
        )

    # The last complete compile is untouched, so a resume still has something to
    # come up against.
    assert (output / "task.toml").read_text(encoding="utf-8") == task_toml
    assert (output / "environment/sidecar/serve.json").is_file()
    # The wreckage is parked next door, and it is obviously unfinished.
    partial = tmp_path / "compiled.partial"
    assert partial.is_dir()
    assert not (partial / "task.toml").exists()


def test_a_later_compile_clears_the_wreckage_of_an_earlier_one(tmp_path):
    """A stale .partial must not be mistaken for state to resume from, and must
    not leak files that the new compile would never have written.
    """
    output = tmp_path / "compiled"
    stale = tmp_path / "compiled.partial"
    (stale / "environment").mkdir(parents=True)
    (stale / "leftover.txt").write_text("from a dead compile\n", encoding="utf-8")

    compiled = compiler.compile_harbor_task(
        _config(tmp_path), output, vero_root=_VERO_ROOT
    )
    assert (compiled / "task.toml").is_file()
    assert not (compiled / "leftover.txt").exists()
    assert not stale.exists()
