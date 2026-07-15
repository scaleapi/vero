from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from vero.evaluation import CommandBackend, CommandBackendConfig, MetricSelector, ObjectiveSpec
from vero.optimization import CommandCandidateProducer, CommandCandidateProducerConfig
from vero.runtime import create_optimization_session
from vero.sandbox import CommandResult, DockerSandbox
from vero.workspace import GitWorkspace


def _docker_available() -> bool:
    executable = shutil.which("docker")
    if executable is None:
        return False
    return (
        subprocess.run(
            [executable, "info"],
            capture_output=True,
            timeout=10,
            check=False,
        ).returncode
        == 0
    )


@pytest.mark.asyncio
async def test_docker_sandbox_owns_an_unmounted_container(monkeypatch):
    commands: list[list[str]] = []

    async def fake_host_command(command, *, timeout=30):
        commands.append(command)
        if command[1] == "run":
            return CommandResult("container-id", "", 0)
        return CommandResult("", "", 0)

    monkeypatch.setattr(
        DockerSandbox,
        "_host_command",
        staticmethod(fake_host_command),
    )

    sandbox = await DockerSandbox.create(
        image="example/image:locked",
        docker_executable="docker",
    )
    await sandbox.close()

    run_command = commands[0]
    assert run_command[:3] == ["docker", "run", "--detach"]
    assert "--volume" not in run_command
    assert "-v" not in run_command
    assert sandbox.host_path("/workspace/project") is None
    assert commands[-1] == ["docker", "rm", "--force", "container-id"]


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon is unavailable")
@pytest.mark.asyncio
async def test_generic_optimization_without_a_shared_filesystem(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.c").write_text(
        '#include <stdio.h>\nint main(void) { printf("1.0\\n"); return 0; }\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=target, check=True)
    subprocess.run(["git", "add", "--all"], cwd=target, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=target,
        check=True,
    )

    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "evaluate.sh").write_text(
        """#!/bin/sh
set -eu
workspace=$1
report=$2
artifacts=$3
cc "$workspace/program.c" -o "$artifacts/program"
score=$("$artifacts/program")
printf '{"schema_version":1,"status":"success","metrics":{"score":%s}}' "$score" > "$report"
""",
        encoding="utf-8",
    )
    producer_root = tmp_path / "producer"
    producer_root.mkdir()
    (producer_root / "improve.sh").write_text(
        "sed -i 's/1.0/2.0/' \"$1/program.c\"\n",
        encoding="utf-8",
    )

    sandbox = await DockerSandbox.create(
        image=os.environ.get("VERO_DOCKER_TEST_IMAGE", "gcc:14-bookworm")
    )
    try:
        remote_target = "/workspace/target"
        assert sandbox.host_path(remote_target) is None
        await sandbox.upload(target, remote_target)
        workspace = await GitWorkspace.from_path(sandbox, remote_target)
        backend = CommandBackend(
            CommandBackendConfig(
                harness_root=str(harness),
                command=[
                    "sh",
                    "{harness}/evaluate.sh",
                    "{workspace}",
                    "{report}",
                    "{artifacts}",
                ],
            )
        )
        producer = CommandCandidateProducer(
            CommandCandidateProducerConfig(
                root=str(producer_root),
                command=["sh", "{producer}/improve.sh", "{workspace}"],
            )
        )
        session = await create_optimization_session(
            workspace=workspace,
            session_dir=tmp_path / "session",
            backend_id="command",
            backend=backend,
            objective=ObjectiveSpec(
                selector=MetricSelector(metric="score"),
                direction="maximize",
            ),
            producers={"default": producer},
            max_candidates=1,
        )

        result = await asyncio.wait_for(session.run(), timeout=180)

        assert result.baseline.objective.value == 1.0
        assert result.best.objective.value == 2.0
        assert (session.session_dir / "database.json").is_file()
        assert await sandbox.read_file(f"{remote_target}/program.c") == (
            '#include <stdio.h>\nint main(void) { printf("1.0\\n"); return 0; }\n'
        )
    finally:
        await sandbox.close()
