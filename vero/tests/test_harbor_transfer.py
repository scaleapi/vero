"""Integration test for EvaluationSidecar._transfer_commit (real git repos).

Validates that a commit is fetched from the (untrusted) mounted agent repo into
the sidecar's own repo and resolved to its sha — the one server.py piece
that can't be unit-tested with mocks.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vero.harbor.server import EvaluationSidecar
from vero.sandbox import LocalSandbox
from vero.workspace.git import GitWorkspace


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(path: Path, content: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    (path / "f.txt").write_text(content)
    _git(path, "add", "f.txt")
    _git(path, "commit", "-q", "-m", "c")
    return _git(path, "rev-parse", "HEAD")


async def _sidecar_for(sidecar_repo: Path, agent_repo: Path, tmp_path: Path):
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(sidecar_repo))
    engine = MagicMock()
    engine.evaluator.workspace = workspace
    return EvaluationSidecar(
        engine=engine,
        split_accesses=[],
        agent_repo_path=agent_repo,
        agent_volume=tmp_path / "av",
        admin_volume=tmp_path / "adv",
    )


@pytest.mark.asyncio
async def test_transfer_fetches_agent_head_into_sidecar_repo(tmp_path):
    agent_repo = tmp_path / "agent"
    sidecar_repo = tmp_path / "sidecar"
    agent_head = _init_repo(agent_repo, "agent work")
    _init_repo(sidecar_repo, "sidecar base")

    sidecar = await _sidecar_for(sidecar_repo, agent_repo, tmp_path)
    sha = await sidecar._transfer_commit(None)  # default = agent HEAD

    assert sha == agent_head
    # the fetched commit object now lives in the sidecar's own repo (tamper-evident copy)
    assert (
        subprocess.run(
            ["git", "-C", str(sidecar_repo), "cat-file", "-e", sha], capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.asyncio
async def test_transfer_explicit_ref(tmp_path):
    agent_repo = tmp_path / "agent"
    sidecar_repo = tmp_path / "sidecar"
    _init_repo(agent_repo, "first")
    # a second commit; transfer the first by explicit sha
    first = _git(agent_repo, "rev-parse", "HEAD")
    (agent_repo / "f.txt").write_text("second")
    _git(agent_repo, "commit", "-aqm", "second")
    _init_repo(sidecar_repo, "sidecar base")

    sidecar = await _sidecar_for(sidecar_repo, agent_repo, tmp_path)
    sha = await sidecar._transfer_commit(first)
    assert sha == first
