from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from vero.candidate import Candidate
from vero.candidate_repository import CandidateRepositoryError, GitCandidateRepository
from vero.sandbox import LocalSandbox, SandboxCapabilities
from vero.workspace import GitWorkspace


class OpaqueLocalSandbox(LocalSandbox):
    """Local execution with paths hidden to exercise remote bundle transfer."""

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(host_paths=False)

    def host_path(self, path: str) -> None:
        return None


def git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def initialize(path: Path) -> str:
    git(path, "init", "-b", "main")
    git(path, "add", "--all")
    git(
        path,
        "-c",
        "user.name=vero",
        "-c",
        "user.email=vero@localhost",
        "commit",
        "-m",
        "baseline",
    )
    return git(path, "rev-parse", "HEAD")


@pytest.mark.asyncio
async def test_git_repository_captures_remote_candidates_and_recreates_them(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    program = source / "program.sh"
    program.write_text("#!/bin/sh\necho baseline\n", encoding="utf-8")
    program.chmod(0o755)
    os.symlink("program.sh", source / "program-link")
    baseline_version = initialize(source)

    source_sandbox = OpaqueLocalSandbox(tmp_path)
    workspace = await GitWorkspace.from_path(source_sandbox, str(source))
    repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates",
        workspace=workspace,
    )
    baseline = Candidate.from_version(baseline_version)
    await repository.capture(baseline, workspace)

    program.write_text("#!/bin/sh\necho candidate\n", encoding="utf-8")
    candidate_version = await workspace.save("candidate")
    candidate = Candidate.from_version(
        candidate_version,
        candidate_id="candidate-1",
        parent_id=baseline.id,
        description="Improve the program",
        metadata={"producer_id": "test"},
    )
    await repository.capture(candidate, workspace)
    assert await repository.capture(candidate, workspace) == candidate

    shutil.rmtree(source)
    reopened = await GitCandidateRepository.open(repository.root)
    assert reopened.list() == (baseline, candidate)

    destination_sandbox = OpaqueLocalSandbox(tmp_path)
    checkout_root: Path | None = None
    async with reopened.checkout(
        candidate,
        sandbox=destination_sandbox,
        name="inspect",
    ) as checkout:
        checkout_root = Path(checkout.root)
        assert await checkout.current_version() == candidate.version
        assert (Path(checkout.project_path) / "program.sh").read_text() == (
            "#!/bin/sh\necho candidate\n"
        )
        mode = (Path(checkout.project_path) / "program.sh").stat().st_mode
        assert mode & stat.S_IXUSR
        assert (Path(checkout.project_path) / "program-link").is_symlink()
        assert not await checkout.is_dirty()
    assert checkout_root is not None
    assert not checkout_root.exists()


@pytest.mark.asyncio
async def test_git_repository_rejects_conflicting_candidate_identity(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.txt").write_text("baseline\n", encoding="utf-8")
    baseline_version = initialize(source)
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(source))
    repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates",
        workspace=workspace,
    )
    candidate = Candidate.from_version(baseline_version, candidate_id="candidate")
    await repository.capture(candidate, workspace)

    conflicting = candidate.model_copy(update={"description": "different"})
    with pytest.raises(ValueError, match="already stored with different data"):
        await repository.capture(conflicting, workspace)


@pytest.mark.asyncio
async def test_git_repository_preserves_nested_project_path(tmp_path: Path):
    source = tmp_path / "source"
    project = source / "packages" / "target"
    project.mkdir(parents=True)
    (project / "program.txt").write_text("baseline\n", encoding="utf-8")
    version = initialize(source)
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(project))
    repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates",
        workspace=workspace,
    )
    candidate = Candidate.from_version(version)
    await repository.capture(candidate, workspace)

    async with repository.checkout(candidate, sandbox=sandbox) as checkout:
        assert Path(checkout.project_path).relative_to(checkout.root) == Path(
            "packages/target"
        )
        assert (Path(checkout.project_path) / "program.txt").read_text() == (
            "baseline\n"
        )


@pytest.mark.asyncio
async def test_git_repository_fails_loudly_when_a_record_loses_its_ref(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.txt").write_text("baseline\n", encoding="utf-8")
    version = initialize(source)
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(source))
    repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates",
        workspace=workspace,
    )
    await repository.capture(Candidate.from_version(version), workspace)
    retained_ref = git(
        repository.repository_path,
        "for-each-ref",
        "--format=%(refname)",
        "refs/vero/candidates",
    )
    git(repository.repository_path, "update-ref", "-d", retained_ref)

    with pytest.raises(CandidateRepositoryError, match="missing Git object"):
        await GitCandidateRepository.open(repository.root)


@pytest.mark.asyncio
async def test_git_repository_rejects_candidates_that_track_agent_context(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.txt").write_text("baseline\n", encoding="utf-8")
    baseline_version = initialize(source)
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(source))
    repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates",
        workspace=workspace,
    )
    await repository.capture(Candidate.from_version(baseline_version), workspace)

    context = source / ".vero"
    context.mkdir()
    (context / "private.json").write_text('{"secret": true}\n', encoding="utf-8")
    git(source, "add", "-f", ".vero/private.json")
    git(
        source,
        "-c",
        "user.name=vero",
        "-c",
        "user.email=vero@localhost",
        "commit",
        "-m",
        "try to capture context",
    )
    candidate = Candidate.from_version(
        git(source, "rev-parse", "HEAD"),
        candidate_id="candidate-with-context",
    )

    with pytest.raises(CandidateRepositoryError, match="reserved agent context"):
        await repository.capture(candidate, workspace)
    assert repository.get(candidate.id) is None


@pytest.mark.asyncio
async def test_git_repository_materializes_visible_history_in_opaque_workspace(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    program = source / "program.txt"
    program.write_text("baseline\n", encoding="utf-8")
    baseline_version = initialize(source)
    sandbox = OpaqueLocalSandbox(tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(source))
    repository = await GitCandidateRepository.create(
        tmp_path / "session" / "candidates",
        workspace=workspace,
    )
    baseline = Candidate.from_version(baseline_version)
    await repository.capture(baseline, workspace)

    program.write_text("first\n", encoding="utf-8")
    first_version = await workspace.save("first candidate")
    first = Candidate.from_version(
        first_version,
        candidate_id="first",
        parent_id=baseline.id,
    )
    await repository.capture(first, workspace)

    git(source, "checkout", "--detach", baseline.version)
    program.write_text("sibling\n", encoding="utf-8")
    sibling_version = await workspace.save("sibling candidate")
    sibling = Candidate.from_version(
        sibling_version,
        candidate_id="sibling",
        parent_id=baseline.id,
    )
    await repository.capture(sibling, workspace)

    async with repository.checkout(
        baseline,
        sandbox=OpaqueLocalSandbox(tmp_path),
        name="history",
    ) as checkout:
        destination = f"{checkout.project_path}/.vero/candidates"
        await repository.materialize_agent_history(
            (baseline, first, sibling),
            workspace=checkout,
            destination=destination,
        )
        index = json.loads(
            (Path(destination) / "index.json").read_text(encoding="utf-8")
        )["candidates"]
        assert {item["candidate_id"] for item in index} == {
            baseline.id,
            first.id,
            sibling.id,
        }
        assert all(
            item["native_ref"].startswith("refs/vero/context/") for item in index
        )
        for item in index:
            assert (
                git(
                    Path(checkout.root),
                    "rev-parse",
                    "--verify",
                    f"{item['native_ref']}^{{commit}}",
                )
                == item["version"]
            )
            if item["candidate_id"] != baseline.id:
                patch = Path(destination) / item["parent_patch_path"]
                assert patch.is_file()
                assert "program.txt" in patch.read_text(encoding="utf-8")
        assert not await checkout.is_dirty()
