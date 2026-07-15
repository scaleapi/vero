"""Tests for the Workspace abstraction (GitWorkspace)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

from vero.sandbox import LocalSandbox
from vero.workspace.git import GitWorkspace


def _init_git_repo(path: Path) -> None:
    """Initialize a git repo with an initial commit."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    (path / "main.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@test",
            "commit",
            "-m",
            "init",
        ],
        cwd=path,
        capture_output=True,
        check=True,
    )


@pytest_asyncio.fixture
async def workspace(tmp_path):
    """Create a GitWorkspace backed by a LocalSandbox."""
    _init_git_repo(tmp_path)
    sandbox = LocalSandbox(root=tmp_path)
    return await GitWorkspace.from_path(sandbox, tmp_path)


class TestConstruction:
    @pytest.mark.asyncio
    async def test_from_path(self, workspace):
        assert workspace.root is not None
        assert workspace.project_path is not None
        assert workspace.name is not None

    @pytest.mark.asyncio
    async def test_sandbox_property(self, workspace):
        assert workspace.sandbox is not None
        assert isinstance(workspace.sandbox, LocalSandbox)

    @pytest.mark.asyncio
    async def test_from_path_subdir(self, tmp_path):
        """When project_path is a subdirectory, root is the repo root and project_path is the subdir."""
        _init_git_repo(tmp_path)
        subdir = tmp_path / "agents" / "my-agent"
        subdir.mkdir(parents=True)
        (subdir / "main.py").write_text("agent code\n")

        sandbox = LocalSandbox(root=tmp_path)
        ws = await GitWorkspace.from_path(sandbox, subdir)

        assert ws.root == str(tmp_path)
        assert ws.project_path == str(subdir)
        assert ws.root != ws.project_path

    @pytest.mark.asyncio
    async def test_not_a_git_repo(self, tmp_path):
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        sandbox = LocalSandbox(root=non_git)
        with pytest.raises(RuntimeError, match="Not a git repository"):
            await GitWorkspace.from_path(sandbox, non_git)


class TestVersioning:
    @pytest.mark.asyncio
    async def test_current_version(self, workspace):
        version = await workspace.current_version()
        assert len(version) == 40  # full SHA

    @pytest.mark.asyncio
    async def test_save(self, workspace):
        # Make a change
        await workspace.sandbox.write_file(
            str(Path(workspace.root) / "new.txt"), "new content\n"
        )
        assert await workspace.is_dirty()

        new_version = await workspace.save("add new.txt")
        assert len(new_version) == 40
        assert not await workspace.is_dirty()

    @pytest.mark.asyncio
    async def test_save_no_changes(self, workspace):
        v1 = await workspace.current_version()
        v2 = await workspace.save("nothing changed")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_restore(self, workspace):
        v1 = await workspace.current_version()

        # Make and save a change
        main_py = str(Path(workspace.root) / "main.py")
        await workspace.sandbox.write_file(main_py, "x = 999\n")
        await workspace.save("change main.py")
        v2 = await workspace.current_version()
        assert v1 != v2

        # Restore to v1
        await workspace.restore(v1)

        # main.py should be back to original
        content = await workspace.sandbox.read_file(main_py)
        assert content == "x = 1\n"


class TestSubdirProject:
    """Tests for when project_path is a subdirectory of the repo root."""

    @pytest_asyncio.fixture
    async def subdir_workspace(self, tmp_path):
        _init_git_repo(tmp_path)
        subdir = tmp_path / "agents" / "my-agent"
        subdir.mkdir(parents=True)
        (subdir / "agent.py").write_text("v1\n")
        # Commit the subdir
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@test",
                "commit",
                "-m",
                "add agent",
            ],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        sandbox = LocalSandbox(root=tmp_path)
        return await GitWorkspace.from_path(sandbox, subdir)

    @pytest.mark.asyncio
    async def test_save_scoped_to_project(self, subdir_workspace):
        """save() only stages files within project_path, not the whole repo."""
        ws = subdir_workspace
        repo_root = ws.root

        # Create a file outside the project subdir
        outside = str(Path(repo_root) / "outside.txt")
        await ws.sandbox.write_file(outside, "outside\n")

        # Create a file inside the project subdir
        inside = str(Path(ws.project_path) / "new.py")
        await ws.sandbox.write_file(inside, "new\n")

        v1 = await ws.current_version()
        await ws.save("save inside only")
        v2 = await ws.current_version()

        # A commit was made (inside file was staged)
        assert v1 != v2

        # The outside file should still be untracked
        result = await ws.sandbox.run(
            ["git", "status", "--porcelain", "outside.txt"], cwd=repo_root
        )
        assert "??" in result.stdout  # untracked

    @pytest.mark.asyncio
    async def test_is_dirty_scoped_to_project(self, subdir_workspace):
        """is_dirty() only reports changes within the project subdirectory."""
        ws = subdir_workspace
        assert not await ws.is_dirty()

        # Modify file OUTSIDE project — should NOT be dirty
        await ws.sandbox.write_file(str(Path(ws.root) / "outside.txt"), "unrelated\n")
        assert not await ws.is_dirty(), (
            "Changes outside project_path should not make workspace dirty"
        )

        # Modify file INSIDE project — should be dirty
        await ws.sandbox.write_file(str(Path(ws.project_path) / "agent.py"), "v2\n")
        assert await ws.is_dirty()


class TestDirtyState:
    @pytest.mark.asyncio
    async def test_canonical_validation_rejects_symlink_escape(
        self, workspace, tmp_path
    ):
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret", encoding="utf-8")
        link = Path(workspace.project_path) / "escape"
        link.symlink_to(outside, target_is_directory=True)

        with pytest.raises(PermissionError, match="Read access denied"):
            await workspace.validate_read_path("escape/secret.txt")
        with pytest.raises(PermissionError, match="Write access denied"):
            await workspace.validate_write_path("escape/new.txt")

    @pytest.mark.asyncio
    async def test_clean_initially(self, workspace):
        assert not await workspace.is_dirty()

    @pytest.mark.asyncio
    async def test_dirty_after_edit(self, workspace):
        await workspace.sandbox.write_file(
            str(Path(workspace.root) / "main.py"), "x = 2\n"
        )
        assert await workspace.is_dirty()

    @pytest.mark.asyncio
    async def test_clean_after_save(self, workspace):
        await workspace.sandbox.write_file(
            str(Path(workspace.root) / "main.py"), "x = 2\n"
        )
        await workspace.save("update")
        assert not await workspace.is_dirty()


class TestHistory:
    @pytest.mark.asyncio
    async def test_diff(self, workspace):
        v1 = await workspace.current_version()
        await workspace.sandbox.write_file(
            str(Path(workspace.root) / "main.py"), "x = 2\n"
        )
        await workspace.save("update")
        v2 = await workspace.current_version()

        diff = await workspace.diff(v1, v2)
        assert "-x = 1" in diff
        assert "+x = 2" in diff

    @pytest.mark.asyncio
    async def test_log(self, workspace):
        log = await workspace.log()
        assert "init" in log

    @pytest.mark.asyncio
    async def test_is_ancestor(self, workspace):
        v1 = await workspace.current_version()
        await workspace.sandbox.write_file(
            str(Path(workspace.root) / "main.py"), "x = 2\n"
        )
        await workspace.save("update")
        v2 = await workspace.current_version()

        assert await workspace.is_ancestor(v1, v2)
        assert not await workspace.is_ancestor(v2, v1)


class TestCopies:
    @pytest.mark.asyncio
    async def test_temp_copy(self, workspace):
        v1 = await workspace.current_version()

        async with workspace.temp_copy(from_version=v1) as copy_ws:
            assert copy_ws.root != workspace.root
            # The copy is a separate git worktree
            assert Path(copy_ws.root).exists()
            # main.py should exist in the copy
            assert (Path(copy_ws.root) / "main.py").exists()
            assert (Path(copy_ws.root) / "main.py").read_text() == "x = 1\n"

        # After context exit, the temp worktree is cleaned up
        assert not Path(copy_ws.root).exists()

        # Original is unchanged
        content = await workspace.sandbox.read_file(
            str(Path(workspace.root) / "main.py")
        )
        assert content == "x = 1\n"

    @pytest.mark.asyncio
    async def test_copies_preserve_subdirectory_project(self, tmp_path):
        _init_git_repo(tmp_path)
        subdir = tmp_path / "packages" / "target"
        subdir.mkdir(parents=True)
        (subdir / "program.py").write_text("value = 1\n")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@test",
                "commit",
                "-m",
                "add target",
            ],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        workspace = await GitWorkspace.from_path(LocalSandbox(root=tmp_path), subdir)

        async with workspace.temp_copy() as copy_workspace:
            relative = Path(copy_workspace.project_path).relative_to(
                copy_workspace.root
            )
            assert relative == Path("packages/target")
            assert Path(copy_workspace.project_path, "program.py").is_file()

        persistent = await workspace.copy(name="subproject-copy")
        try:
            relative = Path(persistent.project_path).relative_to(persistent.root)
            assert relative == Path("packages/target")
        finally:
            await persistent.destroy()

    @pytest.mark.asyncio
    async def test_persistent_copy_uses_hidden_ref_without_branch_leak(
        self,
        workspace,
    ):
        branches_before = await workspace._git(
            "for-each-ref", "--format=%(refname)", "refs/heads"
        )
        copied = await workspace.copy(name="candidate-copy")
        await copied.sandbox.write_file(
            str(Path(copied.root) / "candidate.txt"),
            "candidate\n",
        )
        version = await copied.save("candidate")
        await workspace.retain_version(
            version,
            "sessions/test/candidates/candidate",
        )
        await copied.destroy()

        branches_after = await workspace._git(
            "for-each-ref", "--format=%(refname)", "refs/heads"
        )
        retained = await workspace._git(
            "for-each-ref", "--format=%(objectname)", "refs/vero/sessions"
        )
        assert branches_after == branches_before
        assert version in retained.splitlines()
        assert not Path(copied.root).exists()

    @pytest.mark.asyncio
    async def test_copy_rolls_back_if_cancelled_after_git_creates_worktree(
        self,
        workspace,
        monkeypatch,
    ):
        original_run = workspace.sandbox.run
        injected = False

        async def cancel_after_add(command, **kwargs):
            nonlocal injected
            result = await original_run(command, **kwargs)
            if command[:3] == ["git", "worktree", "add"] and not injected:
                injected = True
                raise asyncio.CancelledError
            return result

        monkeypatch.setattr(workspace.sandbox, "run", cancel_after_add)

        with pytest.raises(asyncio.CancelledError):
            await workspace.copy(name="cancelled-copy")

        worktrees = await workspace._git("worktree", "list", "--porcelain")
        assert worktrees.count("worktree ") == 1
        assert not Path(workspace.root).parent.joinpath("cancelled-copy").exists()


class TestAtVersion:
    @pytest.mark.asyncio
    async def test_at_version(self, workspace):
        v1 = await workspace.current_version()

        # Make a change
        await workspace.sandbox.write_file(
            str(Path(workspace.root) / "main.py"), "x = 2\n"
        )
        await workspace.save("update")

        # Temporarily switch back to v1
        async with workspace.at(v1):
            content = await workspace.sandbox.read_file(
                str(Path(workspace.root) / "main.py")
            )
            assert content == "x = 1\n"

        # After exit, back to latest
        content = await workspace.sandbox.read_file(
            str(Path(workspace.root) / "main.py")
        )
        assert content == "x = 2\n"
