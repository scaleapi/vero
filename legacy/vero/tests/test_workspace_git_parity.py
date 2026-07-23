"""Parity tests: GitWorkspace (sandbox.run) vs GitWorktree (GitPython).

Verifies that both implementations produce identical results for all
shared operations when run against the same git repository.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from git import Repo
from ref_git_worktree import GitWorktree
from vero.sandbox import LocalSandbox
from vero.workspace.git import GitWorkspace

pytestmark = pytest.mark.asyncio


@contextmanager
def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir).resolve()

        repo = Repo.init(test_dir)
        repo.config_writer().set_value("user", "name", "vero").release()
        repo.config_writer().set_value("user", "email", "vero@localhost").release()

        (test_dir / "README.md").write_text("# Test\n")
        (test_dir / "main.py").write_text("print('hello')\n")
        (test_dir / "src").mkdir()
        (test_dir / "src" / "app.py").write_text("# App\n")

        repo.index.add(["README.md", "main.py", "src/app.py"])
        repo.index.commit("Initial commit")

        if repo.active_branch.name != "main":
            repo.git.branch("-m", repo.active_branch.name, "main")

        yield repo, test_dir


class TestCurrentVersionParity:
    """current_version / current_commit must return the same hash."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_initial_commit(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        assert await ws.current_version() == wt.current_commit()

    async def test_after_new_commit(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        # Make a change and commit via workspace
        (test_dir / "new.txt").write_text("content\n")
        ws_commit = await ws.save("Add new.txt")

        # Both should see the same new commit
        assert ws_commit == wt.current_commit()
        assert await ws.current_version() == wt.current_commit()


class TestSaveParity:
    """save / commit_all must produce equivalent commits."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_save_stages_and_commits(self, git_repo):
        _, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        (test_dir / "new.txt").write_text("content\n")
        commit = await ws.save("Add file")

        assert len(commit) == 40
        assert not await ws.is_dirty()

    async def test_save_noop_when_clean(self, git_repo):
        _, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        before = await ws.current_version()
        after = await ws.save("No changes")
        assert before == after


class TestIsDirtyParity:
    """is_dirty / is_project_dirty must agree."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_clean_repo(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        assert await ws.is_dirty() == wt.is_project_dirty()
        assert await ws.is_dirty() is False

    async def test_modified_file(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        (test_dir / "main.py").write_text("modified\n")

        assert await ws.is_dirty() == wt.is_dirty()
        assert await ws.is_dirty() is True

    async def test_untracked_file(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        (test_dir / "untracked.txt").write_text("new\n")

        assert await ws.is_dirty() == wt.is_dirty()
        assert await ws.is_dirty() is True


class TestDiffParity:
    """diff must produce the same output."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_diff_between_commits(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        first_commit = wt.current_commit()

        (test_dir / "main.py").write_text("modified\n")
        wt.commit_all("Modify main.py")

        second_commit = wt.current_commit()

        wt_diff = wt.view_diff(from_commit=first_commit, to_commit=second_commit)
        ws_diff = await ws.diff(from_version=first_commit, to_version=second_commit)

        assert wt_diff == ws_diff


class TestLogParity:
    """log must produce equivalent output."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_log_format(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        # Both use the same format string
        wt_log = wt.repo.git.log("-n10", "--pretty=format:%h - %s (%cr) <%an>")
        ws_log = await ws.log(max_count=10)

        assert wt_log == ws_log

    async def test_log_since_commit(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        first_commit = wt.current_commit()

        (test_dir / "new.txt").write_text("content\n")
        wt.commit_all("Second commit")

        wt_log = wt.repo.git.log("-n10", "--pretty=format:%h - %s (%cr) <%an>", f"{first_commit}..HEAD")
        ws_log = await ws.log(max_count=10, since_version=first_commit)

        assert wt_log == ws_log


class TestIsAncestorParity:
    """is_ancestor must agree."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_ancestor(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        first = wt.current_commit()

        (test_dir / "new.txt").write_text("content\n")
        wt.commit_all("Second")

        second = wt.current_commit()

        wt_result = wt.repo.is_ancestor(first, second)
        ws_result = await ws.is_ancestor(first, second)
        assert wt_result == ws_result is True

        # Reverse should be False
        wt_result_rev = False
        try:
            wt_result_rev = wt.repo.is_ancestor(second, first)
        except Exception:
            pass
        ws_result_rev = await ws.is_ancestor(second, first)
        assert wt_result_rev == ws_result_rev is False

    async def test_non_ancestor(self, git_repo):
        _, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        result = await ws.is_ancestor("nonexistent", "alsonotreal")
        assert result is False


class TestRestoreParity:
    """restore / restore_to_commit must produce equivalent state."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_restore_to_previous(self, git_repo):
        _, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        first = await ws.current_version()

        (test_dir / "main.py").write_text("modified\n")
        await ws.save("Modify")

        assert (test_dir / "main.py").read_text() == "modified\n"

        new_commit = await ws.restore(first)
        assert len(new_commit) == 40

        # File should be restored
        assert (test_dir / "main.py").read_text() == "print('hello')\n"

        # History preserved (not a reset)
        log = await ws.log(max_count=10)
        assert "Restore" in log
        assert "Modify" in log


class TestBranchOperationsParity:
    """Branch operations must agree."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_current_branch(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        assert await ws.current_branch() == wt.current_branch() == "main"

    async def test_branch_exists(self, git_repo):
        repo, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        repo.git.branch("feature")

        assert await ws.branch_exists("main") == wt.branch_exists("main") is True
        assert await ws.branch_exists("feature") == wt.branch_exists("feature") is True
        assert await ws.branch_exists("nope") == wt.branch_exists("nope") is False

    async def test_get_head_commit(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        assert await ws.get_head_commit("main") == wt.get_head_commit("main")

    async def test_checkout_branch(self, git_repo):
        repo, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        await ws.checkout_branch("feature", create=True)
        assert await ws.current_branch() == "feature"
        assert wt.current_branch() == "feature"  # Both see the change


class TestAtParity:
    """at() context manager must switch and restore correctly."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_switch_and_restore(self, git_repo):
        _, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        first = await ws.current_version()

        (test_dir / "new.txt").write_text("content\n")
        await ws.save("Add file")

        original_branch = await ws.current_branch()
        original_commit = await ws.current_version()
        assert original_commit != first

        async with ws.at(first):
            assert await ws.current_version() == first

        # Restored
        assert await ws.current_branch() == original_branch
        assert await ws.current_version() == original_commit


class TestCopyParity:
    """copy / temp_copy must create functional isolated workspaces."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_copy_creates_worktree(self, git_repo):
        _, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        copy_ws = await ws.copy(name="test-copy")
        try:
            assert Path(copy_ws.root).exists()
            assert copy_ws.root != ws.root
            assert await copy_ws.current_version() == await ws.current_version()
        finally:
            await copy_ws.destroy()

    async def test_temp_copy_cleans_up(self, git_repo):
        _, test_dir = git_repo
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        async with ws.temp_copy() as temp_ws:
            temp_root = temp_ws.root
            assert Path(temp_root).exists()
            assert await temp_ws.current_version() == await ws.current_version()

        assert not Path(temp_root).exists()


class TestResolveRefParity:
    """resolve_ref must return the same hash as GitPython."""

    @pytest.fixture
    def git_repo(self):
        with temp_git_repo() as pair:
            yield pair

    async def test_resolve_head(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        wt_hash = wt.repo.commit("HEAD").hexsha
        ws_hash = await ws.resolve_ref("HEAD")
        assert wt_hash == ws_hash

    async def test_resolve_branch(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        wt_hash = wt.repo.commit("main").hexsha
        ws_hash = await ws.resolve_ref("main")
        assert wt_hash == ws_hash

    async def test_resolve_short_hash(self, git_repo):
        _, test_dir = git_repo
        wt = GitWorktree.from_local_path(worktree_path=test_dir)
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir))

        full_hash = wt.current_commit()
        short_hash = full_hash[:7]

        ws_resolved = await ws.resolve_ref(short_hash)
        assert ws_resolved == full_hash


class TestSubdirectoryProject:
    """Tests for when project_path is a subdirectory of the repo root.

    Verifies that is_dirty() and save() are properly scoped to the
    project subdirectory, ignoring changes elsewhere in the repo.
    """

    @pytest.fixture
    def git_repo_with_subdir(self):
        with temp_git_repo() as (repo, test_dir):
            # Add a subdirectory project
            agent_dir = test_dir / "agents" / "my-agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "agent.py").write_text("v1\n")
            repo.index.add([str(agent_dir / "agent.py")])
            repo.index.commit("Add agent subdir")
            yield repo, test_dir, agent_dir

    async def test_is_dirty_ignores_changes_outside_project(self, git_repo_with_subdir):
        _, test_dir, agent_dir = git_repo_with_subdir
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir), project_path=str(agent_dir))

        assert not await ws.is_dirty()

        # Change a file OUTSIDE the project subdir
        (test_dir / "main.py").write_text("changed outside\n")
        assert not await ws.is_dirty(), "Changes outside project_path should not be detected"

    async def test_is_dirty_detects_changes_inside_project(self, git_repo_with_subdir):
        _, test_dir, agent_dir = git_repo_with_subdir
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir), project_path=str(agent_dir))

        (agent_dir / "agent.py").write_text("v2\n")
        assert await ws.is_dirty()

    async def test_save_only_commits_project_files(self, git_repo_with_subdir):
        repo, test_dir, agent_dir = git_repo_with_subdir
        sandbox = LocalSandbox(root=test_dir)
        ws = GitWorkspace(sandbox=sandbox, root=str(test_dir), project_path=str(agent_dir))

        # Change files both inside and outside
        (test_dir / "README.md").write_text("changed readme\n")
        (agent_dir / "agent.py").write_text("v2\n")

        v1 = await ws.current_version()
        await ws.save("update agent")
        v2 = await ws.current_version()
        assert v1 != v2

        # README change should still be uncommitted
        result = await sandbox.run(["git", "status", "--porcelain", "README.md"], cwd=str(test_dir))
        assert result.stdout.strip() != "", "README.md should still be dirty (not staged by save)"

    async def test_from_path_resolves_subdir(self, git_repo_with_subdir):
        _, test_dir, agent_dir = git_repo_with_subdir
        sandbox = LocalSandbox(root=test_dir)
        ws = await GitWorkspace.from_path(sandbox, agent_dir)

        assert ws.root == str(test_dir)
        assert ws.project_path == str(agent_dir)
        assert ws.root != ws.project_path
