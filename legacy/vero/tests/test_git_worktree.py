"""Tests for GitWorktree abstraction."""

import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from git import Repo
from ref_git_worktree import GitWorktree


@contextmanager
def temp_git_repo():
    """Create a temporary git repository for testing.

    Yields a tuple of (repo, test_dir) where:
    - repo: The git.Repo instance
    - test_dir: Path to the repository root directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir).resolve()

        # Initialize git repo
        repo = Repo.init(test_dir)
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Create initial files and commit them
        (test_dir / "README.md").write_text("# Test Repo\n")
        (test_dir / "main.py").write_text("print('hello')\n")

        # Create subdirectory with files
        (test_dir / "src").mkdir()
        (test_dir / "src" / "app.py").write_text("# App code\n")

        # Add and commit all files
        repo.index.add(["README.md", "main.py", "src/app.py"])
        repo.index.commit("Initial commit")

        # Rename default branch to 'main' for consistent testing
        if repo.active_branch.name != "main":
            repo.git.branch("-m", repo.active_branch.name, "main")

        yield repo, test_dir


class TestGitWorktreeBasics:
    """Tests for basic GitWorktree functionality."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    def test_from_local_path_with_worktree_path(self, git_repo):
        """Test creating GitWorktree from local worktree path."""
        _, test_dir = git_repo
        worktree = GitWorktree.from_local_path(worktree_path=test_dir)

        assert worktree.worktree_path == test_dir
        assert worktree.project_path == test_dir

    def test_from_local_path_with_project_path(self, git_repo):
        """Test creating GitWorktree from project path."""
        _, test_dir = git_repo
        worktree = GitWorktree.from_local_path(project_path=test_dir)

        assert worktree.worktree_path == test_dir
        assert worktree.project_path == test_dir

    def test_from_local_path_with_subdir_project(self, git_repo):
        """Test creating GitWorktree with project in subdirectory."""
        _, test_dir = git_repo
        src_dir = test_dir / "src"
        worktree = GitWorktree.from_local_path(project_path=src_dir)

        assert worktree.worktree_path == test_dir
        assert worktree.project_path == src_dir

    def test_from_local_path_relative_project(self, git_repo):
        """Test creating GitWorktree with relative project path."""
        _, test_dir = git_repo
        worktree = GitWorktree.from_local_path(worktree_path=test_dir, project_path=Path("src"))

        assert worktree.worktree_path == test_dir
        assert worktree.project_path == test_dir / "src"

    def test_singleton_behavior(self, git_repo):
        """Test that GitWorktree reuses instances for same path."""
        _, test_dir = git_repo

        worktree1 = GitWorktree.from_local_path(worktree_path=test_dir)
        worktree2 = GitWorktree.from_local_path(worktree_path=test_dir)

        assert worktree1 is worktree2

    def test_singleton_different_project_path_raises(self, git_repo):
        """Test that GitWorktree raises if same worktree with different project_path."""
        repo, test_dir = git_repo

        # Must keep a reference to prevent garbage collection
        worktree = GitWorktree.from_local_path(worktree_path=test_dir, project_path=test_dir)

        with pytest.raises(ValueError, match="different project_path"):
            GitWorktree(repo=repo, project_path=test_dir / "src")

        # Keep reference alive until end of test
        assert worktree is not None


class TestGitWorktreeProperties:
    """Tests for GitWorktree properties."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    @pytest.fixture
    def worktree(self, git_repo):
        """Create a GitWorktree instance."""
        _, test_dir = git_repo
        return GitWorktree.from_local_path(worktree_path=test_dir)

    def test_main_branch(self, worktree):
        """Test main_branch property."""
        assert worktree.main_branch == "main"

    def test_main_branch_setter(self, worktree, git_repo):
        """Test setting main_branch."""
        repo, _ = git_repo
        # Create another branch
        repo.git.branch("develop")

        worktree.main_branch = "develop"
        assert worktree.main_branch == "develop"

    def test_main_branch_setter_invalid(self, worktree):
        """Test setting invalid main_branch raises."""
        with pytest.raises(AssertionError):
            worktree.main_branch = "nonexistent"

    def test_worktree_name(self, worktree, git_repo):
        """Test worktree_name property."""
        _, test_dir = git_repo
        assert worktree.worktree_name == test_dir.name

    def test_repo_name(self, worktree, git_repo):
        """Test repo_name property."""
        _, test_dir = git_repo
        assert worktree.repo_name == test_dir.name

    def test_project_relative_path(self, git_repo):
        """Test project_relative_path property."""
        _, test_dir = git_repo
        worktree = GitWorktree.from_local_path(
            worktree_path=test_dir, project_path=test_dir / "src"
        )
        assert worktree.project_relative_path == Path("src")

    def test_is_main_worktree(self, worktree):
        """Test is_main_worktree property."""
        assert worktree.is_main_worktree is True

    def test_remote_url_none(self, worktree):
        """Test remote_url returns None when no remote."""
        assert worktree.remote_url is None

    def test_http_url_none(self, worktree):
        """Test http_url returns None when no remote."""
        assert worktree.http_url is None


class TestGitWorktreeReadOperations:
    """Tests for GitWorktree read-only operations."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    @pytest.fixture
    def worktree(self, git_repo):
        """Create a GitWorktree instance."""
        _, test_dir = git_repo
        return GitWorktree.from_local_path(worktree_path=test_dir)

    def test_current_branch(self, worktree):
        """Test current_branch returns branch name."""
        assert worktree.current_branch() == "main"

    def test_current_commit(self, worktree):
        """Test current_commit returns commit hash."""
        commit = worktree.current_commit()
        assert isinstance(commit, str)
        assert len(commit) == 40  # SHA-1 hash

    def test_operates_on_full_repo(self, worktree):
        """Test operates_on_full_repo returns True for full repo."""
        assert worktree.operates_on_full_repo() is True

    def test_operates_on_full_repo_false(self, git_repo):
        """Test operates_on_full_repo returns False for subdirectory."""
        _, test_dir = git_repo
        worktree = GitWorktree.from_local_path(project_path=test_dir / "src")
        assert worktree.operates_on_full_repo() is False

    def test_list_branches(self, worktree, git_repo):
        """Test list_branches returns all branches."""
        repo, _ = git_repo
        repo.git.branch("feature")
        repo.git.branch("develop")

        branches = worktree.list_branches()
        assert "main" in branches
        assert "feature" in branches
        assert "develop" in branches

    def test_branch_exists(self, worktree, git_repo):
        """Test branch_exists checks branch existence."""
        repo, _ = git_repo
        repo.git.branch("feature")

        assert worktree.branch_exists("main") is True
        assert worktree.branch_exists("feature") is True
        assert worktree.branch_exists("nonexistent") is False

    def test_get_head_commit(self, worktree):
        """Test get_head_commit returns branch head."""
        commit = worktree.get_head_commit("main")
        assert commit == worktree.current_commit()

    def test_is_dirty_clean(self, worktree):
        """Test is_dirty returns False for clean repo."""
        assert worktree.is_dirty() is False

    def test_is_dirty_with_changes(self, worktree, git_repo):
        """Test is_dirty returns True with uncommitted changes."""
        _, test_dir = git_repo
        (test_dir / "new_file.txt").write_text("content\n")
        assert worktree.is_dirty() is True

    def test_is_project_dirty_clean(self, worktree):
        """Test is_project_dirty returns False for clean project."""
        assert worktree.is_project_dirty() is False

    def test_is_project_dirty_with_changes(self, git_repo):
        """Test is_project_dirty with modified files in a subdirectory project."""
        _, test_dir = git_repo
        # Use subdirectory as project to test project-specific dirty check
        worktree = GitWorktree.from_local_path(
            worktree_path=test_dir, project_path=test_dir / "src"
        )
        (test_dir / "src" / "app.py").write_text("modified\n")
        assert worktree.is_project_dirty() is True

    def test_list_project_modified_files(self, git_repo):
        """Test list_project_modified_files with a subdirectory project."""
        _, test_dir = git_repo
        # Use subdirectory as project to test project-specific file listing
        worktree = GitWorktree.from_local_path(
            worktree_path=test_dir, project_path=test_dir / "src"
        )
        (test_dir / "src" / "app.py").write_text("modified\n")
        (test_dir / "src" / "untracked.txt").write_text("new\n")

        modified = worktree.list_project_modified_files()
        assert "src/app.py" in modified
        assert "src/untracked.txt" in modified

    def test_list_worktrees(self, worktree, git_repo):
        """Test list_worktrees returns worktree info."""
        _, test_dir = git_repo
        worktrees = worktree.list_worktrees()

        assert test_dir in worktrees
        assert worktrees[test_dir]["branch_name"] == "main"
        assert worktrees[test_dir]["is_detached"] is False

    def test_as_dict(self, worktree):
        """Test as_dict returns correct dictionary."""
        d = worktree.as_dict()

        assert "worktree_path" in d
        assert "project_path" in d
        assert "branch" in d
        assert "commit" in d
        assert d["branch"] == "main"
        assert d["is_main_worktree"] is True

    def test_remote_exists_false(self, worktree):
        """Test remote_exists returns False without remote."""
        assert worktree.remote_exists() is False


class TestGitWorktreeWriteOperations:
    """Tests for GitWorktree write operations."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    @pytest.fixture
    def worktree(self, git_repo):
        """Create a GitWorktree instance."""
        repo, test_dir = git_repo
        return GitWorktree.from_local_path(worktree_path=test_dir)

    def test_checkout_branch_existing(self, worktree, git_repo):
        """Test checking out an existing branch."""
        repo, _ = git_repo
        repo.git.branch("feature")

        worktree.checkout_branch("feature")
        assert worktree.current_branch() == "feature"

    def test_checkout_branch_create(self, worktree):
        """Test creating and checking out a new branch."""
        worktree.checkout_branch("new-feature", from_="main", maybe_create=True)
        assert worktree.current_branch() == "new-feature"
        assert worktree.branch_exists("new-feature")

    def test_checkout_commit(self, worktree):
        """Test checking out a commit (detached HEAD)."""
        commit = worktree.current_commit()
        worktree.checkout_branch("temp", from_=commit, maybe_create=True)
        worktree.checkout_commit(commit)

        assert worktree.current_branch() is None
        assert worktree.current_commit() == commit

    def test_delete_branch(self, worktree, git_repo):
        """Test deleting a branch."""
        repo, _ = git_repo
        repo.git.branch("to-delete")
        assert worktree.branch_exists("to-delete")

        worktree.delete_branch("to-delete")
        assert worktree.branch_exists("to-delete") is False

    def test_commit_files(self, worktree, git_repo):
        """Test committing specific files."""
        _, test_dir = git_repo
        (test_dir / "new.txt").write_text("new content\n")

        commit_hash = worktree.commit_files(["new.txt"], "Add new file")

        assert len(commit_hash) == 40
        assert worktree.current_commit() == commit_hash
        assert worktree.is_dirty() is False

    def test_commit_files_empty_raises(self, worktree):
        """Test commit_files with empty list raises."""
        with pytest.raises(ValueError, match="No files"):
            worktree.commit_files([], "Empty commit")

    def test_commit_all(self, worktree, git_repo):
        """Test committing all changes."""
        _, test_dir = git_repo
        (test_dir / "new.txt").write_text("content\n")
        (test_dir / "main.py").write_text("modified\n")

        commit_hash = worktree.commit_all("Commit all changes")

        assert len(commit_hash) == 40
        assert worktree.is_dirty() is False

    def test_commit_all_project_only(self, git_repo):
        """Test commit_all with project_only flag."""
        repo, test_dir = git_repo
        worktree = GitWorktree.from_local_path(
            worktree_path=test_dir, project_path=test_dir / "src"
        )

        # Modify files inside and outside project
        (test_dir / "src" / "app.py").write_text("modified\n")
        (test_dir / "main.py").write_text("also modified\n")

        worktree.commit_all("Commit project only", project_only=True)

        # Project file committed, but root file still modified
        modified = worktree.list_project_modified_files()
        assert "src/app.py" not in modified

    def test_reset_to_commit(self, worktree, git_repo):
        """Test resetting to a previous commit."""
        _, test_dir = git_repo
        original_commit = worktree.current_commit()

        # Make a new commit
        (test_dir / "new.txt").write_text("content\n")
        worktree.commit_files(["new.txt"], "Add file")
        assert worktree.current_commit() != original_commit

        # Reset to original
        worktree.reset_to_commit(original_commit)
        assert worktree.current_commit() == original_commit
        assert not (test_dir / "new.txt").exists()

    def test_view_diff(self, worktree, git_repo):
        """Test viewing diff between commits."""
        _, test_dir = git_repo
        original_commit = worktree.current_commit()

        (test_dir / "main.py").write_text("modified\n")
        worktree.commit_all("Modify main.py")

        diff = worktree.view_diff(from_commit=original_commit)
        assert "main.py" in diff
        assert "modified" in diff


class TestGitWorktreeManagement:
    """Tests for GitWorktree worktree management."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    @pytest.fixture
    def worktree(self, git_repo):
        """Create a GitWorktree instance."""
        repo, test_dir = git_repo
        return GitWorktree.from_local_path(worktree_path=test_dir)

    def test_add_worktree_new_branch(self, worktree, git_repo):
        """Test adding a worktree with a new branch."""
        _, test_dir = git_repo
        new_path = test_dir.parent / "new-worktree"

        try:
            new_worktree = worktree.add_worktree(
                target_path=new_path, branch_name="feature", from_="main"
            )

            assert new_worktree.worktree_path == new_path
            assert new_worktree.current_branch() == "feature"
            assert new_worktree.is_main_worktree is False
        finally:
            # Cleanup
            if new_path.exists():
                new_worktree.remove_worktree()

    def test_add_worktree_existing_branch(self, worktree, git_repo):
        """Test adding a worktree with an existing branch."""
        repo, test_dir = git_repo
        repo.git.branch("existing")
        new_path = test_dir.parent / "existing-worktree"

        try:
            new_worktree = worktree.add_worktree(target_path=new_path, branch_name="existing")

            assert new_worktree.worktree_path == new_path
            assert new_worktree.current_branch() == "existing"
        finally:
            if new_path.exists():
                new_worktree.remove_worktree()

    def test_quick_spawn(self, worktree, git_repo):
        """Test quick_spawn creates worktree with random name."""
        _, test_dir = git_repo
        new_worktree = None

        try:
            new_worktree = worktree.quick_spawn()

            assert new_worktree.worktree_path.exists()
            assert new_worktree.worktree_path != worktree.worktree_path
            assert new_worktree.current_branch() is not None
        finally:
            if new_worktree:
                new_worktree.remove_worktree()

    def test_remove_worktree(self, worktree, git_repo):
        """Test removing a worktree."""
        _, test_dir = git_repo
        new_path = test_dir.parent / "to-remove"

        new_worktree = worktree.add_worktree(
            target_path=new_path, branch_name="remove-me", from_="main"
        )
        assert new_path.exists()

        result = new_worktree.remove_worktree()
        assert result is True
        assert not new_path.exists()

    def test_remove_main_worktree_raises(self, worktree):
        """Test removing main worktree raises."""
        with pytest.raises(AssertionError, match="main worktree"):
            worktree.remove_worktree()

    def test_get_random_worktree_name(self, worktree, git_repo):
        """Test get_random_worktree_name generates unique names."""
        _, test_dir = git_repo
        name1 = worktree.get_random_worktree_name()
        name2 = worktree.get_random_worktree_name()

        assert name1 != name2
        assert test_dir.name in name1

    def test_get_random_worktree_path(self, worktree, git_repo):
        """Test get_random_worktree_path returns valid path."""
        _, test_dir = git_repo
        path = worktree.get_random_worktree_path()

        assert path.parent == test_dir.parent
        assert test_dir.name in path.name


class TestGitWorktreeContextManagers:
    """Tests for GitWorktree context managers."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    @pytest.fixture
    def worktree(self, git_repo):
        """Create a GitWorktree instance."""
        repo, test_dir = git_repo
        return GitWorktree.from_local_path(worktree_path=test_dir)

    @pytest.mark.asyncio
    async def test_switch_to_commit(self, worktree, git_repo):
        """Test switch_to_commit context manager."""
        _, test_dir = git_repo

        # Create a new commit
        (test_dir / "new.txt").write_text("content\n")
        worktree.commit_files(["new.txt"], "Add new.txt")

        original_branch = worktree.current_branch()
        original_commit = worktree.current_commit()

        # Get a previous commit
        previous_commit = worktree.repo.git.rev_parse("HEAD~1")

        async with worktree.switch_to_commit(previous_commit):
            assert worktree.current_commit() == previous_commit
            assert worktree.current_branch() is None  # Detached HEAD

        # Restored to original state
        assert worktree.current_branch() == original_branch
        assert worktree.current_commit() == original_commit

    @pytest.mark.asyncio
    async def test_in_new_worktree(self, worktree, git_repo):
        """Test in_new_worktree context manager."""
        _, test_dir = git_repo

        async with worktree.in_new_worktree(branch_name="temp-branch") as temp_wt:
            assert temp_wt.worktree_path != worktree.worktree_path
            assert temp_wt.current_branch() == "temp-branch"
            assert temp_wt.worktree_path.exists()

        # Worktree removed after context exits
        assert not temp_wt.worktree_path.exists()

    @pytest.mark.asyncio
    async def test_in_new_worktree_random_path(self, worktree):
        """Test in_new_worktree with random path."""
        async with worktree.in_new_worktree() as temp_wt:
            path = temp_wt.worktree_path
            assert path.exists()

        assert not path.exists()

    @pytest.mark.asyncio
    async def test_locked(self, worktree):
        """Test locked context manager."""
        async with worktree.locked(caller="test"):
            # Should be able to acquire lock
            assert worktree._lock.locked()

        assert not worktree._lock.locked()


class TestGitWorktreeRegistry:
    """Tests for GitWorktree registry/singleton management."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    def test_get_existing_instance(self, git_repo):
        """Test GitWorktree.get returns existing instance."""
        repo, test_dir = git_repo
        worktree = GitWorktree.from_local_path(worktree_path=test_dir)

        retrieved = GitWorktree.get(test_dir)
        assert retrieved is worktree

    def test_get_nonexistent_returns_none(self, git_repo):
        """Test GitWorktree.get returns None for unknown path."""
        _, test_dir = git_repo
        result = GitWorktree.get(test_dir / "nonexistent")
        assert result is None

    def test_remove_from_registry(self, git_repo):
        """Test GitWorktree.remove removes from registry."""
        repo, test_dir = git_repo
        # Must keep a reference to prevent garbage collection
        worktree = GitWorktree.from_local_path(worktree_path=test_dir)
        assert GitWorktree.get(test_dir) is worktree

        result = GitWorktree.remove(test_dir)
        assert result is True
        assert GitWorktree.get(test_dir) is None

    def test_remove_nonexistent_returns_false(self, git_repo):
        """Test GitWorktree.remove returns False for unknown path."""
        _, test_dir = git_repo
        result = GitWorktree.remove(test_dir / "nonexistent")
        assert result is False


class TestGitWorktreeInferMainBranch:
    """Tests for main branch inference."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    def test_infer_main_branch_main(self, git_repo):
        """Test infer_main_branch detects 'main'."""
        repo, _ = git_repo
        assert GitWorktree.infer_main_branch(repo) == "main"

    def test_infer_main_branch_master(self, git_repo):
        """Test infer_main_branch detects 'master'."""
        repo, _ = git_repo
        repo.git.branch("-m", "main", "master")
        assert GitWorktree.infer_main_branch(repo) == "master"

    def test_infer_main_branch_develop(self, git_repo):
        """Test infer_main_branch detects 'develop'."""
        repo, _ = git_repo
        repo.git.branch("-m", "main", "develop")
        assert GitWorktree.infer_main_branch(repo) == "develop"


class TestGitWorktreeValidation:
    """Tests for GitWorktree validation."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository."""
        with temp_git_repo() as (repo, test_dir):
            yield repo, test_dir

    def test_project_path_outside_worktree_raises(self, git_repo):
        """Test that project path outside worktree raises ValueError."""
        repo, test_dir = git_repo

        with pytest.raises(ValueError, match="not a subfolder"):
            GitWorktree(repo=repo, project_path=test_dir.parent)

    def test_project_path_not_exists_raises(self, git_repo):
        """Test that non-existent project path raises AssertionError."""
        repo, test_dir = git_repo

        with pytest.raises(AssertionError, match="does not exist"):
            GitWorktree(repo=repo, project_path=test_dir / "nonexistent")

    def test_project_path_file_raises(self, git_repo):
        """Test that file as project path raises AssertionError."""
        repo, test_dir = git_repo

        with pytest.raises(AssertionError, match="not a directory"):
            GitWorktree(repo=repo, project_path=test_dir / "main.py")

    def test_from_local_path_no_paths_raises(self, git_repo):
        """Test from_local_path with no paths raises ValueError."""
        with pytest.raises(ValueError, match="project_path is required"):
            GitWorktree.from_local_path()
