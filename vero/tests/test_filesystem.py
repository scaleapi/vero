"""Tests for the filesystem abstraction."""

from pathlib import Path

import pytest
from vero.filesystem import AccessRule, AccessType, Filesystem, WorkspaceAccessPolicy


class TestAccessType:
    """Tests for AccessType enum."""

    def test_can_read_exclude(self):
        assert not AccessType.EXCLUDE.can_read()

    def test_can_read_read(self):
        assert AccessType.READ.can_read()

    def test_can_read_write(self):
        assert AccessType.WRITE.can_read()

    def test_can_write_exclude(self):
        assert not AccessType.EXCLUDE.can_write()

    def test_can_write_read(self):
        assert not AccessType.READ.can_write()

    def test_can_write_write(self):
        assert AccessType.WRITE.can_write()


class TestAccessRule:
    """Tests for AccessRule with wcmatch patterns.

    Note: wcmatch with GLOBSTAR flag allows * to cross directory boundaries.
    Use more specific patterns when directory-limited matching is needed.
    """

    def test_simple_filename_match(self):
        rule = AccessRule(AccessType.READ, "file.txt")
        assert rule.matches("file.txt")
        assert not rule.matches("other.txt")
        # With GLOBSTAR, * can cross dirs, so exact match is strict
        assert not rule.matches("dir/file.txt")

    def test_single_star_wildcard(self):
        """Test * matches characters within a single path component."""
        rule = AccessRule(AccessType.READ, "*.py")
        assert rule.matches("test.py")
        assert rule.matches("file.py")
        assert not rule.matches("test.txt")
        # * does not cross directory boundaries
        assert not rule.matches("dir/test.py")

    def test_single_star_in_middle(self):
        rule = AccessRule(AccessType.READ, "test_*.py")
        assert rule.matches("test_file.py")
        assert rule.matches("test_something.py")
        assert not rule.matches("file.py")

    def test_double_star_globstar(self):
        """Test ** matches any path including directory separators."""
        rule = AccessRule(AccessType.READ, "**/*.py")
        # ** can match zero or more directories
        assert rule.matches("test.py")
        assert rule.matches("src/test.py")
        assert rule.matches("src/deep/nested/test.py")
        assert not rule.matches("src/test.txt")

    def test_double_star_at_start(self):
        rule = AccessRule(AccessType.READ, "**/test.py")
        # ** at start matches zero or more directories
        assert rule.matches("test.py")
        assert rule.matches("src/test.py")
        assert rule.matches("a/b/c/test.py")
        assert not rule.matches("test.txt")

    def test_double_star_at_end(self):
        rule = AccessRule(AccessType.READ, "src/**")
        assert rule.matches("src/file.py")
        assert rule.matches("src/deep/nested/file.py")
        assert not rule.matches("other/file.py")
        # should not match the directory itself
        assert not rule.matches("src")
        assert not rule.matches("src/")

    def test_double_star_in_middle(self):
        rule = AccessRule(AccessType.READ, "src/**/test.py")
        assert rule.matches("src/test.py")
        assert rule.matches("src/a/test.py")
        assert rule.matches("src/a/b/c/test.py")
        assert not rule.matches("other/test.py")
        # should not match the directory itself
        assert not rule.matches("src")
        assert not rule.matches("src/")

    def test_question_mark_single_char(self):
        """Test ? matches exactly one character."""
        rule = AccessRule(AccessType.READ, "file?.txt")
        assert rule.matches("file1.txt")
        assert rule.matches("fileA.txt")
        assert not rule.matches("file.txt")  # ? requires exactly one char
        assert not rule.matches("file12.txt")  # ? matches only one char

    def test_question_mark_multiple(self):
        rule = AccessRule(AccessType.READ, "test_??.py")
        assert rule.matches("test_01.py")
        assert rule.matches("test_AB.py")
        assert not rule.matches("test_1.py")  # needs exactly 2 chars
        assert not rule.matches("test_123.py")

    def test_bracket_character_class(self):
        """Test [...] character classes."""
        rule = AccessRule(AccessType.READ, "file[123].txt")
        assert rule.matches("file1.txt")
        assert rule.matches("file2.txt")
        assert rule.matches("file3.txt")
        assert not rule.matches("file4.txt")
        assert not rule.matches("file.txt")

    def test_bracket_range(self):
        rule = AccessRule(AccessType.READ, "test_[a-z].py")
        assert rule.matches("test_a.py")
        assert rule.matches("test_m.py")
        assert rule.matches("test_z.py")
        assert not rule.matches("test_A.py")  # uppercase not in range
        assert not rule.matches("test_1.py")

    def test_bracket_negation(self):
        """Test [!...] negated character classes."""
        rule = AccessRule(AccessType.READ, "file[!0-9].txt")
        assert rule.matches("filea.txt")
        assert rule.matches("fileX.txt")
        assert not rule.matches("file1.txt")
        assert not rule.matches("file9.txt")

    def test_brace_alternatives(self):
        """Test {a,b,c} brace expansion alternatives."""
        rule = AccessRule(AccessType.READ, "*.{py,js,ts}")
        assert rule.matches("file.py")
        assert rule.matches("file.js")
        assert rule.matches("file.ts")
        assert not rule.matches("file.txt")
        assert not rule.matches("file.go")

    def test_brace_with_directories(self):
        rule = AccessRule(AccessType.READ, "{src,lib}/**")
        assert rule.matches("src/file.py")
        assert rule.matches("lib/file.py")
        assert rule.matches("src/deep/file.py")
        assert not rule.matches("test/file.py")

    def test_nested_directories_pattern(self):
        """Test matching node_modules anywhere."""
        rule = AccessRule(AccessType.READ, "node_modules/**")
        assert rule.matches("node_modules/package/index.js")
        # For deep matching, use a different pattern or combine

    def test_nested_directories_deep(self):
        """Test **/dir/** for nested directory matching."""
        rule = AccessRule(AccessType.READ, "**/node_modules/**")
        # This pattern requires at least one dir before node_modules
        assert rule.matches("src/node_modules/package/file.js")
        # To also match root-level, need separate rule or use different pattern

    def test_hidden_files_pattern(self):
        rule = AccessRule(AccessType.EXCLUDE, ".git/**")
        assert rule.matches(".git/config")
        assert rule.matches(".git/objects/abc123")
        assert not rule.matches("git/config")

    def test_hidden_files_nested(self):
        """Test matching .git in subdirectories."""
        rule = AccessRule(AccessType.EXCLUDE, "**/.git/**")
        assert rule.matches("submodule/.git/config")
        # Root level .git needs separate rule

    def test_hidden_files_in_non_hidden_directory(self):
        """Test matching .git in subdirectories."""
        rule = AccessRule(AccessType.EXCLUDE, "**/tests/**")
        assert rule.matches("tests/.gitignore")

    def test_pycache_pattern(self):
        rule = AccessRule(AccessType.EXCLUDE, "__pycache__/**")
        assert rule.matches("__pycache__/module.cpython-39.pyc")
        assert not rule.matches("src/cache/file.py")

    def test_pycache_nested(self):
        """Test matching __pycache__ in subdirectories."""
        rule = AccessRule(AccessType.EXCLUDE, "**/__pycache__/**")
        assert rule.matches("src/__pycache__/test.pyc")

    def test_exact_directory_match(self):
        """Test matching directory names (use dir/** for contents)."""
        rule = AccessRule(AccessType.READ, "src")
        assert rule.matches("src")
        # For directory contents, use src/**
        rule2 = AccessRule(AccessType.READ, "src/**")
        assert rule2.matches("src/file.py")

    def test_complex_pattern(self):
        """Test combination of multiple glob features."""
        rule = AccessRule(AccessType.READ, "src/**/*.{py,pyi}")
        assert rule.matches("src/package/module.py")
        assert rule.matches("src/types/stubs.pyi")
        assert rule.matches("src/a/b/c.py")
        assert not rule.matches("src/readme.md")
        assert not rule.matches("tests/test.py")

    def test_forward_slash_edge_case(self):
        rule = AccessRule(AccessType.READ, "src")
        assert rule.matches("src")
        assert rule.matches("src/")

        rule = AccessRule(AccessType.READ, "src/")
        assert rule.matches("src")
        assert rule.matches("src/")


class TestWorkspaceAccessPolicy:
    def test_remote_root_does_not_need_to_exist_on_host(self):
        policy = WorkspaceAccessPolicy(
            root="/remote-only/target",
            default_access=AccessType.WRITE,
        )

        assert policy.validate_read("src/program.c") == (
            "/remote-only/target/src/program.c"
        )
        assert policy.validate_write("src/../README.md") == (
            "/remote-only/target/README.md"
        )

    def test_paths_outside_remote_root_are_excluded(self):
        policy = WorkspaceAccessPolicy(
            root="/remote-only/target",
            default_access=AccessType.WRITE,
        )

        assert not policy.can_read("../secret")
        with pytest.raises(PermissionError, match="Read access denied"):
            policy.validate_read("../secret")


class TestFilesystem:
    """Tests for Filesystem class."""

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory structure for testing."""
        # Create directories
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "deep").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / ".git").mkdir()
        # Create files
        (tmp_path / "README.md").touch()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "src" / "deep" / "nested.py").touch()
        (tmp_path / "tests" / "test_main.py").touch()
        return tmp_path

    def test_init_validates_root_dir_exists(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="Directory does not exist"):
            Filesystem(root=non_existent)

    def test_init_validates_root_dir_is_directory(self, tmp_path: Path):
        file_path = tmp_path / "file.txt"
        file_path.touch()
        with pytest.raises(ValueError, match="Path is not a directory"):
            Filesystem(root=file_path)

    def test_resolve_path_absolute(self, temp_dir: Path):
        fs = Filesystem(root=temp_dir)
        abs_path = temp_dir / "src" / "main.py"
        assert fs.resolve_path(abs_path) == abs_path

    def test_resolve_path_relative(self, temp_dir: Path):
        fs = Filesystem(root=temp_dir)
        assert fs.resolve_path("src/main.py") == temp_dir / "src" / "main.py"

    def test_get_relative_path_within_root(self, temp_dir: Path):
        fs = Filesystem(root=temp_dir)
        assert fs.get_relative_path("src/main.py") == "src/main.py"
        assert fs.get_relative_path(temp_dir / "src" / "main.py") == "src/main.py"

    def test_get_relative_path_outside_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        fs = Filesystem(root=root)
        outside_path = tmp_path / "other" / "file.txt"
        assert fs.get_relative_path(outside_path) is None

    def test_default_access_exclude(self, temp_dir: Path):
        """Default behavior is to exclude all paths."""
        fs = Filesystem(root=temp_dir)
        assert fs.get_access("src/main.py") == AccessType.EXCLUDE
        assert not fs.can_read("src/main.py")
        assert not fs.can_write("src/main.py")

    def test_default_access_override(self, temp_dir: Path):
        """Can override default access."""
        fs = Filesystem(root=temp_dir, default_access=AccessType.READ)
        assert fs.get_access("src/main.py") == AccessType.READ
        assert fs.can_read("src/main.py")
        assert not fs.can_write("src/main.py")

    def test_single_read_rule(self, temp_dir: Path):
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**")],
        )
        assert fs.can_read("src/main.py")
        assert not fs.can_write("src/main.py")
        assert fs.can_read("README.md")

    def test_single_write_rule(self, temp_dir: Path):
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.WRITE, "src/**")],
        )
        assert fs.can_read("src/main.py")
        assert fs.can_write("src/main.py")
        assert not fs.can_read("tests/test_main.py")

    def test_rule_order_last_wins(self, temp_dir: Path):
        """Last matching rule determines access."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[
                AccessRule(AccessType.READ, "**"),  # Allow read to all
                AccessRule(AccessType.EXCLUDE, ".git/**"),  # Exclude .git
            ],
        )
        assert fs.can_read("src/main.py")
        assert not fs.can_read(".git/config")

    def test_rule_order_complex(self, temp_dir: Path):
        """Complex rule ordering scenario."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[
                AccessRule(AccessType.READ, "**"),  # Read all
                AccessRule(AccessType.WRITE, "src/**"),  # Write to src/
            ],
        )
        assert fs.get_access("README.md") == AccessType.READ
        assert fs.get_access("src/main.py") == AccessType.WRITE
        assert fs.get_access("tests/test_main.py") == AccessType.READ

    def test_paths_outside_root_excluded(self, tmp_path: Path):
        """Paths outside root are always excluded."""
        root = tmp_path / "root"
        root.mkdir()
        fs = Filesystem(
            root=root,
            accesses=[AccessRule(AccessType.WRITE, "**")],  # Write to everything
        )
        outside_path = tmp_path / "other_dir" / "file.py"
        assert fs.get_access(outside_path) == AccessType.EXCLUDE
        assert not fs.can_read(outside_path)
        assert not fs.can_write(outside_path)

    def test_validate_read_success(self, temp_dir: Path):
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**")],
        )
        resolved = fs.validate_read("src/main.py")
        assert resolved == temp_dir / "src" / "main.py"

    def test_validate_read_denied(self, temp_dir: Path):
        fs = Filesystem(root=temp_dir)  # No access rules
        with pytest.raises(PermissionError, match="Read access denied"):
            fs.validate_read("src/main.py")

    def test_validate_write_success(self, temp_dir: Path):
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.WRITE, "**")],
        )
        resolved = fs.validate_write("src/main.py")
        assert resolved == temp_dir / "src" / "main.py"

    def test_validate_write_denied_exclude(self, temp_dir: Path):
        fs = Filesystem(root=temp_dir)  # No access rules
        with pytest.raises(PermissionError, match="Write access denied"):
            fs.validate_write("src/main.py")

    def test_validate_write_denied_read_only(self, temp_dir: Path):
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**")],
        )
        with pytest.raises(PermissionError, match="Write access denied"):
            fs.validate_write("src/main.py")


class TestFilesystemGlobPatterns:
    """Focused tests for glob pattern matching in filesystem context."""

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create directory structure for glob testing."""
        dirs = [
            "src",
            "src/utils",
            "src/models",
            "tests",
            "tests/unit",
            "tests/integration",
            ".git",
            ".git/objects",
            "node_modules",
            "node_modules/package",
            "__pycache__",
            "src/__pycache__",
        ]
        for d in dirs:
            (tmp_path / d).mkdir(parents=True, exist_ok=True)

        files = [
            "README.md",
            "setup.py",
            "config.yaml",
            "src/main.py",
            "src/app.py",
            "src/utils/helpers.py",
            "src/models/user.py",
            "tests/test_main.py",
            "tests/conftest.py",
            "tests/unit/test_utils.py",
            "tests/integration/test_api.py",
            ".git/config",
            ".git/objects/abc123",
            "node_modules/package/index.js",
            "__pycache__/main.cpython-39.pyc",
            "src/__pycache__/app.cpython-39.pyc",
        ]
        for f in files:
            (tmp_path / f).touch()

        return tmp_path

    def test_globstar_all_files(self, temp_dir: Path):
        """Test ** pattern matches all files."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**")],
        )
        assert fs.can_read("setup.py")
        assert fs.can_read("src/main.py")
        assert fs.can_read("src/utils/helpers.py")
        assert fs.can_read("README.md")

    def test_directory_subtree_access(self, temp_dir: Path):
        """Test dir/** pattern for directory subtree."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.WRITE, "src/**")],
        )
        assert fs.can_write("src/main.py")
        assert fs.can_write("src/utils/helpers.py")
        assert fs.can_write("src/models/user.py")
        assert not fs.can_write("tests/test_main.py")
        assert not fs.can_write("setup.py")

    def test_multiple_directories_brace(self, temp_dir: Path):
        """Test {a,b}/** for multiple directories."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "{src,tests}/**")],
        )
        assert fs.can_read("src/main.py")
        assert fs.can_read("tests/test_main.py")
        assert fs.can_read("tests/unit/test_utils.py")
        assert not fs.can_read("setup.py")
        assert not fs.can_read("node_modules/package/index.js")

    def test_exclude_hidden_directories(self, temp_dir: Path):
        """Test excluding hidden directories with .dir/** pattern."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, ".git/**"),
            ],
        )
        assert fs.can_read("src/main.py")
        assert fs.can_read("README.md")
        assert not fs.can_read(".git/config")
        assert not fs.can_read(".git/objects/abc123")

    def test_exclude_node_modules(self, temp_dir: Path):
        """Test excluding node_modules at root."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "node_modules/**"),
            ],
        )
        assert fs.can_read("src/main.py")
        assert not fs.can_read("node_modules/package/index.js")

    def test_exclude_pycache(self, temp_dir: Path):
        """Test excluding __pycache__ directories."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[
                AccessRule(AccessType.READ, "**"),
                AccessRule(AccessType.EXCLUDE, "__pycache__/**"),
                AccessRule(AccessType.EXCLUDE, "**/__pycache__/**"),
            ],
        )
        assert fs.can_read("src/main.py")
        assert not fs.can_read("__pycache__/main.cpython-39.pyc")
        assert not fs.can_read("src/__pycache__/app.cpython-39.pyc")

    def test_question_mark_single_char_files(self, temp_dir: Path):
        """Test ? for single character matching in filenames."""
        # Create specific test files
        (temp_dir / "v1.txt").touch()
        (temp_dir / "v2.txt").touch()
        (temp_dir / "v10.txt").touch()

        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "v?.txt")],
        )
        assert fs.can_read("v1.txt")
        assert fs.can_read("v2.txt")
        assert not fs.can_read("v10.txt")  # ? matches only one char

    def test_bracket_numeric_range(self, temp_dir: Path):
        """Test [0-9] numeric range matching."""
        # Create specific test files
        for i in range(12):
            (temp_dir / f"file{i}.txt").touch()

        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "file[0-5].txt")],
        )
        assert fs.can_read("file0.txt")
        assert fs.can_read("file5.txt")
        assert not fs.can_read("file6.txt")
        assert not fs.can_read("file10.txt")

    def test_multiple_extensions_brace(self, temp_dir: Path):
        """Test {ext1,ext2} for multiple extensions with **."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**/*.{py,yaml}")],
        )
        assert fs.can_read("setup.py")
        assert fs.can_read("config.yaml")
        assert fs.can_read("src/main.py")
        assert not fs.can_read("README.md")
        assert not fs.can_read("node_modules/package/index.js")

    def test_test_files_pattern(self, temp_dir: Path):
        """Test pattern for test files."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**/test_*.py")],
        )
        assert fs.can_read("tests/test_main.py")
        assert fs.can_read("tests/unit/test_utils.py")
        assert fs.can_read("tests/integration/test_api.py")
        assert not fs.can_read("tests/conftest.py")
        assert not fs.can_read("src/main.py")

    def test_conftest_specific(self, temp_dir: Path):
        """Test matching specific filename anywhere."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**/conftest.py")],
        )
        assert fs.can_read("tests/conftest.py")
        assert not fs.can_read("tests/test_main.py")

    def test_layered_permissions(self, temp_dir: Path):
        """Test realistic layered permission scenario.

        Note: Rules are applied in order, last match wins.
        To exclude __pycache__ within src/, the exclude rule must come AFTER the src/** rule.
        """
        fs = Filesystem(
            root=temp_dir,
            accesses=[
                # Base: read everything
                AccessRule(AccessType.READ, "**"),
                # Exclude sensitive/generated dirs at root
                AccessRule(AccessType.EXCLUDE, ".git/**"),
                AccessRule(AccessType.EXCLUDE, "node_modules/**"),
                AccessRule(AccessType.EXCLUDE, "__pycache__/**"),
                # Allow write to source code
                AccessRule(AccessType.WRITE, "src/**"),
                # Allow write to tests
                AccessRule(AccessType.WRITE, "tests/**"),
                # Exclude __pycache__ everywhere (must come after src/** to override)
                AccessRule(AccessType.EXCLUDE, "**/__pycache__/**"),
            ],
        )
        # Read-only
        assert fs.get_access("README.md") == AccessType.READ
        assert fs.get_access("setup.py") == AccessType.READ
        assert fs.get_access("config.yaml") == AccessType.READ

        # Excluded
        assert fs.get_access(".git/config") == AccessType.EXCLUDE
        assert fs.get_access("node_modules/package/index.js") == AccessType.EXCLUDE
        assert fs.get_access("__pycache__/main.cpython-39.pyc") == AccessType.EXCLUDE
        assert fs.get_access("src/__pycache__/app.cpython-39.pyc") == AccessType.EXCLUDE

        # Writable
        assert fs.get_access("src/main.py") == AccessType.WRITE
        assert fs.get_access("src/utils/helpers.py") == AccessType.WRITE
        assert fs.get_access("tests/test_main.py") == AccessType.WRITE
        assert fs.get_access("tests/unit/test_utils.py") == AccessType.WRITE


class TestFilesystemEdgeCases:
    """Edge case tests for filesystem."""

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "normal").mkdir()
        (tmp_path / "normal" / "file.txt").touch()
        return tmp_path

    def test_empty_accesses_list(self, temp_dir: Path):
        """Empty accesses uses default access."""
        fs = Filesystem(root=temp_dir, accesses=[])
        assert fs.get_access("normal/file.txt") == AccessType.EXCLUDE

        fs2 = Filesystem(root=temp_dir, accesses=[], default_access=AccessType.READ)
        assert fs2.get_access("normal/file.txt") == AccessType.READ

    def test_path_normalization(self, temp_dir: Path):
        """Paths are normalized for matching."""
        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**")],
        )
        # These should all resolve to the same file
        assert fs.can_read("normal/file.txt")
        assert fs.can_read("normal/../normal/file.txt")
        assert fs.can_read("./normal/file.txt")

    def test_resolve_path_with_dotdot(self, temp_dir: Path):
        """Test path resolution with parent directory references."""
        fs = Filesystem(root=temp_dir)
        # Going up and back down from root
        resolved = fs.resolve_path("normal/../normal/file.txt")
        assert resolved == temp_dir / "normal" / "file.txt"

    def test_special_chars_in_filename(self, temp_dir: Path):
        """Test files with special characters (not glob chars)."""
        special_file = temp_dir / "file with spaces.txt"
        special_file.touch()

        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**")],
        )
        assert fs.can_read("file with spaces.txt")

    def test_deeply_nested_path(self, temp_dir: Path):
        """Test very deeply nested paths."""
        deep_path = temp_dir / "a" / "b" / "c" / "d" / "e" / "f"
        deep_path.mkdir(parents=True)
        (deep_path / "file.py").touch()

        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**")],
        )
        assert fs.can_read("a/b/c/d/e/f/file.py")

    def test_root_level_file(self, temp_dir: Path):
        """Test files at root level."""
        (temp_dir / "root.txt").touch()

        fs = Filesystem(
            root=temp_dir,
            accesses=[AccessRule(AccessType.READ, "**")],
        )
        assert fs.can_read("root.txt")

    def test_overlapping_rules_precedence(self, temp_dir: Path):
        """Test that later rules override earlier ones."""
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "secret.py").touch()

        fs = Filesystem(
            root=temp_dir,
            accesses=[
                AccessRule(AccessType.WRITE, "src/**"),  # Write to all of src
                AccessRule(AccessType.READ, "src/secret.py"),  # But secret is read-only
            ],
        )
        assert fs.get_access("src/secret.py") == AccessType.READ  # Last rule wins
