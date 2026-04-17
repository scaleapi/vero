"""Tests for .veroaccess file parsing and loading."""

from pathlib import Path

import pytest
from vero.core.veroaccess import (
    VEROACCESS_FILENAME,
    VeroAccessParseError,
    generate_veroaccess_auto,
    load_default_accesses,
    load_veroaccess,
    parse_veroaccess,
    resolve_filesystem_accesses,
)
from vero.filesystem import AccessRule, AccessType


class TestParseVeroaccess:
    """Tests for parse_veroaccess function."""

    def test_empty_file(self):
        """Empty file returns empty rules list."""
        rules = parse_veroaccess("")
        assert rules == []

    def test_comments_only(self):
        """File with only comments returns empty rules list."""
        content = """
        # This is a comment
        # Another comment
        """
        rules = parse_veroaccess(content)
        assert rules == []

    def test_single_section(self):
        """Parse single section with patterns."""
        content = """
[read]
tests/**
*.md
"""
        rules = parse_veroaccess(content)
        assert len(rules) == 2
        assert rules[0] == AccessRule(access_type=AccessType.READ, pattern="tests/**")
        assert rules[1] == AccessRule(access_type=AccessType.READ, pattern="*.md")

    def test_multiple_sections(self):
        """Parse multiple sections."""
        content = """
[exclude]
__pycache__/**

[read]
tests/**

[write]
src/**
"""
        rules = parse_veroaccess(content)
        assert len(rules) == 3
        assert rules[0].access_type == AccessType.EXCLUDE
        assert rules[0].pattern == "__pycache__/**"
        assert rules[1].access_type == AccessType.READ
        assert rules[1].pattern == "tests/**"
        assert rules[2].access_type == AccessType.WRITE
        assert rules[2].pattern == "src/**"

    def test_section_can_appear_multiple_times(self):
        """Same section can appear multiple times (rules added in order)."""
        content = """
[read]
first/**

[exclude]
secret/**

[read]
second/**
"""
        rules = parse_veroaccess(content)
        assert len(rules) == 3
        assert rules[0] == AccessRule(access_type=AccessType.READ, pattern="first/**")
        assert rules[1] == AccessRule(access_type=AccessType.EXCLUDE, pattern="secret/**")
        assert rules[2] == AccessRule(access_type=AccessType.READ, pattern="second/**")

    def test_inline_comments_not_supported(self):
        """Inline comments are treated as part of the pattern."""
        content = """
[read]
tests/**  # this is a comment
"""
        rules = parse_veroaccess(content)
        # The whole line including comment becomes the pattern
        assert rules[0].pattern == "tests/**  # this is a comment"

    def test_whitespace_handling(self):
        """Leading/trailing whitespace is stripped."""
        content = """
[read]
   tests/**   
   src/**
"""
        rules = parse_veroaccess(content)
        assert rules[0].pattern == "tests/**"
        assert rules[1].pattern == "src/**"

    def test_case_insensitive_section_names(self):
        """Section names are case-insensitive."""
        content = """
[READ]
tests/**

[Exclude]
secret/**

[WRITE]
src/**
"""
        rules = parse_veroaccess(content)
        assert rules[0].access_type == AccessType.READ
        assert rules[1].access_type == AccessType.EXCLUDE
        assert rules[2].access_type == AccessType.WRITE

    def test_invalid_section_name_raises(self):
        """Invalid section name raises VeroAccessParseError."""
        content = """
[invalid]
pattern/**
"""
        with pytest.raises(VeroAccessParseError, match="Invalid section 'invalid'"):
            parse_veroaccess(content)

    def test_pattern_before_section_raises(self):
        """Pattern before any section raises VeroAccessParseError."""
        content = """
# Some comment
tests/**
[read]
src/**
"""
        with pytest.raises(VeroAccessParseError, match="appears before any section header"):
            parse_veroaccess(content)

    def test_error_includes_line_number(self):
        """Parse errors include line number."""
        content = """
[read]
valid/**
[badname]
pattern/**
"""
        with pytest.raises(VeroAccessParseError, match="Line 4"):
            parse_veroaccess(content)


class TestLoadDefaultAccesses:
    """Tests for load_default_accesses function."""

    def test_loads_default_file(self):
        """Default accesses can be loaded."""
        rules = load_default_accesses()
        assert len(rules) > 0
        assert all(isinstance(r, AccessRule) for r in rules)

    def test_default_excludes_pycache(self):
        """Default rules exclude __pycache__."""
        rules = load_default_accesses()
        patterns = [r.pattern for r in rules if r.access_type == AccessType.EXCLUDE]
        assert any("__pycache__" in p for p in patterns)

    def test_default_excludes_pytest_cache(self):
        """Default rules exclude .pytest_cache."""
        rules = load_default_accesses()
        patterns = [r.pattern for r in rules if r.access_type == AccessType.EXCLUDE]
        assert any(".pytest_cache" in p for p in patterns)

    def test_default_excludes_data_dirs(self):
        """Default rules exclude data directories."""
        rules = load_default_accesses()
        patterns = [r.pattern for r in rules if r.access_type == AccessType.EXCLUDE]
        assert any("tests/data" in p for p in patterns)
        assert any(p == "data" or p == "data/**" for p in patterns)

    def test_default_protects_tests(self):
        """Default rules make tests read-only."""
        rules = load_default_accesses()
        read_patterns = [r.pattern for r in rules if r.access_type == AccessType.READ]
        assert any("tests" in p for p in read_patterns)

    def test_default_protects_vero_tasks(self):
        """Default rules make vero_tasks read-only."""
        rules = load_default_accesses()
        read_patterns = [r.pattern for r in rules if r.access_type == AccessType.READ]
        assert any("vero_tasks" in p for p in read_patterns)

    def test_default_includes_mandatory_veroaccess_rule(self):
        """Default rules include mandatory .veroaccess read-only rule at the end."""
        rules = load_default_accesses()
        # Last rule should be the mandatory .veroaccess protection
        last_rule = rules[-1]
        assert last_rule.pattern == VEROACCESS_FILENAME
        assert last_rule.access_type == AccessType.READ


class TestLoadVeroaccess:
    """Tests for load_veroaccess function."""

    def test_returns_none_when_file_missing(self, tmp_path: Path):
        """Returns None when .veroaccess doesn't exist."""
        rules = load_veroaccess(tmp_path)
        assert rules is None

    def test_loads_existing_file(self, tmp_path: Path):
        """Loads rules from existing .veroaccess file."""
        veroaccess_content = """
[read]
docs/**

[exclude]
secret/**
"""
        (tmp_path / VEROACCESS_FILENAME).write_text(veroaccess_content)

        rules = load_veroaccess(tmp_path)
        assert rules is not None
        # Should have 2 from file + 1 mandatory
        assert len(rules) == 3
        assert rules[0] == AccessRule(access_type=AccessType.READ, pattern="docs/**")
        assert rules[1] == AccessRule(access_type=AccessType.EXCLUDE, pattern="secret/**")

    def test_appends_mandatory_rules(self, tmp_path: Path):
        """Mandatory rules are appended to loaded rules."""
        veroaccess_content = """
[write]
src/**
"""
        (tmp_path / VEROACCESS_FILENAME).write_text(veroaccess_content)

        rules = load_veroaccess(tmp_path)
        # Last rule should be the mandatory .veroaccess protection
        last_rule = rules[-1]
        assert last_rule.pattern == VEROACCESS_FILENAME
        assert last_rule.access_type == AccessType.READ

    def test_mandatory_rule_cannot_be_overridden(self, tmp_path: Path):
        """Even if user tries to make .veroaccess writable, mandatory rule wins."""
        veroaccess_content = """
[write]
.veroaccess
**
"""
        (tmp_path / VEROACCESS_FILENAME).write_text(veroaccess_content)

        rules = load_veroaccess(tmp_path)
        # Last matching rule wins, and mandatory rule is appended last
        last_rule = rules[-1]
        assert last_rule.pattern == VEROACCESS_FILENAME
        assert last_rule.access_type == AccessType.READ


class TestResolveFilesystemAccesses:
    """Tests for resolve_filesystem_accesses function."""

    def test_uses_project_file_when_present(self, tmp_path: Path):
        """Uses project .veroaccess when it exists."""
        veroaccess_content = """
[read]
custom/**
"""
        (tmp_path / VEROACCESS_FILENAME).write_text(veroaccess_content)

        rules = resolve_filesystem_accesses(tmp_path)
        patterns = [r.pattern for r in rules]
        assert "custom/**" in patterns

    def test_falls_back_to_default_when_missing(self, tmp_path: Path):
        """Falls back to default accesses when no .veroaccess exists."""
        rules = resolve_filesystem_accesses(tmp_path)
        # Should get default rules
        patterns = [r.pattern for r in rules]
        # Default rules include __pycache__ exclusions
        assert any("__pycache__" in p for p in patterns)

    def test_project_rules_dont_include_default_rules(self, tmp_path: Path):
        """Project .veroaccess completely replaces defaults (doesn't merge)."""
        veroaccess_content = """
[read]
only_this/**
"""
        (tmp_path / VEROACCESS_FILENAME).write_text(veroaccess_content)

        rules = resolve_filesystem_accesses(tmp_path)
        patterns = [r.pattern for r in rules]
        # Should NOT have default patterns like __pycache__ exclusions
        # (except .veroaccess which is mandatory)
        assert "only_this/**" in patterns
        # The only rules should be: only_this/** and .veroaccess (mandatory)
        assert len(rules) == 2


class TestIntegrationWithFilesystem:
    """Integration tests with Filesystem class."""

    def test_default_rules_work_with_filesystem(self, tmp_path: Path):
        """Default rules can be used with Filesystem."""
        from vero.filesystem import Filesystem

        # Create directory structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.pyc").touch()

        rules = load_default_accesses()
        fs = Filesystem(root=tmp_path, accesses=rules, default_access=AccessType.WRITE)

        # Source files should be writable (default)
        assert fs.can_write("src/main.py")

        # Tests should be read-only
        assert fs.can_read("tests/test_main.py")
        assert not fs.can_write("tests/test_main.py")

        # __pycache__ should be excluded
        assert not fs.can_read("__pycache__/cache.pyc")

    def test_project_veroaccess_works_with_filesystem(self, tmp_path: Path):
        """Project .veroaccess rules work with Filesystem."""
        from vero.filesystem import Filesystem

        # Create directory structure
        (tmp_path / "allowed").mkdir()
        (tmp_path / "allowed" / "file.py").touch()
        (tmp_path / "secret").mkdir()
        (tmp_path / "secret" / "file.py").touch()

        # Create project .veroaccess
        veroaccess_content = """
[write]
allowed/**

[exclude]
secret/**
"""
        (tmp_path / VEROACCESS_FILENAME).write_text(veroaccess_content)

        rules = resolve_filesystem_accesses(tmp_path)
        fs = Filesystem(root=tmp_path, accesses=rules, default_access=AccessType.EXCLUDE)

        # Allowed directory should be writable
        assert fs.can_write("allowed/file.py")

        # Secret directory should be excluded
        assert not fs.can_read("secret/file.py")

        # .veroaccess should be read-only (mandatory rule)
        assert fs.can_read(VEROACCESS_FILENAME)
        assert not fs.can_write(VEROACCESS_FILENAME)


class TestGenerateVeroAccessAuto:
    """Tests for generate_veroaccess_auto function."""

    def _parse_generated(self, content: str) -> list[AccessRule]:
        """Helper to parse generated content into rules."""
        return parse_veroaccess(content)

    def test_empty_project(self, tmp_path: Path):
        """Empty directory still generates valid .veroaccess with noise rules."""
        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)
        assert len(rules) > 0
        # Should have noise exclusions and .veroaccess protection
        patterns = [r.pattern for r in rules]
        assert "**/__pycache__" in patterns
        assert ".veroaccess" in patterns

    def test_detects_test_directory(self, tmp_path: Path):
        """Tests directory is detected and marked read-only."""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").touch()

        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)

        read_patterns = [r.pattern for r in rules if r.access_type == AccessType.READ]
        assert "tests/" in read_patterns or "tests/**" in read_patterns

    def test_detects_data_directory(self, tmp_path: Path):
        """Data directories are excluded."""
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "train.csv").touch()

        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)

        exclude_patterns = [r.pattern for r in rules if r.access_type == AccessType.EXCLUDE]
        assert "data" in exclude_patterns or "data/**" in exclude_patterns

    def test_detects_tests_data_subdirectory(self, tmp_path: Path):
        """tests/data/ is explicitly excluded."""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "data").mkdir()
        (tmp_path / "tests" / "data" / "fixture.json").touch()

        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)

        exclude_patterns = [r.pattern for r in rules if r.access_type == AccessType.EXCLUDE]
        assert "tests/data" in exclude_patterns or "tests/data/**" in exclude_patterns

    def test_detects_vero_tasks(self, tmp_path: Path):
        """vero_tasks directory is marked read-only."""
        (tmp_path / "vero_tasks").mkdir()
        (tmp_path / "vero_tasks" / "main.py").touch()

        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)

        read_patterns = [r.pattern for r in rules if r.access_type == AccessType.READ]
        assert "vero_tasks" in read_patterns or "vero_tasks/**" in read_patterns

    def test_detects_nested_vero_tasks(self, tmp_path: Path):
        """Nested vero_tasks (e.g. src/pkg/vero_tasks/) detected with ** pattern."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "pkg").mkdir()
        (tmp_path / "src" / "pkg" / "vero_tasks").mkdir()

        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)

        read_patterns = [r.pattern for r in rules if r.access_type == AccessType.READ]
        assert "**/vero_tasks" in read_patterns
        assert "**/vero_tasks/**" in read_patterns

    def test_detects_config_files(self, tmp_path: Path):
        """pyproject.toml is marked read-only."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)

        read_patterns = [r.pattern for r in rules if r.access_type == AccessType.READ]
        assert "pyproject.toml" in read_patterns

    def test_always_protects_veroaccess(self, tmp_path: Path):
        """Generated content always includes .veroaccess as read-only."""
        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)

        read_patterns = [r.pattern for r in rules if r.access_type == AccessType.READ]
        assert ".veroaccess" in read_patterns

    def test_noise_dirs_excluded(self, tmp_path: Path):
        """Noise directories like .venv and node_modules are excluded when present."""
        (tmp_path / ".venv").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "__pycache__").mkdir()

        content = generate_veroaccess_auto(tmp_path)
        rules = self._parse_generated(content)

        exclude_patterns = [r.pattern for r in rules if r.access_type == AccessType.EXCLUDE]
        assert "**/__pycache__" in exclude_patterns
        # .venv is hidden so not in top-level scan, but node_modules should be there
        assert "node_modules" in exclude_patterns or "node_modules/**" in exclude_patterns

    def test_generated_content_is_parseable(self, tmp_path: Path):
        """Generated .veroaccess can be round-tripped through parse_veroaccess."""
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "data").mkdir()
        (tmp_path / "data").mkdir()
        (tmp_path / "vero_tasks").mkdir()
        (tmp_path / "pyproject.toml").touch()

        content = generate_veroaccess_auto(tmp_path)
        # Should not raise
        rules = parse_veroaccess(content)
        assert len(rules) > 0

    def test_generated_content_works_with_filesystem(self, tmp_path: Path):
        """Generated rules work correctly with Filesystem class."""
        from vero.filesystem import Filesystem

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "agent.py").touch()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_agent.py").touch()
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "train.csv").touch()

        content = generate_veroaccess_auto(tmp_path)
        rules = parse_veroaccess(content)
        fs = Filesystem(root=tmp_path, accesses=rules, default_access=AccessType.WRITE)

        # src/ should be writable (no rule -> default write)
        assert fs.can_write("src/agent.py")

        # tests/ should be read-only
        assert fs.can_read("tests/test_agent.py")
        assert not fs.can_write("tests/test_agent.py")

        # data/ should be excluded
        assert not fs.can_read("data/train.csv")
