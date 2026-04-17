"""Parser and loader for .veroaccess files.

.veroaccess files define filesystem access rules for Vero agents, similar to how
.gitignore defines ignore patterns. The file uses INI-style sections to group
patterns by access type.

File format:
    [exclude]
    tests/data/**
    **/__pycache__/**

    [read]
    tests/**
    .veroaccess

    [write]
    src/**

Rules are evaluated in order, with the last matching rule determining access level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from vero.filesystem import AccessRule, AccessType

from .constants import _DEFAULT_VEROACCESS_PATH, VEROACCESS_FILENAME

logger = logging.getLogger(__name__)

# Mandatory rule: .veroaccess itself must be read-only
# This prevents the agent from modifying its own access rules
_MANDATORY_RULES = [
    AccessRule(access_type=AccessType.READ, pattern=VEROACCESS_FILENAME),
]


class VeroAccessParseError(ValueError):
    """Raised when a .veroaccess file cannot be parsed."""

    pass


def parse_veroaccess(content: str) -> list[AccessRule]:
    """Parse .veroaccess file content into a list of AccessRule objects.

    Args:
        content: The raw content of a .veroaccess file.

    Returns:
        List of AccessRule objects in the order they appear in the file.

    Raises:
        VeroAccessParseError: If the file contains invalid syntax.
    """
    rules: list[AccessRule] = []
    current_section: AccessType | None = None

    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Section header: [exclude], [read], or [write]
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].lower()
            try:
                current_section = AccessType(section_name)
            except ValueError:
                raise VeroAccessParseError(
                    f"Line {line_num}: Invalid section '{section_name}'. "
                    f"Must be one of: exclude, read, write"
                )
            continue

        # Pattern line - must be under a section
        if current_section is None:
            raise VeroAccessParseError(
                f"Line {line_num}: Pattern '{line}' appears before any section header. "
                f"Add a section like [read] or [exclude] first."
            )

        rules.append(AccessRule(access_type=current_section, pattern=line))

    return rules


def _ensure_mandatory_rules(rules: list[AccessRule]) -> list[AccessRule]:
    """Append mandatory rules at the end to ensure they cannot be overridden.

    Since last-match wins, appending ensures these rules take precedence.
    Currently enforces that .veroaccess is always read-only.
    """
    return rules + _MANDATORY_RULES


def load_default_accesses() -> list[AccessRule]:
    """Load the default access rules from the bundled default.veroaccess.

    Returns:
        List of AccessRule objects from the default configuration.

    Raises:
        VeroAccessParseError: If the default file cannot be parsed.
    """
    content = _DEFAULT_VEROACCESS_PATH.read_text()
    rules = parse_veroaccess(content)
    return _ensure_mandatory_rules(rules)


def load_veroaccess(project_root: Path) -> list[AccessRule] | None:
    """Load .veroaccess from a project root directory.

    Args:
        project_root: Path to the project root directory.

    Returns:
        List of AccessRule objects if .veroaccess exists, None otherwise.

    Raises:
        VeroAccessParseError: If the file exists but cannot be parsed.
    """
    veroaccess_path = project_root / VEROACCESS_FILENAME
    if not veroaccess_path.exists():
        return None
    rules = parse_veroaccess(veroaccess_path.read_text())
    return _ensure_mandatory_rules(rules)


def resolve_filesystem_accesses(project_root: Path) -> list[AccessRule]:
    """Resolve filesystem accesses for a project.

    Checks for a .veroaccess file in the project root. If found, uses those rules.
    Otherwise, falls back to the default access rules.

    Args:
        project_root: Path to the project root directory.

    Returns:
        List of AccessRule objects to use for the project.
    """
    project_rules = load_veroaccess(project_root)
    if project_rules is not None:
        return project_rules
    return load_default_accesses()


# ---------------------------------------------------------------------------
# .veroaccess generation
# ---------------------------------------------------------------------------

# Directories that are always noise — never useful to an optimizer
_NOISE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    "dist",
    "build",
    ".venv",
    ".env",
    "node_modules",
    ".git",
}

# Directories that typically contain evaluation/ground-truth data
_DATA_DIRS = {"data", "datasets", "fixtures"}

# Directories containing tests
_TEST_DIRS = {"tests", "test"}

# Directories containing vero task definitions
_TASK_DIRS = {"vero_tasks"}

# Config files that should be read-only
_READ_ONLY_FILES = {"pyproject.toml", "setup.py", "setup.cfg", ".veroaccess"}


@dataclass
class _AccessEntry:
    """An access rule with an optional comment for generation."""

    access_type: AccessType
    pattern: str
    comment: str = ""


def _format_veroaccess(entries: list[_AccessEntry]) -> str:
    """Format access entries into .veroaccess file content.

    Groups entries by access type in the order: exclude, read, write.
    """
    grouped: dict[AccessType, list[_AccessEntry]] = {
        AccessType.EXCLUDE: [],
        AccessType.READ: [],
        AccessType.WRITE: [],
    }
    for entry in entries:
        grouped[entry.access_type].append(entry)

    lines = [
        "# Vero agent filesystem access rules",
        "# Last matching rule wins (like .gitignore)",
        "#",
        "# Sections:",
        "#   [exclude] - No access at all",
        "#   [read]    - Read-only access",
        "#   [write]   - Read and write access",
    ]

    for access_type in (AccessType.EXCLUDE, AccessType.READ, AccessType.WRITE):
        section_entries = grouped[access_type]
        if not section_entries:
            continue
        lines.append("")
        lines.append(f"[{access_type.value}]")
        prev_comment = None
        for entry in section_entries:
            if entry.comment and entry.comment != prev_comment:
                lines.append(f"# {entry.comment}")
                prev_comment = entry.comment
            lines.append(entry.pattern)

    lines.append("")  # trailing newline
    return "\n".join(lines)


def generate_veroaccess_auto(project_root: Path) -> str:
    """Scan project structure and generate a tailored .veroaccess file.

    Classification rules:
    - Known noise dirs (__pycache__, .git, etc.) -> exclude
    - Data dirs (data/, datasets/, fixtures/, tests/data/) -> exclude
    - Test dirs (tests/, test/) -> read
    - vero_tasks/ (anywhere) -> read
    - Config files (pyproject.toml, setup.py) -> read
    - .veroaccess -> read (mandatory)
    - Everything else -> write (implicit via default access)
    """
    entries: list[_AccessEntry] = []

    # Collect what actually exists at the top level
    existing_dirs: set[str] = set()
    existing_files: set[str] = set()
    for child in sorted(project_root.iterdir()):
        if child.is_dir():
            existing_dirs.add(child.name)
        elif child.is_file():
            existing_files.add(child.name)

    # Also check for nested vero_tasks
    has_nested_vero_tasks = False
    for p in project_root.rglob("vero_tasks"):
        if p.is_dir() and p.parent != project_root:
            has_nested_vero_tasks = True
            break

    # --- Exclude: noise directories (use ** patterns for nested ones) ---
    noise_found = existing_dirs & _NOISE_DIRS
    # Always add recursive patterns for dirs that can appear nested
    always_recursive = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for dirname in sorted(always_recursive):
        entries.append(_AccessEntry(AccessType.EXCLUDE, f"**/{dirname}", "Noise"))
        entries.append(_AccessEntry(AccessType.EXCLUDE, f"**/{dirname}/**", "Noise"))

    # Top-level only noise dirs
    for dirname in sorted(noise_found - always_recursive):
        entries.append(_AccessEntry(AccessType.EXCLUDE, dirname, "Noise"))
        entries.append(_AccessEntry(AccessType.EXCLUDE, f"{dirname}/**", "Noise"))

    # --- Exclude: data directories ---
    data_found = existing_dirs & _DATA_DIRS
    for dirname in sorted(data_found):
        entries.append(_AccessEntry(AccessType.EXCLUDE, dirname, "Data — prevent leakage"))
        entries.append(_AccessEntry(AccessType.EXCLUDE, f"{dirname}/**", "Data — prevent leakage"))

    # tests/data specifically
    test_dirs_found = existing_dirs & _TEST_DIRS
    for tdir in sorted(test_dirs_found):
        test_data = project_root / tdir / "data"
        if test_data.is_dir():
            entries.append(_AccessEntry(AccessType.EXCLUDE, f"{tdir}/data", "Test data — prevent leakage"))
            entries.append(_AccessEntry(AccessType.EXCLUDE, f"{tdir}/data/**", "Test data — prevent leakage"))

    # --- Read: test directories ---
    for tdir in sorted(test_dirs_found):
        entries.append(_AccessEntry(AccessType.READ, f"{tdir}/", "Test suite — read-only"))
        entries.append(_AccessEntry(AccessType.READ, f"{tdir}/**", "Test suite — read-only"))

    # --- Read: vero_tasks ---
    if "vero_tasks" in existing_dirs:
        entries.append(_AccessEntry(AccessType.READ, "vero_tasks", "Task definitions — protected"))
        entries.append(_AccessEntry(AccessType.READ, "vero_tasks/**", "Task definitions — protected"))
    if has_nested_vero_tasks:
        entries.append(_AccessEntry(AccessType.READ, "**/vero_tasks", "Task definitions — protected"))
        entries.append(_AccessEntry(AccessType.READ, "**/vero_tasks/**", "Task definitions — protected"))

    # --- Read: config files ---
    read_only_found = existing_files & _READ_ONLY_FILES
    for fname in sorted(read_only_found):
        entries.append(_AccessEntry(AccessType.READ, fname, "Config — read-only"))

    # Always include .veroaccess as read-only
    if ".veroaccess" not in read_only_found:
        entries.append(_AccessEntry(AccessType.READ, ".veroaccess", "Access rules — protected"))

    return _format_veroaccess(entries)
