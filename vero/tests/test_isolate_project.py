"""Tests for project isolation with dependency resolution."""

import subprocess

import pytest

from vero.evaluation.evaluator import _resolve_vero_dependency


@pytest.fixture
def project_with_vero_dep(tmp_path):
    """Create a fake project mimicking examples/gsm8k-agent structure.

    The relative path "../../" from examples/gsm8k-agent resolves to
    the repo root where vero's pyproject.toml lives.
    """
    # Mimic: vero/examples/gsm8k-agent (2 levels: ../.. = vero root)
    project_dir = tmp_path / "vero" / "examples" / "gsm8k-agent"
    project_dir.mkdir(parents=True)

    # ../../ from examples/gsm8k-agent = vero root
    vero_dir = tmp_path / "vero"
    (vero_dir / "pyproject.toml").write_text(
        '[project]\nname = "scale-vero"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
    )

    # Create the isolated copy
    isolated_dir = tmp_path / "isolated" / "gsm8k-agent"
    isolated_dir.mkdir(parents=True)

    return project_dir, isolated_dir, vero_dir


def test_resolve_vero_dependency_rewrites_path(project_with_vero_dep):
    """Vero path dep is resolved to absolute via uv add."""
    project_dir, isolated_dir, vero_dir = project_with_vero_dep

    (isolated_dir / "pyproject.toml").write_text(
        '[project]\nname = "gsm8k-agent"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        "dependencies = []\n\n"
        "[dependency-groups]\n"
        'dev = ["scale-vero"]\n\n'
        "[tool.uv.sources]\n"
        'scale-vero = { path = "../../", editable = true }\n'
    )

    # uv add will fail because the isolated dir isn't a real uv project,
    # but we can check that it's called with the right absolute path
    # by catching the subprocess error and inspecting args
    try:
        _resolve_vero_dependency(isolated_dir, project_dir)
    except subprocess.CalledProcessError as e:
        # uv add was called — check the command had the absolute path
        cmd = e.cmd if hasattr(e, "cmd") else []
        abs_vero = str(vero_dir)
        assert any(abs_vero in str(arg) for arg in cmd), f"Expected {abs_vero} in cmd: {cmd}"


def test_resolve_vero_dependency_errors_on_unknown_relative_path(project_with_vero_dep):
    """Non-vero relative path deps raise ValueError."""
    _, isolated_dir, _ = project_with_vero_dep

    (isolated_dir / "pyproject.toml").write_text(
        '[project]\nname = "gsm8k-agent"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        "dependencies = []\n\n"
        "[tool.uv.sources]\n"
        'some-other-pkg = { path = "../../other/pkg" }\n'
    )

    with pytest.raises(ValueError, match="Unsupported relative path dependency"):
        _resolve_vero_dependency(isolated_dir, isolated_dir)


def test_resolve_vero_dependency_no_op_without_path_deps(project_with_vero_dep):
    """No error when there are no path deps."""
    _, isolated_dir, _ = project_with_vero_dep

    (isolated_dir / "pyproject.toml").write_text(
        '[project]\nname = "gsm8k-agent"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        "dependencies = []\n\n"
        "[tool.uv.sources]\n"
        'harbor = { git = "https://github.com/example/harbor.git" }\n'
    )

    # Should be a no-op, no error
    _resolve_vero_dependency(isolated_dir, isolated_dir)


def test_resolve_vero_dependency_no_op_without_pyproject(tmp_path):
    """No error when pyproject.toml doesn't exist."""
    _resolve_vero_dependency(tmp_path, tmp_path)
