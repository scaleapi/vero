"""Test that uv run --with-editable correctly overlays a package version.

Creates two copies of a 'greeter' package with different behavior,
and a 'consumer' package that depends on greeter. Verifies that:
1. uv run --project consumer → uses the installed greeter (v1)
2. uv run --project consumer --with-editable greeter_v2 → uses greeter v2
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest


def _write_package(root: Path, name: str, version: str, code: str) -> Path:
    """Create a minimal uv package."""
    pkg_dir = root / name
    pkg_dir.mkdir(parents=True)
    src_dir = pkg_dir / "src" / name.replace("-", "_")
    src_dir.mkdir(parents=True)

    (pkg_dir / "pyproject.toml").write_text(textwrap.dedent(f"""\
        [project]
        name = "{name}"
        version = "{version}"
        requires-python = ">=3.11"

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{name.replace('-', '_')}"]
    """))

    (src_dir / "__init__.py").write_text(code)
    return pkg_dir


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with greeter_v1, greeter_v2, and consumer."""

    # greeter v1: returns "hello from v1"
    greeter_v1 = _write_package(
        tmp_path, "greeter", "1.0.0",
        'def greet(): return "hello from v1"\n',
    )

    # greeter v2: returns "hello from v2" (same package name, different code)
    greeter_v2_dir = tmp_path / "greeter_v2_src"
    greeter_v2_dir.mkdir()
    greeter_v2 = _write_package(
        greeter_v2_dir, "greeter", "2.0.0",
        'def greet(): return "hello from v2"\n',
    )

    # consumer: depends on greeter (path dep to v1)
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    (consumer / "pyproject.toml").write_text(textwrap.dedent(f"""\
        [project]
        name = "consumer"
        version = "0.1.0"
        requires-python = ">=3.11"
        dependencies = ["greeter"]

        [tool.uv.sources]
        greeter = {{ path = "{greeter_v1}" }}
    """))

    # Sync consumer to install greeter v1
    subprocess.run(
        ["uv", "sync", "--project", str(consumer)],
        capture_output=True,
        check=True,
    )

    return greeter_v1, greeter_v2, consumer


@pytest.mark.asyncio
async def test_with_editable_overlays_package(workspace):
    """Verify --with-editable replaces the installed package."""
    greeter_v1, greeter_v2, consumer = workspace

    script = "from greeter import greet; print(greet())"

    # Run normally → should get v1
    result_v1 = subprocess.run(
        ["uv", "run", "--project", str(consumer), "python", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result_v1.returncode == 0, f"v1 failed: {result_v1.stderr}"
    assert "hello from v1" in result_v1.stdout

    # Run with --with-editable pointing to v2 → should get v2
    result_v2 = subprocess.run(
        ["uv", "run", "--project", str(consumer), "--with-editable", str(greeter_v2), "python", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result_v2.returncode == 0, f"v2 failed: {result_v2.stderr}"
    assert "hello from v2" in result_v2.stdout

    # Run normally again → still v1 (no mutation)
    result_v1_again = subprocess.run(
        ["uv", "run", "--project", str(consumer), "python", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result_v1_again.returncode == 0
    assert "hello from v1" in result_v1_again.stdout
