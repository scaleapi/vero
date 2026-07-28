from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from vero.cli import main


def initialize_repository(path: Path) -> str:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "--all"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.skipif(shutil.which("cc") is None, reason="a C compiler is required")
def test_declarative_c_example_optimizes_a_non_python_target(tmp_path: Path):
    source = Path(__file__).parents[1] / "examples" / "c-matmul"
    example = tmp_path / "c-matmul"
    shutil.copytree(source, example, ignore=shutil.ignore_patterns("__pycache__"))
    baseline_source = (example / "target" / "matmul.c").read_text(encoding="utf-8")
    baseline_version = initialize_repository(example / "target")
    config = example / "vero.toml"
    runner = CliRunner()

    evaluated = runner.invoke(main, ["evaluate", "--config", str(config)])
    assert evaluated.exit_code == 0, evaluated.output
    assert f"Baseline: {baseline_version}" in evaluated.output

    optimized = runner.invoke(main, ["run", "--config", str(config)])
    assert optimized.exit_code == 0, optimized.output
    assert "Best: no feasible candidate" not in optimized.output
    assert f"Best: {baseline_version}" not in optimized.output
    assert (example / "target" / "matmul.c").read_text(
        encoding="utf-8"
    ) == baseline_source

    database = json.loads(
        (example / ".vero" / "session" / "database.json").read_text(encoding="utf-8")
    )
    records = list(database["evaluations"].values())
    assert len(records) == 2
    assert all(record["objective"]["feasible"] for record in records)
    assert min(record["objective"]["value"] for record in records) < max(
        record["objective"]["value"] for record in records
    )

    worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=example / "target",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert worktrees.count("worktree ") == 1
