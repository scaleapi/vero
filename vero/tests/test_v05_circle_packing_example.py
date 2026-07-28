from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from vero.cli import main


def test_circle_packing_harness_applies_request_seed(tmp_path: Path):
    example = Path(__file__).parents[1] / "examples" / "circle-packing"
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "packing.py").write_text(
        """\
import random

def run_packing():
    centers = []
    for row in range(6):
        for column in range(5):
            if len(centers) == 26:
                break
            centers.append([0.1 + 0.2 * column, 0.1 + 0.16 * row])
    radii = [random.uniform(0.01, 0.015) for _ in centers]
    return centers, radii, sum(radii)
""",
        encoding="utf-8",
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"schema_version": 1, "request": {"seed": 12345}}),
        encoding="utf-8",
    )

    scores = []
    layouts = []
    for attempt in range(2):
        report = tmp_path / f"report-{attempt}.json"
        artifacts = tmp_path / f"artifacts-{attempt}"
        completed = subprocess.run(
            [
                sys.executable,
                str(example / "harness" / "evaluate.py"),
                "--workspace",
                str(workspace),
                "--request",
                str(request),
                "--report",
                str(report),
                "--artifacts",
                str(artifacts),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stderr == ""
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["metrics"]["valid"] == 1.0
        scores.append(payload["metrics"]["sum_radii"])
        layouts.append(
            (artifacts / "circle-packing" / "layout.json").read_text(
                encoding="utf-8"
            )
        )

    assert scores[0] == scores[1]
    assert layouts[0] == layouts[1]


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


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required")
def test_circle_packing_example_evaluates_and_preserves_artifacts(
    tmp_path: Path, monkeypatch
):
    source = Path(__file__).parents[1] / "examples" / "circle-packing"
    example = tmp_path / "circle-packing"
    shutil.copytree(
        source,
        example,
        ignore=shutil.ignore_patterns(".git", ".vero", ".evals", ".venv", "__pycache__"),
    )
    baseline_version = initialize_repository(example / "target")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))

    result = CliRunner().invoke(
        main,
        ["evaluate", "--config", str(example / "vero.toml")],
    )

    assert result.exit_code == 0, result.output
    assert f"Baseline: {baseline_version}" in result.output
    database = json.loads(
        (example / ".vero" / "session" / "database.json").read_text(
            encoding="utf-8"
        )
    )
    records = list(database["evaluations"].values())
    assert len(records) == 2
    assert {record["request"]["evaluation_set"]["name"] for record in records} == {
        "development",
        "final",
    }
    assert all(record["objective"]["feasible"] for record in records)
    assert all(record["request"]["seed"] == 1337 for record in records)
    assert all(
        record["report"]["metrics"]["sum_radii"]
        == pytest.approx(0.9597642169962064)
        for record in records
    )

    for record in records:
        artifact_root = (
            example
            / ".vero"
            / "session"
            / "evaluations"
            / record["id"]
            / "artifacts"
            / "circle-packing"
        )
        assert (artifact_root / "layout.json").is_file()
        assert (artifact_root / "layout.svg").read_text(
            encoding="utf-8"
        ).startswith("<svg")
        validation = json.loads(
            (artifact_root / "validation.json").read_text(encoding="utf-8")
        )
        assert validation["valid"] is True

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=example / "target",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""
