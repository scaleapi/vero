from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

import vero
from vero.cli import main
from vero.candidate import Candidate


def test_root_package_exposes_only_canonical_program_api():
    assert vero.Candidate is Candidate
    assert not hasattr(vero, "Experiment")


def test_canonical_root_and_tools_do_not_import_legacy_core():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, vero, vero.tools; "
                "assert not any(name == 'vero.core' or name.startswith('vero.core.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def initialize_repository(path: Path) -> None:
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


def test_cli_optimizes_non_python_program_and_inspects_session(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("slow\n", encoding="utf-8")
    initialize_repository(target)

    harness = tmp_path / "harness"
    harness.mkdir()
    harness_script = harness / "evaluate.py"
    harness_script.write_text(
        """
import json
import sys
from pathlib import Path
workspace, report = map(Path, sys.argv[1:])
latency = 1.0 if (workspace / "program.txt").read_text().strip() == "fast" else 9.0
report.write_text(json.dumps({
    "schema_version": 1,
    "status": "success",
    "metrics": {"latency_ms": latency},
}))
""",
        encoding="utf-8",
    )
    producer = tmp_path / "producer"
    producer.mkdir()
    producer_script = producer / "improve.py"
    producer_script.write_text(
        """
import sys
from pathlib import Path
Path(sys.argv[1], "program.txt").write_text("fast\\n")
""",
        encoding="utf-8",
    )
    session_dir = tmp_path / "sessions" / "cli-run"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "optimize",
            str(target),
            "--harness-root",
            str(harness),
            "--evaluate",
            shlex.join(
                [
                    sys.executable,
                    str(harness_script),
                    "{workspace}",
                    "{report}",
                ]
            ),
            "--producer-root",
            str(producer),
            "--produce",
            shlex.join([sys.executable, str(producer_script), "{workspace}"]),
            "--metric",
            "latency_ms",
            "--direction",
            "minimize",
            "--evaluation-set",
            "performance",
            "--parameter",
            'threshold=0.5',
            "--session-dir",
            str(session_dir),
            "--session-id",
            "cli-session",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Baseline:" in result.output
    assert "(9.0)" in result.output
    assert "Best:" in result.output
    assert "(1.0)" in result.output
    assert (target / "program.txt").read_text(encoding="utf-8") == "slow\n"

    inspect_result = runner.invoke(main, ["session", "inspect", str(session_dir)])
    assert inspect_result.exit_code == 0, inspect_result.output
    manifest = json.loads(inspect_result.output)
    assert manifest["id"] == "cli-session"
    assert manifest["status"] == "completed"

    list_result = runner.invoke(
        main,
        ["session", "list", "--root", str(session_dir.parent)],
    )
    assert list_result.exit_code == 0, list_result.output
    assert "cli-session\tcompleted" in list_result.output


def test_cli_requires_exactly_one_producer(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    harness = tmp_path / "harness"
    harness.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "optimize",
            str(target),
            "--harness-root",
            str(harness),
            "--evaluate",
            "evaluate {workspace} {report}",
            "--metric",
            "score",
            "--direction",
            "maximize",
        ],
    )

    assert result.exit_code == 2
    assert "exactly one of --produce or --agent" in result.output
