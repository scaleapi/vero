from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_advertised_python_matmul_evaluation_runs_end_to_end(tmp_path: Path):
    script = (
        Path(__file__).parents[2]
        / "vero-tasks"
        / "examples"
        / "matmul-kernel"
        / "run.py"
    )
    environment = dict(os.environ)
    environment["UV_CACHE_DIR"] = str(
        Path(os.environ.get("UV_CACHE_DIR", tmp_path / "uv-cache")).resolve()
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--eval-only",
            "--work-dir",
            str(tmp_path / "example"),
        ],
        cwd=script.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Baseline score:" in result.stdout
    assert (tmp_path / "example" / "session" / "manifest.json").exists()
