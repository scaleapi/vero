import shutil
import subprocess
from pathlib import Path

import pytest

from vero.config import build_program_runtime, load_config


def _initialize_target(target: Path) -> str:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=target,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "--all"],
        cwd=target,
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
        cwd=target,
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.asyncio
async def test_c_target_is_optimized_end_to_end_through_vero_toml(tmp_path: Path):
    if shutil.which("cc") is None:
        pytest.skip("C compiler is not available")
    source = Path(__file__).parent.parent / "examples" / "c-matmul"
    example = tmp_path / "c-matmul"
    shutil.copytree(source, example)
    target = example / "target"
    baseline_commit = _initialize_target(target)

    target_files = [path for path in target.rglob("*") if ".git" not in path.parts]
    assert not any(path.suffix == ".py" for path in target_files)
    assert not (target / "pyproject.toml").exists()

    runtime = await build_program_runtime(
        load_config(example / "vero.toml"),
        require_optimizer=True,
    )
    result = await runtime.policy.run()

    assert result.baseline.request.candidate.commit == baseline_commit
    assert len(result.evaluations) == 2
    candidate = result.evaluations[1]
    assert result.baseline.objective.feasible is True
    assert candidate.objective.feasible is True
    assert candidate.objective.value < result.baseline.objective.value
    assert result.best == candidate
    assert candidate.request.candidate.commit != baseline_commit
    assert "Cache-friendly" in (target / "matmul.c").read_text()
