"""Tests for task metrics file-based communication between subprocess and evaluator."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vero.evaluator import Evaluator
from vero.utils.asyncio import SubprocessResult

pytestmark = pytest.mark.asyncio


@pytest.fixture
def experiment_dir(tmp_path):
    """Create a temporary experiment directory with a params file."""
    params_file = tmp_path / "evaluation_parameters.json"
    params_file.write_text("{}")
    return tmp_path, params_file


@pytest.fixture
def evaluator():
    """Create an Evaluator with mocked workspace."""
    ws = MagicMock()
    ws.project_path = Path("/fake/project")
    return Evaluator(workspace=ws, session_id="test-session")


async def test_run_task_reads_metrics_from_file(evaluator, experiment_dir):
    """_run_task reads metrics.json written by the subprocess, not stdout."""
    tmp_path, params_file = experiment_dir
    expected_metrics = {"num_samples": 5, "avg_score": 0.8}

    # Simulate: subprocess writes metrics.json, stdout has noise
    def fake_subprocess(*args, **kwargs):
        (tmp_path / "metrics.json").write_text(json.dumps(expected_metrics))
        return SubprocessResult(
            args=["fake"],
            stdout="[INFO] noisy library output\nprogress bar stuff\n",
            stderr="",
            returncode=0,
        )

    with patch("vero.evaluator.run_subprocess_with_tee", new=AsyncMock(side_effect=fake_subprocess)):
        with patch("vero.evaluator.UvRunParameters.from_env", return_value=MagicMock(get_cmd=lambda: ["uv", "run"])):
            result = await evaluator._run_task(
                Path("/fake/project"), "test_task", params_file
            )

    assert result == expected_metrics


async def test_run_task_returns_none_when_no_metrics_file(evaluator, experiment_dir):
    """_run_task returns None when metrics.json is not written."""
    _, params_file = experiment_dir

    def fake_subprocess(*args, **kwargs):
        return SubprocessResult(args=["fake"], stdout="", stderr="", returncode=0)

    with patch("vero.evaluator.run_subprocess_with_tee", new=AsyncMock(side_effect=fake_subprocess)):
        with patch("vero.evaluator.UvRunParameters.from_env", return_value=MagicMock(get_cmd=lambda: ["uv", "run"])):
            result = await evaluator._run_task(
                Path("/fake/project"), "test_task", params_file
            )

    assert result is None


async def test_run_task_returns_none_on_invalid_metrics_json(evaluator, experiment_dir):
    """_run_task returns None when metrics.json contains invalid JSON."""
    tmp_path, params_file = experiment_dir

    def fake_subprocess(*args, **kwargs):
        (tmp_path / "metrics.json").write_text("not valid json {{{")
        return SubprocessResult(args=["fake"], stdout="", stderr="", returncode=0)

    with patch("vero.evaluator.run_subprocess_with_tee", new=AsyncMock(side_effect=fake_subprocess)):
        with patch("vero.evaluator.UvRunParameters.from_env", return_value=MagicMock(get_cmd=lambda: ["uv", "run"])):
            result = await evaluator._run_task(
                Path("/fake/project"), "test_task", params_file
            )

    assert result is None


async def test_run_task_saves_subprocess_output(evaluator, experiment_dir):
    """_run_task saves stdout/stderr to log files for debugging."""
    tmp_path, params_file = experiment_dir

    def fake_subprocess(*args, **kwargs):
        (tmp_path / "metrics.json").write_text("{}")
        return SubprocessResult(
            args=["fake"],
            stdout="some stdout",
            stderr="some stderr",
            returncode=0,
        )

    with patch("vero.evaluator.run_subprocess_with_tee", new=AsyncMock(side_effect=fake_subprocess)):
        with patch("vero.evaluator.UvRunParameters.from_env", return_value=MagicMock(get_cmd=lambda: ["uv", "run"])):
            await evaluator._run_task(Path("/fake/project"), "test_task", params_file)

    assert (tmp_path / "subprocess_stdout.log").read_text() == "some stdout"
    assert (tmp_path / "subprocess_stderr.log").read_text() == "some stderr"
