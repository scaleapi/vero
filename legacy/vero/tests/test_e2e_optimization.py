"""E2E integration test: matrix multiply kernel optimization.

Exercises the full Workspace ← Sandbox architecture:
- GitWorkspace created from project path
- Sandbox provides filesystem + shell for git operations
- Evaluator runs task in subprocess at specific commits
- Workspace.save() commits changes, evaluator sees new version

Uses example packages from examples/matmul-kernel and examples/matmul-eval.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Set up kernel, task project, and dataset in a temp dir using the example's setup_workspace."""
    vero_home = tmp_path / "vero_home"
    vero_home.mkdir()
    (vero_home / "sessions").mkdir()
    (vero_home / "datasets").mkdir()
    monkeypatch.setenv("VERO_HOME_DIR", str(vero_home))

    # Use the example's setup function
    sys_path_hack = str(Path(__file__).resolve().parent.parent / "examples" / "matmul-kernel")
    import sys

    sys.path.insert(0, sys_path_hack)
    try:
        from run import setup_workspace

        kernel_dir, task_dir, dataset_path = setup_workspace(tmp_path)
    finally:
        sys.path.pop(0)

    yield kernel_dir, task_dir, dataset_path, vero_home


async def test_matmul_kernel_evaluates(workspace):
    """Naive kernel evaluates correctly — all samples produce valid scores."""
    kernel_dir, task_dir, dataset_path, vero_home = workspace

    from vero.evaluator import run_evaluation

    result = await run_evaluation(
        project_path=kernel_dir,
        dataset=str(dataset_path),
        split="test",
        task="matmul",
        task_project=task_dir,
        task_module="matmul_eval.matmul_task",
        timeout=120,
        vero_home=vero_home,
    )

    assert result is not None
    assert len(result.sample_results) == 5

    for sr in result.sample_results.values():
        assert sr.score is not None
        assert sr.score < 999999.0, "Kernel produced incorrect results"
        assert sr.score > 0, "Score should be time in ms"
        assert sr.metrics["correct"] == 1.0
        assert sr.metrics["time_ms"] > 0


async def test_kernel_change_changes_score(workspace):
    """Modifying kernel code and re-evaluating produces different scores."""
    kernel_dir, task_dir, dataset_path, vero_home = workspace

    from vero.evaluator import run_evaluation

    # Evaluate naive kernel
    result_v1 = await run_evaluation(
        project_path=kernel_dir,
        dataset=str(dataset_path),
        split="test",
        task="matmul",
        task_project=task_dir,
        task_module="matmul_eval.matmul_task",
        sample_ids=[0],
        timeout=120,
        vero_home=vero_home,
    )
    naive_score = result_v1.sample_results[0].score
    assert naive_score < 999999.0

    # Replace kernel with a broken implementation
    init_py = kernel_dir / "src" / "matmul_kernel" / "__init__.py"
    init_py.write_text(textwrap.dedent("""\
        def multiply(a, b):
            return [[0.0]]
    """))
    subprocess.run(["git", "add", "."], cwd=kernel_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", "commit", "-m", "break kernel"],
        cwd=kernel_dir,
        capture_output=True,
        check=True,
    )

    # Re-evaluate — should get penalty score
    result_v2 = await run_evaluation(
        project_path=kernel_dir,
        dataset=str(dataset_path),
        split="test",
        task="matmul",
        task_project=task_dir,
        task_module="matmul_eval.matmul_task",
        sample_ids=[0],
        timeout=120,
        vero_home=vero_home,
    )
    broken_score = result_v2.sample_results[0].score
    assert broken_score == 999999.0, f"Broken kernel should get penalty score, got {broken_score}"


async def test_workspace_save_and_evaluate(workspace):
    """Full loop: GitWorkspace.save() commits, evaluator runs at that commit."""
    kernel_dir, task_dir, dataset_path, vero_home = workspace

    from vero.sandbox import LocalSandbox
    from vero.workspace.git import GitWorkspace

    sandbox = LocalSandbox(root=kernel_dir)
    ws = await GitWorkspace.from_path(sandbox, kernel_dir)

    # Verify initial state
    assert not await ws.is_dirty()
    initial_version = await ws.current_version()

    # Edit kernel via sandbox (the way an agent would)
    init_path = str(Path(kernel_dir) / "src" / "matmul_kernel" / "__init__.py")
    await sandbox.write_file(
        init_path,
        textwrap.dedent("""\
            def multiply(a, b):
                \"\"\"Still naive but with a minor tweak.\"\"\"
                n = len(a)
                m = len(b[0])
                k = len(b)
                result = [[0.0] * m for _ in range(n)]
                for i in range(n):
                    for j in range(m):
                        s = 0.0
                        for p in range(k):
                            s += a[i][p] * b[p][j]
                        result[i][j] = s
                return result
        """),
    )

    assert await ws.is_dirty()

    # Save via workspace (creates git commit)
    new_version = await ws.save("Optimize inner loop accumulator")
    assert new_version != initial_version
    assert not await ws.is_dirty()

    # Evaluate at the new commit
    from datasets import DatasetDict

    from vero.core.dataset.store import save_dataset
    from vero.evaluator import Evaluator

    session_id = "test-workspace-eval"
    ds = DatasetDict.load_from_disk(str(dataset_path))
    save_dataset(vero_home / "sessions", vero_home / "datasets", session_id, "dataset", ds)

    evaluator = Evaluator(
        workspace=ws,
        session_id=session_id,
        vero_home=vero_home,
        task_project=task_dir,
        task_module="matmul_eval.matmul_task",
    )

    experiment = await evaluator.evaluate(
        commit=new_version,
        dataset_id="dataset",
        split="test",
        task="matmul",
        sample_ids=[0],
    )

    assert experiment.result is not None
    assert len(experiment.result.sample_results) == 1
    sr = experiment.result.sample_results[0]
    assert sr.score < 999999.0, "Modified kernel should still be correct"
    assert sr.metrics["correct"] == 1.0


async def test_policy_run_optimizes_kernel(workspace):
    """Full Policy.run() loop: agent reads kernel, modifies it, evaluates, iterates.

    Requires an LLM API key (ANTHROPIC_API_KEY or LITELLM_API_KEY).
    """
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("LITELLM_API_KEY"):
        pytest.skip("No API key available")

    kernel_dir, task_dir, dataset_path, vero_home = workspace

    from jinja2 import Template

    from vero.agents.vero import VeroAgent
    from vero.policy import Policy
    from vero.tools.experiment_runner import SplitBudget

    agent = VeroAgent()

    prompt_template = Template(
        "You are optimizing a matrix multiply kernel for speed.\n\n"
        "The kernel is in src/matmul_kernel/__init__.py — it has a single function:\n"
        "  multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]\n\n"
        "Your goal: make multiply() as fast as possible while keeping correctness.\n"
        "The score is the average execution time in milliseconds (lower is better).\n"
        "Incorrect results get a penalty score of 999999.0.\n\n"
        "You may use any approach: numpy, list comprehensions, ctypes, cython, compiled extensions,\n"
        "caching, algorithmic improvements, or anything else you can think of.\n"
        "You can add dependencies to pyproject.toml if needed.\n"
        "The only constraint is that the function signature must stay the same.\n\n"
        "Take your time. Read the code, think about your approach, then implement and evaluate.\n"
        "You have a budget of 5 evaluation runs — use them wisely.\n"
        "After each evaluation, review the results and iterate."
    )

    from vero.sandbox import LocalSandbox

    policy = Policy(
        sandbox=LocalSandbox(root=kernel_dir),
        vero_home=vero_home,
        project_path=kernel_dir,
        dataset=dataset_path,
        agent=agent,
        task="matmul",
        task_project=str(task_dir),
        task_module="matmul_eval.matmul_task",
        use_copy=False,
        budget=[
            SplitBudget(split="test", total_run_budget=5),
        ],
        split_accesses=[],
        max_turns=100,
        prompt_template=prompt_template,
    )

    best = await policy.run(skip_initial_eval=False, eval_split="test")

    # The agent should have made at least one evaluation
    assert policy.session.db is not None
    experiments = policy.session.db.get_experiments()
    assert len(experiments) >= 1, "Agent should have run at least one evaluation"

    # At least one experiment should have correct results
    has_correct = any(
        any(sr.metrics.get("correct", 0) == 1.0 for sr in exp.result.sample_results.values())
        for exp in experiments
        if exp.result and exp.result.sample_results
    )
    assert has_correct, "At least one experiment should produce correct results"


async def test_policy_run_with_artifacts(workspace):
    """Full Policy.run() with artifacts-style optimization.

    Uses DatasetArtifact + TracesArtifact to dump data into _vero/ on the filesystem.
    The agent reads data via file tools instead of DatasetViewer/ExperimentViewer.

    Requires an LLM API key.
    """
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("LITELLM_API_KEY"):
        pytest.skip("No API key available")

    kernel_dir, task_dir, dataset_path, vero_home = workspace

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "matmul-kernel"))
    try:
        from run import run_optimization
    finally:
        sys.path.pop(0)

    await run_optimization(kernel_dir, task_dir, dataset_path, use_artifacts=True)

    # Verify artifacts were materialized
    vero_dir = Path(kernel_dir) / "_vero"
    assert vero_dir.exists(), "_vero directory should exist"
    assert (vero_dir / "datasets").exists(), "Dataset artifacts should be materialized"

    # Verify traces were written after the initial eval
    traces_dir = vero_dir / "traces"
    assert traces_dir.exists(), "Traces directory should exist"
    trace_dirs = list(traces_dir.iterdir())
    assert len(trace_dirs) >= 1, "At least one trace should be materialized"

    # Check a trace has summary.json
    import json

    summary_path = trace_dirs[0] / "summary.json"
    assert summary_path.exists(), "Trace should have summary.json"
    summary = json.loads(summary_path.read_text())
    assert "score" in summary
    assert "status" in summary
