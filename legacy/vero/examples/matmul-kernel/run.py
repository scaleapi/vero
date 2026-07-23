"""Run the matmul kernel optimization loop.

Usage:
    uv run examples/matmul-kernel/run.py [--eval-only]

Requires:
    - ANTHROPIC_API_KEY or LITELLM_API_KEY + LITELLM_BASE_URL
    - examples/matmul-eval to be uv synced
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parent
VERO_ROOT = EXAMPLES_DIR.parent


def create_dataset(dest: Path) -> Path:
    """Create a matmul dataset with test matrices."""
    from datasets import Dataset, DatasetDict

    import random

    def _random_matrix(n: int, m: int, seed: int) -> list[list[float]]:
        rng = random.Random(seed)
        return [[rng.uniform(-10, 10) for _ in range(m)] for _ in range(n)]

    def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        n, k, m = len(a), len(b), len(b[0])
        return [[sum(a[i][p] * b[p][j] for p in range(k)) for j in range(m)] for i in range(n)]

    # Small matrices for correctness, slightly larger for timing.
    # Keep sizes modest so the dataset viewer doesn't blow up agent context.
    matrices_a = [
        [[1, 2], [3, 4]],
        [[1, 0], [0, 1]],
        _random_matrix(8, 8, seed=42),
        _random_matrix(10, 10, seed=43),
        _random_matrix(12, 12, seed=44),
    ]
    matrices_b = [
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
        _random_matrix(8, 8, seed=52),
        _random_matrix(10, 10, seed=53),
        _random_matrix(12, 12, seed=54),
    ]
    expected = [_matmul(a, b) for a, b in zip(matrices_a, matrices_b)]

    ds = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "matrix_a": matrices_a,
                    "matrix_b": matrices_b,
                    "expected": expected,
                }
            )
        }
    )
    dataset_path = dest / "dataset"
    ds.save_to_disk(str(dataset_path))
    return dataset_path


def setup_workspace(work_dir: Path) -> tuple[Path, Path, Path]:
    """Copy kernel and task project to a working directory, return (kernel_dir, task_dir, dataset_path)."""
    # Copy kernel
    kernel_dir = work_dir / "matmul-kernel"
    shutil.copytree(
        SCRIPT_DIR,
        kernel_dir,
        ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc"),
    )

    # Fix vero path in kernel's pyproject.toml
    kernel_pyproject = kernel_dir / "pyproject.toml"
    kernel_pyproject.write_text(
        kernel_pyproject.read_text().replace(
            'path = "../../", editable = true',
            f'path = "{VERO_ROOT}", editable = true',
        )
    )

    # Init git (with .gitignore to prevent build artifacts from being tracked)
    (kernel_dir / ".gitignore").write_text("__pycache__/\n*.pyc\n.venv/\n*.egg-info/\n")
    subprocess.run(["git", "init"], cwd=kernel_dir, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=kernel_dir, capture_output=True, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "init",
        ],
        cwd=kernel_dir,
        capture_output=True,
        check=True,
    )

    # Copy task project
    task_dir = work_dir / "matmul-eval"
    shutil.copytree(
        EXAMPLES_DIR / "matmul-eval",
        task_dir,
        ignore=shutil.ignore_patterns(".venv", "uv.lock", "__pycache__"),
    )
    # Fix vero path
    pyproject = task_dir / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            'path = "../../", editable = true',
            f'path = "{VERO_ROOT}", editable = true',
        )
    )
    subprocess.run(["uv", "sync"], cwd=task_dir, capture_output=True, check=True)

    # Create dataset
    dataset_path = create_dataset(work_dir)

    return kernel_dir, task_dir, dataset_path


async def run_eval_only(kernel_dir: Path, task_dir: Path, dataset_path: Path) -> None:
    """Run a single evaluation of the kernel."""
    from vero.evaluator import run_evaluation

    result = await run_evaluation(
        project_path=kernel_dir,
        dataset=str(dataset_path),
        split="test",
        task="matmul",
        task_project=task_dir,
        task_module="matmul_eval.matmul_task",
        timeout=120,
    )

    print(f"\nResults: {len(result.sample_results)} samples")
    for sid, sr in result.sample_results.items():
        print(
            f"  Sample {sid}: score={sr.score:.4f}ms correct={sr.metrics.get('correct', '?')}"
        )
    print(f"  Mean score: {result.score():.4f}ms")


PROMPT_BASE = (
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
)

PROMPT_ARTIFACTS = (
    "The dataset samples are materialized as JSON files in _vero/datasets/dataset/test/.\n"
    "After each evaluation, traces are written to _vero/traces/{split}__{commit}/.\n"
    "Each trace directory has a summary.json with the score and per-sample results.\n"
    "Use file reading tools to inspect data and results.\n\n"
)

PROMPT_TAIL = (
    "Take your time. Read the code, think about your approach, then implement and evaluate.\n"
    "You have a budget of 5 evaluation runs — use them wisely.\n"
    "After each evaluation, review the results and iterate."
)


PROMPT_RESOURCES = (
    "The multiply function is registered as a vero resource under the 'kernel' namespace.\n"
    "Use the ResourceControl tools to view and edit it by name (kernel.multiply).\n"
    "Do NOT edit files directly — use resource tools instead.\n\n"
)


def _make_agent(agent_type: str, use_artifacts: bool, use_resources: bool):
    """Create an agent based on type.

    Returns (agent, artifacts, prompt_template).
    """
    from jinja2 import Template

    artifacts_list = []
    if use_artifacts or agent_type == "claude-code":
        from vero.artifacts import DatasetArtifact, TracesArtifact

        artifacts_list = [DatasetArtifact(), TracesArtifact()]

    prompt_text = PROMPT_BASE
    if use_resources:
        prompt_text += PROMPT_RESOURCES
    if artifacts_list:
        prompt_text += PROMPT_ARTIFACTS
    prompt_text += PROMPT_TAIL
    prompt_template = Template(prompt_text)

    if agent_type == "claude-code":
        from vero.agents.claude_code import ClaudeCodeAgent

        agent = ClaudeCodeAgent()
        return agent, artifacts_list, prompt_template

    # VeroAgent
    from vero.agents.vero import VeroAgent

    if use_resources:
        from vero.tools import (
            BashTool,
            DatasetViewer,
            ExperimentRunnerTool,
            ExperimentViewer,
            FileRead,
            GitControl,
            GitViewer,
            Grep,
            ResourceControl,
            SubAgentTool,
            TodoList,
            think,
        )

        tool_sets = [
            BashTool(),
            DatasetViewer(),
            ExperimentRunnerTool(),
            ExperimentViewer(),
            FileRead(),
            GitControl(),
            GitViewer(),
            Grep(),
            ResourceControl(),
            SubAgentTool(),
            TodoList(),
            think,
        ]
        agent = VeroAgent(tool_sets=tool_sets)
    elif use_artifacts:
        from vero.tools import (
            BashTool,
            ExperimentRunnerTool,
            FileRead,
            FileWrite,
            GitControl,
            GitViewer,
            Grep,
            SubAgentTool,
            TodoList,
            think,
        )

        tool_sets = [
            BashTool(),
            ExperimentRunnerTool(),
            FileRead(),
            FileWrite(),
            GitControl(),
            GitViewer(),
            Grep(),
            SubAgentTool(),
            TodoList(),
            think,
        ]
        agent = VeroAgent(tool_sets=tool_sets)
    else:
        agent = VeroAgent()

    return agent, artifacts_list, prompt_template


async def run_optimization(
    kernel_dir: Path,
    task_dir: Path,
    dataset_path: Path,
    use_artifacts: bool = False,
    use_resources: bool = False,
    agent_type: str = "vero",
) -> None:
    """Run the full optimization loop.

    Args:
        use_artifacts: Dump data to _vero/ instead of using viewer tools.
        use_resources: Use ResourceControl instead of FileWrite (edits by resource name).
        agent_type: "vero" (OpenAI Agents SDK) or "claude-code" (Claude Agent SDK).
    """
    from vero.policy import Policy
    from vero.tools.experiment_runner import SplitBudget

    agent, artifacts, prompt_template = _make_agent(agent_type, use_artifacts, use_resources)

    policy = Policy(
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
        artifacts=artifacts,
        max_turns=100,
        prompt_template=prompt_template,
        console_verbose=False,
    )

    best = await policy.run(skip_initial_eval=False, eval_split="test")

    print(f"\nBest commit: {best.commit}")
    print(f"Best score: {best.score}")

    experiments = policy.session.db.get_experiments()
    print(f"Total experiments: {len(experiments)}")
    for i, exp in enumerate(experiments):
        print(
            f"  Experiment {i}: score={exp.result.score():.4f}ms status={exp.result.status}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run matmul kernel optimization")
    parser.add_argument(
        "--eval-only", action="store_true", help="Just evaluate, don't optimize"
    )
    parser.add_argument(
        "--artifacts", action="store_true",
        help="Use filesystem artifacts instead of DatasetViewer/ExperimentViewer",
    )
    parser.add_argument(
        "--resources", action="store_true",
        help="Use ResourceControl instead of FileWrite (edits by resource name)",
    )
    parser.add_argument(
        "--agent", type=str, default="vero", choices=["vero", "claude-code"],
        help="Agent backend: 'vero' (OpenAI Agents SDK) or 'claude-code' (Claude Agent SDK)",
    )
    parser.add_argument(
        "--work-dir", type=str, default=None, help="Working directory (default: temp)"
    )
    args = parser.parse_args()

    import logging

    logging.getLogger().setLevel(logging.INFO)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="matmul-"))
        print(f"Working directory: {work_dir}")

    kernel_dir, task_dir, dataset_path = setup_workspace(work_dir)

    if args.eval_only:
        asyncio.run(run_eval_only(kernel_dir, task_dir, dataset_path))
    else:
        asyncio.run(run_optimization(
            kernel_dir, task_dir, dataset_path,
            use_artifacts=args.artifacts, use_resources=args.resources,
            agent_type=args.agent,
        ))


if __name__ == "__main__":
    main()
