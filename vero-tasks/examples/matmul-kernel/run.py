"""Run VeRO's matrix-multiplication program optimization example.

Run this file from the scale-vero project environment:

    uv run python examples/matmul-kernel/run.py --eval-only
    uv run python examples/matmul-kernel/run.py --agent vero
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from vero.evaluation import (
    EvaluationBudget,
    EvaluationDefinition,
    EvaluationPlan,
    EvaluationPrincipal,
    EvaluationSet,
    MetricAggregation,
    MetricSelector,
    ObjectiveSpec,
    PythonTaskBackend,
    PythonTaskBackendConfig,
    PythonTaskEvaluationConfig,
)
from vero.optimization import SequentialStrategy
from vero.runtime import create_local_optimization_session

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATOR_DIR = SCRIPT_DIR.parent / "matmul-eval"

INSTRUCTION = """You are optimizing a matrix multiplication function for speed.

The target is src/matmul_kernel/__init__.py. Preserve the public multiply(a, b)
signature and numerical correctness. You may change the implementation and add
target dependencies. Use the evaluate tool with evaluation="matmul" when
measurement would help; the
objective is mean score in milliseconds, so lower is better.
"""


def _multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(a_value * b[p][j] for p, a_value in enumerate(row))
            for j in range(len(b[0]))
        ]
        for row in a
    ]


def create_cases(path: Path) -> Path:
    def matrix(rows: int, columns: int, seed: int) -> list[list[float]]:
        generator = random.Random(seed)
        return [
            [generator.uniform(-10, 10) for _ in range(columns)]
            for _ in range(rows)
        ]

    inputs = [
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[1, 0], [0, 1]], [[9, 10], [11, 12]]),
        (matrix(8, 8, 42), matrix(8, 8, 52)),
        (matrix(10, 10, 43), matrix(10, 10, 53)),
        (matrix(12, 12, 44), matrix(12, 12, 54)),
    ]
    cases = [
        {
            "id": f"matrix-{index}",
            "matrix_a": a,
            "matrix_b": b,
            "expected": _multiply(a, b),
        }
        for index, (a, b) in enumerate(inputs)
    ]
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def create_target(work_dir: Path) -> Path:
    target = work_dir / "matmul-kernel"
    target.mkdir()
    shutil.copytree(SCRIPT_DIR / "src", target / "src")
    shutil.copy2(SCRIPT_DIR / "pyproject.toml", target / "pyproject.toml")
    (target / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.pyc\n*.egg-info/\n",
        encoding="utf-8",
    )
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
    return target


def create_backend(cases: Path) -> PythonTaskBackend:
    return PythonTaskBackend(
        PythonTaskBackendConfig(
            harness_root=str(EVALUATOR_DIR),
            module="matmul_eval.matmul_task",
            task="matmul",
            evaluations=[
                PythonTaskEvaluationConfig(name="matmul", cases_path=str(cases))
            ],
            passthrough_environment=["UV_CACHE_DIR"],
        )
    )


async def run_example(
    *,
    work_dir: Path,
    agent_name: str | None,
    max_proposals: int,
    max_evaluations: int | None,
) -> None:
    target = create_target(work_dir)
    cases = create_cases(work_dir / "cases.json")
    evaluation_set = EvaluationSet(name="matmul")
    agent_budget = (
        EvaluationBudget(
            backend_id="python-task",
            evaluation_set_key=evaluation_set.budget_key("python-task"),
            principal=EvaluationPrincipal.AGENT,
            total_runs=max_evaluations,
        )
        if max_evaluations is not None
        else None
    )
    system_budget = (
        EvaluationBudget(
            backend_id="python-task",
            evaluation_set_key=evaluation_set.budget_key("python-task"),
            principal=EvaluationPrincipal.SYSTEM,
            total_runs=max_evaluations,
        )
        if max_evaluations is not None
        else None
    )
    evaluation_plan = EvaluationPlan(
        evaluations=[
            EvaluationDefinition(
                evaluation_set=evaluation_set,
                agent_budget=agent_budget,
                system_budget=system_budget,
            )
        ],
        selection_evaluation="matmul",
    )
    producers = {}
    if agent_name is not None:
        from vero.agents import AgentCandidateProducer

        if agent_name == "claude":
            from vero.agents import ClaudeCodeAgent

            coding_agent = ClaudeCodeAgent()
        else:
            from vero.agents import VeroAgent

            coding_agent = VeroAgent()
        producers["default"] = AgentCandidateProducer(
            coding_agent,
            prompt=INSTRUCTION,
            max_turns=100,
        )

    session = await create_local_optimization_session(
        project_path=target,
        session_dir=work_dir / "session",
        backend_id="python-task",
        backend=create_backend(cases),
        objective=ObjectiveSpec(
            selector=MetricSelector(
                metric="score",
                aggregation=MetricAggregation.MEAN,
                case_failure_value=1.0e12,
            ),
            direction="minimize",
            failure_value=1.0e12,
        ),
        evaluation_plan=evaluation_plan,
        strategy=SequentialStrategy(instruction=INSTRUCTION),
        producers=producers,
        parameters={"n_repeats": 100},
        max_proposals=max_proposals,
    )
    result = await session.run()
    print(f"Session: {session.session_dir}")
    print(f"Baseline score: {result.baseline.objective.value:.6f} ms")
    if result.best is not None:
        print(f"Best score: {result.best.objective.value:.6f} ms")
        print(f"Best version: {result.best.request.candidate.version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the baseline without starting a coding agent.",
    )
    parser.add_argument(
        "--agent",
        choices=["vero", "claude"],
        default="vero",
        help="Coding-agent adapter used for optimization.",
    )
    parser.add_argument("--max-proposals", type=int, default=5)
    parser.add_argument(
        "--max-evaluations",
        type=int,
        help=(
            "Optional evaluation-run budget, including the baseline, agent "
            "checkpoints, and completed candidates. By default evaluations are "
            "not separately capped."
        ),
    )
    parser.add_argument("--work-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.max_proposals < 0:
        parser.error("--max-proposals must be non-negative")
    if arguments.max_evaluations is not None and arguments.max_evaluations < 1:
        parser.error("--max-evaluations must be positive")

    work_dir = arguments.work_dir or Path(tempfile.mkdtemp(prefix="vero-matmul-"))
    work_dir = work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {work_dir}")
    asyncio.run(
        run_example(
            work_dir=work_dir,
            agent_name=None if arguments.eval_only else arguments.agent,
            max_proposals=0 if arguments.eval_only else arguments.max_proposals,
            max_evaluations=arguments.max_evaluations,
        )
    )


if __name__ == "__main__":
    main()
