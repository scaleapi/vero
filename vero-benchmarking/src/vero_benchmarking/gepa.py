"""
GEPA adapter for Vero benchmarking.

Bridges GEPA's optimization loop with Vero's evaluation infrastructure,
using VeroResources as the candidate components that GEPA mutates.

Usage:
    python -m vero_benchmarking.gepa --task math --model sonnet
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar
from uuid import uuid4

from gepa.api import optimize
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.proposer.reflective_mutation.base import LanguageModel
from vero.core.dataset import DatasetInfo
from vero.core.db.dataset import DatasetSample
from vero.core.db.result import SampleResult
from vero.core.resource import ResourceDiscovery, StaticResourceInfo
from vero.core.sessions import create_session_dir, get_session_dir
from vero.evaluator import run_evaluation
from vero.utils import random_readable_id

from datasets import DatasetDict
from vero_benchmarking.tasks.base import OptimizationTask

SpanT = TypeVar("SpanT")
logger = logging.getLogger(__name__)


class VeroGEPAAdapter(
    GEPAAdapter[DatasetSample, list[list[SpanT]], list[SampleResult]]
):
    """Adapter that connects GEPA to Vero's evaluation infrastructure.

    Uses VeroResources as the candidate components. GEPA proposes mutations
    to resource source code, and Vero evaluates the modified agent.
    """

    def __init__(
        self,
        optimization_task: OptimizationTask,
        commit: str = "HEAD",
    ):
        self.optimization_task = optimization_task
        self.project_path = Path(optimization_task.project_path)
        self.dataset_path = Path(optimization_task.dataset_path)
        self.dataset_id = self.dataset_path.stem
        self.task = optimization_task.task
        self.commit = commit
        self.resource_namespace = optimization_task.resource_namespace

        # Isolate the project into a fresh repo under a session directory
        self.session_id = str(uuid4())
        create_session_dir(self.session_id)

        isolated_dir = get_session_dir(self.session_id) / self.project_path.name
        isolated_dir.mkdir(parents=True, exist_ok=True)

        repo_root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=self.project_path,
            capture_output=True,
            text=True,
        )
        if repo_root_result.returncode == 0:
            repo_root = Path(repo_root_result.stdout.strip())
            project_rel = self.project_path.resolve().relative_to(repo_root)
            strip = len(project_rel.parts)
            archive = subprocess.Popen(
                ["git", "archive", self.commit, str(project_rel)],
                cwd=repo_root,
                stdout=subprocess.PIPE,
            )
            subprocess.run(
                ["tar", "xf", "-", "--strip-components", str(strip)],
                cwd=isolated_dir,
                stdin=archive.stdout,
                check=True,
            )
            archive.wait()
        else:
            import shutil

            shutil.copytree(self.project_path, isolated_dir, dirs_exist_ok=True)

        # Fix relative vero source paths in pyproject.toml so they resolve
        # from the isolated directory (which is no longer in the original repo tree)
        self._fix_vero_source_path(isolated_dir)

        subprocess.run(
            ["git", "init"], cwd=isolated_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "add", "."], cwd=isolated_dir, capture_output=True, check=True
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
                "Initial commit (GEPA isolated)",
            ],
            cwd=isolated_dir,
            capture_output=True,
            check=True,
        )

        self._project_path = isolated_dir
        self._repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=isolated_dir,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        self._package_rel_path = str(isolated_dir.relative_to(self._repo_root))
        logger.info(
            f"GEPA session {self.session_id}: isolated project at {isolated_dir}"
        )

        # Load dataset
        self._dataset = DatasetDict.load_from_disk(str(self.dataset_path))

        # Discover resources
        self._resources = self._discover_resource_infos()
        self._seed_candidate = {
            f"{r.namespace}::{r.name}": r.source for r in self._resources
        }

    @staticmethod
    def _fix_vero_source_path(project_dir: Path) -> None:
        """Fix relative scale-vero source paths in pyproject.toml.

        When a project is isolated into a session directory, relative paths
        no longer resolve.
        Replace them with the absolute path to the current vero installation.
        """
        import re

        import vero

        vero_path = Path(vero.__file__).parent.parent.parent
        pyproject = project_dir / "pyproject.toml"
        if not pyproject.exists():
            return
        content = pyproject.read_text()
        # Match: scale-vero = { path = "...", editable = true }
        new_content = re.sub(
            r'(scale-vero\s*=\s*\{\s*path\s*=\s*)"[^"]*"',
            rf'\1"{vero_path}"',
            content,
        )
        if new_content != content:
            pyproject.write_text(new_content)
            logger.info(f"Fixed vero source path in {pyproject}")

    def _discover_resource_infos(self) -> list[StaticResourceInfo]:
        resources = ResourceDiscovery.discover_at_commit(
            repo_path=self._repo_root,
            commit="HEAD",
            package_rel_path=self._package_rel_path,
        )
        if self.resource_namespace:
            resources = [r for r in resources if r.namespace == self.resource_namespace]
        if not resources:
            raise ValueError(
                f"No resources found in {self.project_path} "
                f"(namespace={self.resource_namespace}). "
                f"Ensure functions are decorated with @resource()."
            )
        for r in resources:
            logger.info(f"Discovered resource: {r.namespace}::{r.name}")
        return resources

    def _apply_candidate(self, candidate: dict[str, str]) -> str:
        """Apply candidate resource source code to the worktree and commit."""
        current_resources = ResourceDiscovery.discover_at_commit(
            repo_path=self._repo_root,
            commit="HEAD",
            package_rel_path=self._package_rel_path,
        )
        for resource_info in current_resources:
            key = f"{resource_info.namespace}::{resource_info.name}"
            if key not in candidate:
                continue
            new_source = candidate[key]
            if new_source == resource_info.source:
                continue
            file_path = self._repo_root / resource_info.file_path
            content = file_path.read_text()
            new_content = content.replace(resource_info.source, new_source)

            # Validate that the replacement produces valid Python
            try:
                compile(new_content, str(file_path), "exec")
            except SyntaxError:
                logger.warning(
                    f"GEPA proposed invalid Python for {key}, skipping replacement"
                )
                continue

            file_path.write_text(new_content)

        subprocess.run(
            ["git", "add", "--all"],
            cwd=self._repo_root,
            capture_output=True,
            check=True,
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
                "GEPA candidate evaluation",
                "--no-verify",
            ],
            cwd=self._repo_root,
            capture_output=True,
            check=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self._repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    @property
    def seed_candidate(self) -> dict[str, str]:
        return self._seed_candidate

    def components(self) -> list[str]:
        return list(self._seed_candidate.keys())

    def evaluate(
        self,
        batch: list[DatasetSample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        split = batch[0].split
        assert all(s.split == split for s in batch)
        sample_ids = [s.sample_id for s in batch]

        commit = self._apply_candidate(candidate)

        coro = run_evaluation(
            project_path=str(self._project_path),
            dataset_path=str(self.dataset_path),
            task=self.task,
            split=split,
            sample_ids=sample_ids,
            commit=commit,
            session_id=self.session_id,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            experiment_result = loop.run_until_complete(coro)
        else:
            experiment_result = asyncio.run(coro)

        sample_results = list(experiment_result.sample_results.values())
        scores = [sr.score if sr.score is not None else 0.0 for sr in sample_results]
        trajectories = (
            [sr.execution_trace for sr in sample_results] if capture_traces else None
        )
        return EvaluationBatch(
            scores=scores, trajectories=trajectories, outputs=sample_results
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[list[list[SpanT]], list[SampleResult]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        reflective_dataset: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            reflective_dataset[component] = []
            for sample_result in eval_batch.outputs:
                reflective_dataset[component].append(
                    {
                        "Inputs": sample_result.input or {},
                        "Generated Outputs": sample_result.execution_trace,
                        "Feedback": sample_result.feedback,
                        "Error": sample_result.error,
                        "Score": sample_result.score,
                    }
                )
        return reflective_dataset


def dataset_info_to_samples(info: DatasetInfo) -> dict[str, list[DatasetSample]]:
    return {
        split: [
            DatasetSample(dataset_id=info.id, split=split, sample_id=i)
            for i in range(info.splits[split])
        ]
        for split in info.splits
    }


def get_reflection_lm(model: str) -> LanguageModel:
    """Create a reflection LM using the OpenAI client, routing through the proxy."""
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("LITELLM_BASE_URL"),
        api_key=os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY")),
    )

    def reflection_lm(prompt: str) -> str | None:
        return (
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            .choices[0]
            .message.content
        )

    return reflection_lm


REFLECTION_PROMPT_TEMPLATE = """The component below is a Python function decorated with @resource(). \
It is part of an AI agent's codebase and will be executed as Python code.

Current component source code:
```python
<curr_instructions>
```

The following are examples of task inputs, the agent's outputs, and feedback:
```
<inputs_outputs_feedback>
```

Your task is to write an improved version of the Python function above. You MUST:
1. Keep the @resource() decorator and function signature exactly the same.
2. Only modify the function body (e.g. prompt strings, logic, constants).
3. Output syntactically valid Python that can replace the current function.
4. Preserve all imports that the function depends on.

Provide the new function within ``` blocks."""


def run_gepa(
    task_name: str,
    model: str = "sonnet",
    commit: str = "HEAD",
    max_train_runs: int | None = None,
    max_validation_runs: int | None = None,
    reflection_minibatch_size: int = 32,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    enable_wandb: bool = False,
    wandb_project: str = "vero-gepa-benchmarking",
    skip_initial_eval: bool = False,
) -> dict[str, Any]:
    """Run GEPA optimization on a task. Returns results dict."""
    from vero_benchmarking.runner import MODELS
    from vero_benchmarking.tasks import load_task

    task = load_task(task_name)
    model_str = MODELS[model]

    max_train_runs = max_train_runs or task.train_budget or 5
    max_validation_runs = max_validation_runs or task.validation_budget or 0
    score_threshold = task.score_threshold

    adapter = VeroGEPAAdapter(optimization_task=task, commit=commit)

    dataset_info = DatasetInfo(
        id=adapter.dataset_id,
        splits={split: len(adapter._dataset[split]) for split in adapter._dataset},
        features={
            split: list(adapter._dataset[split].features) for split in adapter._dataset
        },
    )
    samples = dataset_info_to_samples(dataset_info)

    trainset = samples["train"]
    has_validation = "validation" in samples
    has_test = "test" in samples
    valset = samples["validation"] if has_validation else trainset
    testset = samples["test"] if has_test else valset

    if max_train_samples:
        trainset = trainset[:max_train_samples]
    if max_val_samples:
        valset = valset[:max_val_samples]

    max_metric_calls = max_train_runs * len(trainset) + max_validation_runs * len(
        valset
    )
    logger.info(f"Task: {task.task}, Model: {model_str}")
    logger.info(
        f"Train budget: {max_train_runs}, Validation budget: {max_validation_runs}"
    )
    logger.info(
        f"Max metric calls: {max_metric_calls}, Score threshold: {score_threshold}"
    )

    reflection_lm = get_reflection_lm(model_str)

    if not skip_initial_eval:
        initial_eval_set = testset if has_test else valset
        initial_eval_split = "test" if has_test else "validation"
        logger.info(f"Running initial {initial_eval_split} evaluation...")
        initial_results = adapter.evaluate(
            initial_eval_set, adapter.seed_candidate, capture_traces=True
        )
        logger.info(
            f"Initial {initial_eval_split} score: {sum(initial_results.scores) / len(initial_results.scores):.4f}"
        )

    run_name = f"gepa_{dataset_info.id}_{model}_{random_readable_id()}"

    # Build stop callbacks
    stop_callbacks = []
    if score_threshold is not None:
        from gepa.utils.stop_condition import ScoreThresholdStopper

        stop_callbacks.append(ScoreThresholdStopper(threshold=score_threshold))

    result = optimize(
        seed_candidate=adapter.seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        perfect_score=1.0,
        skip_perfect_score=False,
        reflection_prompt_template=REFLECTION_PROMPT_TEMPLATE,
        stop_callbacks=stop_callbacks if stop_callbacks else None,
        run_dir=str(Path("results") / run_name),
        use_wandb=enable_wandb,
        wandb_init_kwargs={"project": wandb_project, "name": run_name},
    )

    final_eval_set = testset if has_test else valset
    final_eval_split = "test" if has_test else "validation"
    logger.info(f"Running final {final_eval_split} evaluation...")
    final_results = adapter.evaluate(
        final_eval_set, result.best_candidate, capture_traces=True
    )
    final_score = sum(final_results.scores) / len(final_results.scores)
    logger.info(f"Final {final_eval_split} score: {final_score:.4f}")

    for key, value in result.best_candidate.items():
        print(f"\n{'=' * 60}")
        print(f"Optimized resource: {key}")
        print(f"{'=' * 60}")
        print(value)

    # Persist best candidate and final score
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "best_candidate.json", "w") as f:
        json.dump(result.best_candidate, f, indent=2)
    with open(run_dir / "final_results.json", "w") as f:
        json.dump(
            {"final_score": final_score, "session_id": adapter.session_id}, f, indent=2
        )
    logger.info(f"Saved best candidate and results to {run_dir}")

    return {
        "session_id": adapter.session_id,
        "best_candidate": result.best_candidate,
        "final_score": final_score,
        "run_name": run_name,
    }


def main():
    import argparse

    from vero_benchmarking.runner import MODELS

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run GEPA optimization on a Vero agent"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task name (e.g. math, gsm8k)"
    )
    parser.add_argument(
        "--model", type=str, default="sonnet", choices=list(MODELS.keys())
    )
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--max-train-runs", type=int, default=None)
    parser.add_argument("--max-validation-runs", type=int, default=None)
    parser.add_argument("--reflection-minibatch-size", type=int, required=False)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--skip-initial-eval", action="store_true")
    args = parser.parse_args()

    run_gepa(
        task_name=args.task,
        model=args.model,
        commit=args.commit,
        max_train_runs=args.max_train_runs,
        max_validation_runs=args.max_validation_runs,
        reflection_minibatch_size=args.reflection_minibatch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        enable_wandb=args.enable_wandb,
        skip_initial_eval=args.skip_initial_eval,
    )


if __name__ == "__main__":
    main()
