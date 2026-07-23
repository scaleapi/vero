#!/usr/bin/env python3
"""
Terminal Bench optimization experiment.

Runs ClaudeCodeAgent to optimize the TerminusKira agent on Terminal Bench 2.0.
Supports two information access modes for A/B comparison:
  - "tools": Agent uses ExperimentViewer/DatasetViewer MCP tools
  - "artifacts": Agent reads traces/datasets from _vero/ filesystem

Usage:
    # Quick test (3 samples, low budget)
    uv run python scripts/run_terminal_bench.py --sample-budget 3 --max-turns 20

    # Full run with tools-based information access
    uv run python scripts/run_terminal_bench.py --mode tools --sample-budget 50

    # Full run with artifacts-based information access
    uv run python scripts/run_terminal_bench.py --mode artifacts --sample-budget 50

    # Resume from existing session
    uv run python scripts/run_terminal_bench.py --resume <session-id> --mode tools
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402
from vero.agents.claude_code import ClaudeCodeAgent  # noqa: E402
from vero.artifacts import DatasetArtifact, TracesArtifact  # noqa: E402
from vero.core.dataset import SplitAccess  # noqa: E402
from vero.core.evaluation import BaseEvaluationParameters  # noqa: E402
from vero.core.sessions import (  # noqa: E402
    create_session_dir,
    get_session_dir,
    get_session_experiments_dir,
)
from vero.policy import Policy  # noqa: E402
from vero.tools import (  # noqa: E402
    DatasetViewer,
    ExperimentRunnerTool,
    ExperimentViewer,
)
from vero.tools.experiment_runner import SplitBudget  # noqa: E402

from vero_benchmarking.tasks import load_task  # noqa: E402

# =============================================================================
# Constants
# =============================================================================

logger = logging.getLogger(__name__)

TASK_NAME = "terminal-bench"

EVAL_TIMEOUT = (
    10 * 3600
)  # 10 hours — Harbor tasks are long-running (subprocess timeout)
SAMPLE_TIMEOUT = 3600  # 1 hour per sample


# =============================================================================
# Skill: analysis guide
# =============================================================================

ANALYSIS_SKILL = {
    "experiment-analysis": {
        "SKILL.md": """\
---
name: experiment-analysis
description: >
  Analyze evaluation experiment results to identify failure patterns, strengths,
  and optimization opportunities. Use before making changes to understand why the
  agent fails on specific tasks.
metadata:
  version: "1.0"
  author: vero-benchmarking
---

# Analyzing Experiment Results

## Where things are

### Artifacts mode (_vero/ filesystem)
- `_vero/traces/` — experiment results: `summary.json` has aggregate stats, per-sample
  JSON files have score, output, error, and execution_trace for each task
- `_vero/datasets/` — task definitions: each sample describes what the agent was asked to do

### Tools mode (ExperimentViewer / DatasetViewer)
- `ExperimentViewer` — query experiment results programmatically: view experiment table,
  sample results, individual traces, trace summaries
- `DatasetViewer` — query dataset samples: get info, stats, view samples by range

## How to analyze

1. **Start with the big picture.** What's the overall score? How many errors vs successes?
2. **Categorize failures.** Don't just read one or two — sample 10-15 failures and look for
   patterns. Common categories: crashes/exceptions, timeouts, wrong approach, context exhaustion.
3. **Understand successes too.** What do passing tasks have in common? Are they all simple, or
   did some complex ones pass? Why?
4. **Look at traces, not just scores.** A score of 0 doesn't tell you *why* it failed. Read the
   execution trace to see what the agent actually did.
5. **Quantify.** "Some tasks fail" is weak. "13% crash with AttributeError at line 416" is
   actionable.

## What to look for

- **Crashes/exceptions** — instant failures with no recovery. These are free wins to fix.
- **Infinite loops** — agent keeps going but never terminates. Look for high episode counts.
- **Context explosion** — token usage growing without bound. Look for 10M+ input tokens.
- **Timeouts** — agent ran out of time. Was it making progress or stuck?
- **Wrong strategy** — agent tried the wrong approach entirely. Could better instructions help?
- **Missing capabilities** — agent needed a tool or skill it didn't have.

## Pitfalls

- Don't assume all failures have the same root cause — categorize first.
- Don't fix symptoms. A timeout might be caused by an infinite loop, not a short time limit.
- Don't over-index on a single sample. Look for patterns across 10+ samples before drawing
  conclusions.
""",
    },
}


# =============================================================================
# Prompt
# =============================================================================


def build_prompt(mode: str, sample_budget: int) -> str:
    """Build the optimization prompt for the agent."""

    if mode == "tools":
        data_access = """\
You have access to ExperimentViewer and DatasetViewer tools to inspect results and task definitions.
Use them to understand what happened in the baseline evaluation before making changes."""
    else:
        data_access = """\
Baseline evaluation results are in _vero/traces/ (summary.json + per-sample results).
Dataset task definitions are in _vero/datasets/.
Read these to understand what happened in the baseline evaluation before making changes."""

    return f"""\
# Terminal Bench 2.0 Optimization

You are optimizing the TerminusKira agent to improve its score on Terminal Bench 2.0 — a benchmark
where an AI agent completes real-world terminal tasks in sandboxed environments.

## Current State

A baseline evaluation has already been run. The agent achieved ~30% success rate on 89 tasks.
{data_access}

There is an analysis guide available in the context store under the `experiment-analysis`
namespace — read it before starting your analysis.

## Your Goal

Modify the agent's scaffolding code to improve its success rate. The agent codebase is in the
current working directory.

## Budget

You have a budget of **{sample_budget} samples** on the test split. Each sample you evaluate
counts against this budget. Plan your experiments carefully — you cannot afford to run the full
89-sample suite multiple times. Use targeted subsets to test hypotheses.

Use the `evaluate_commit` tool to run evaluations. You can specify which sample IDs to evaluate.

## What You Can Change

- Agent scaffolding code (prompts, tools, workflow, error handling, context management)
- How the agent interacts with the terminal environment
- Tool definitions and descriptions

## What You Cannot Change

- The underlying model (do not swap models)
- Task definitions and evaluation criteria
- The Harbor infrastructure or sandbox environments

## Approach

1. **Analyze first.** Understand why the agent fails before changing anything. Look at failing
   traces — are they crashes, timeouts, loops, or wrong answers?
2. **Fix bugs before optimizing.** If there are outright crashes or parsing errors, fix those
   first — they're free wins.
3. **Test targeted hypotheses.** Pick a small set of samples that exhibit the failure mode you're
   addressing. Evaluate on those specifically.
4. **Commit and evaluate.** After each change, commit your work and evaluate to measure impact.
5. **Be bold.** Structural changes (tools, workflow, error recovery) beat prompt tweaks.

## When You're Done

Provide the git commit hash of your best-performing version in your final response.
"""


# =============================================================================
# Policy construction
# =============================================================================


DEFAULT_WANDB_PROJECT = "vero-terminal-bench"


def make_policy(
    mode: str,
    sample_budget: int | None,
    model: str = "claude-sonnet-4-5-20250929",
    max_turns: int = 200,
    effort: str = "high",
    resume_session_id: str | None = None,
    fork_session_id: str | None = None,
    baseline_session_id: str | None = None,
    dataset_session_id: str | None = None,
    git_ref: str = "main",
    max_concurrency: int = 100,
    environment: str = "scale-sandbox",
    console_verbose: bool = False,
    enable_wandb: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
) -> Policy:
    """Build a Policy for Terminal Bench optimization.

    Args:
        mode: "tools" (ExperimentViewer/DatasetViewer MCP tools) or
              "artifacts" (filesystem artifacts in _vero/).
        sample_budget: Total number of samples the agent can evaluate on the test split.
              If None, uses the task default (50).
        model: Claude model to use.
        max_turns: Maximum agent turns.
        effort: Claude reasoning effort level.
        resume_session_id: Resume from existing session instead of forking baseline.
        fork_session_id: Fork from an existing session (copies experiments + project, fresh agent).
        git_ref: Git ref to start from (default: main). Used with --fork to start from a specific commit.
        console_verbose: Full JSON panels (True) or compact one-liners (False).
    """
    task = load_task(TASK_NAME)

    # --- Agent ---
    options = ClaudeAgentOptions(
        model=model,
        permission_mode="bypassPermissions",
        allowed_tools=["WebSearch", "WebFetch", "Task", "Bash"],
        effort=effort,  # type: ignore
    )

    if mode == "tools":
        tool_sets = [DatasetViewer(), ExperimentRunnerTool(), ExperimentViewer()]
        artifacts = []
    elif mode == "artifacts":
        tool_sets = [ExperimentRunnerTool()]
        artifacts = [TracesArtifact(), DatasetArtifact()]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'tools' or 'artifacts'.")

    agent = ClaudeCodeAgent(
        options=options,
        tool_sets=tool_sets,
        output_format=None,
    )

    # --- Budget (single "test" split) ---
    if sample_budget is not None:
        budget = [SplitBudget(split="test", total_sample_budget=sample_budget)]
    else:
        budget = task.budget

    # --- Split access: test is the only split, make it viewable ---
    split_accesses = [SplitAccess.viewable("test")]

    # --- Subprocess env: explicit vars forwarded to eval subprocesses ---
    subprocess_env_vars = [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_BASE",
        "LITELLM_BASE_URL",
        "LITELLM_API_KEY",
        "RUNLOOP_API_KEY",
        "DAYTONA_API_KEY",
    ]

    # --- Build policy ---
    common_kwargs = dict(
        agent=agent,
        task=task.task,
        budget=budget,
        split_accesses=split_accesses,
        artifacts=artifacts,
        skills=ANALYSIS_SKILL,
        subprocess_env_vars=subprocess_env_vars,
        evaluation_parameters=BaseEvaluationParameters(
            timeout=EVAL_TIMEOUT,
            sample_timeout=SAMPLE_TIMEOUT,
            max_concurrency=max_concurrency,
            task_params={"environment": environment},
        ),
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
        enable_console=True,
        console_verbose=console_verbose,
        max_turns=max_turns,
    )

    if fork_session_id:
        # Fork: new session seeded with experiments + project from source session.
        # The main repo in the source session has all commits from all worktrees,
        # so git_ref can point to any commit (including the agent's best).
        from vero.core.sessions import find_project_dir_in_session

        new_session_id = str(uuid4())
        new_session_dir = create_session_dir(new_session_id)

        # Copy experiments
        source_experiments = get_session_experiments_dir(fork_session_id)
        if source_experiments.exists():
            shutil.copytree(source_experiments, new_session_dir / "experiments")

        # Copy dataset mapping
        src_datasets = get_session_dir(fork_session_id) / "datasets.json"
        if src_datasets.exists():
            shutil.copy2(src_datasets, new_session_dir / "datasets.json")

        # Find the main repo (not a worktree) in the source session
        source_session_dir = get_session_dir(fork_session_id)
        # The main repo is the one whose .git is a directory, not a file
        main_repo = None
        for d in source_session_dir.iterdir():
            git_path = d / ".git"
            if d.is_dir() and git_path.exists() and git_path.is_dir():
                main_repo = d
                break

        if main_repo is None:
            raise ValueError(f"No git repo found in session {fork_session_id}")

        dest_project = new_session_dir / main_repo.name
        shutil.copytree(main_repo, dest_project)
        logger.info(f"Forked project from {fork_session_id}: {main_repo.name}")

        policy = Policy(
            project_path=str(dest_project),
            dataset=str(task.dataset_path),
            session_id=new_session_id,
            git_ref=git_ref,
            isolate=False,
            **common_kwargs,
        )
    elif resume_session_id:
        policy = Policy.resume(
            session_id=resume_session_id,
            dataset=str(task.dataset_path),
            restore_agent_state=False,
            **common_kwargs,
        )
    else:
        # Create session seeded with baseline experiment results
        new_session_id = str(uuid4())
        new_session_dir = create_session_dir(new_session_id)

        # Copy baseline experiments so the agent can see prior results
        if baseline_session_id:
            baseline_experiments = get_session_experiments_dir(baseline_session_id)
            if baseline_experiments.exists():
                shutil.copytree(baseline_experiments, new_session_dir / "experiments")

        # Copy dataset mapping (needed for DatasetViewer to find the dataset in global cache)
        if dataset_session_id:
            src_datasets = get_session_dir(dataset_session_id) / "datasets.json"
            if src_datasets.exists():
                shutil.copy2(src_datasets, new_session_dir / "datasets.json")

        policy = Policy(
            project_path=str(task.project_path),
            dataset=str(task.dataset_path),
            session_id=new_session_id,
            git_ref=git_ref,
            isolate=True,
            **common_kwargs,
        )

    return policy


# =============================================================================
# Main
# =============================================================================


async def run(
    policy: Policy,
    mode: str,
    sample_budget: int,
    skip_initial_eval: bool = True,
    skip_final_eval: bool = True,
) -> None:
    """Run the optimization loop.

    The agent drives its own evaluations via the evaluate_commit tool.
    Initial and final evals are optional orchestrator-level bookends.
    """
    prompt = build_prompt(mode=mode, sample_budget=sample_budget)

    async with policy:
        if not skip_initial_eval:
            initial = await policy.evaluate_commit(
                policy.session.base_commit, split="test"
            )
            print(f"Initial score: {initial.result.score()}")

        await policy.step(input=prompt)

        best = policy.get_best_non_baseline_commit()

        if not skip_final_eval and best.commit:
            final = await policy.evaluate_commit(best.commit, split="test")
            print(f"Final score: {final.result.score()}")

    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Session ID: {policy.session_id}")
    if best.commit:
        print(f"Best commit: {best.commit}")
        print(f"Best score:  {best.score}")
    else:
        print("No improvements found.")


async def eval_only(
    policy: Policy,
    commit: str,
    split: str = "test",
    sample_ids: list[int] | None = None,
) -> None:
    """Evaluate a specific commit without running the agent.

    Use this to run a full evaluation on the best commit from a previous
    agent run, after inspecting the changes.
    """
    async with policy:
        print(f"Evaluating commit {commit[:8]} on {split} split...")
        if sample_ids:
            print(f"  Sample IDs: {sample_ids}")
        experiment = await policy.evaluate_commit(
            commit,
            split=split,
            sample_ids=sample_ids,
        )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Session ID: {policy.session_id}")
    print(f"Commit: {commit}")
    print(f"Score: {experiment.result.score()}")
    print(f"Samples: {len(experiment.result.sample_results)}")
    print(f"Error rate: {experiment.result.error_rate()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminal Bench optimization experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tools", "artifacts"],
        default="tools",
        help="Information access mode: 'tools' (MCP viewers) or 'artifacts' (filesystem)",
    )
    parser.add_argument(
        "--sample-budget",
        type=int,
        default=None,
        help="Total number of samples the agent can evaluate (default: task default of 50)",
    )
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--effort", type=str, default="high", choices=["low", "medium", "high"]
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from session ID"
    )
    parser.add_argument(
        "--fork",
        type=str,
        default=None,
        help="Fork from session ID (copies experiments + project, fresh agent)",
    )
    parser.add_argument(
        "--baseline-session",
        type=str,
        default=None,
        help="Session ID with baseline experiments to seed from",
    )
    parser.add_argument(
        "--dataset-session",
        type=str,
        default=None,
        help="Session ID with dataset mapping (datasets.json)",
    )
    parser.add_argument(
        "--git-ref",
        type=str,
        default="main",
        help="Git ref to start from (e.g. a commit hash from a previous agent)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip agent loop, just evaluate a commit. Requires --resume and --commit.",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Commit hash to evaluate (used with --eval-only)",
    )
    parser.add_argument(
        "--sample-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample IDs to evaluate (used with --eval-only)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100,
        help="Max concurrent eval tasks (default: 100)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="scale-sandbox",
        help="Harbor environment type (scale-sandbox, modal, docker)",
    )
    parser.add_argument(
        "--run-initial-eval", action="store_true", help="Run baseline eval before agent"
    )
    parser.add_argument(
        "--run-final-eval",
        action="store_true",
        help="Run final eval on best commit after agent",
    )
    parser.add_argument(
        "--enable-wandb", action="store_true", help="Enable wandb logging"
    )
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--verbose", action="store_true", help="Verbose console output")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval_only:
        if not args.resume:
            print("Error: --eval-only requires --resume <session-id>")
            return
        if not args.commit:
            print("Error: --eval-only requires --commit <hash>")
            return

        policy = make_policy(
            mode=args.mode,
            sample_budget=args.sample_budget,
            resume_session_id=args.resume,
            max_concurrency=args.max_concurrency,
            environment=args.environment,
            console_verbose=args.verbose,
            enable_wandb=args.enable_wandb,
            wandb_project=args.wandb_project,
        )
        asyncio.run(
            eval_only(
                policy,
                commit=args.commit,
                sample_ids=args.sample_ids,
            )
        )
    else:
        # Resolve effective sample budget for the prompt
        effective_budget = args.sample_budget
        if effective_budget is None:
            task = load_task(TASK_NAME)
            if task.budget:
                effective_budget = task.budget[0].total_sample_budget
            else:
                effective_budget = 50

        policy = make_policy(
            mode=args.mode,
            sample_budget=args.sample_budget,
            model=args.model,
            max_turns=args.max_turns,
            effort=args.effort,
            resume_session_id=args.resume,
            fork_session_id=args.fork,
            baseline_session_id=args.baseline_session,
            dataset_session_id=args.dataset_session,
            git_ref=args.git_ref,
            max_concurrency=args.max_concurrency,
            environment=args.environment,
            console_verbose=args.verbose,
            enable_wandb=args.enable_wandb,
            wandb_project=args.wandb_project,
        )
        asyncio.run(
            run(
                policy,
                mode=args.mode,
                sample_budget=effective_budget,
                skip_initial_eval=not args.run_initial_eval,
                skip_final_eval=not args.run_final_eval,
            )
        )


if __name__ == "__main__":
    main()
