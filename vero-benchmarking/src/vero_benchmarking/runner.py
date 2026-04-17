#!/usr/bin/env python
"""
Vero benchmarking runner - optimization and baseline evaluation.

Usage:
    # Run optimization with VeroAgent
    python -m vero_benchmarking.runner vero --task gsm8k

    # Run optimization with ClaudeCodeAgent
    python -m vero_benchmarking.runner claude-code --task drop

    # Run baseline evaluation
    python -m vero_benchmarking.runner baseline --task math --models gpt-5.2-codex
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from vero.agents.claude_code import ClaudeCodeAgent
from vero.agents.vero import VeroAgent, default_tool_sets
from vero.core.evaluation import BaseEvaluationParameters
from vero.evaluator import run_evaluation
from vero.policy import BestVersion, Policy
from vero.tools import (
    BashTool,
    ContextStore,
    DatasetViewer,
    ExperimentRunnerTool,
    ExperimentViewer,
    FileRead,
    FileWrite,
    Grep,
    ResourceControl,
    WebFetch,
    WebSearch,
)

from vero_benchmarking.constants import DEFAULT_MANIFEST_PATH, DEFAULT_RESULTS_DIR
from vero_benchmarking.tasks import load_task
from vero_benchmarking.utils import get_model

# =============================================================================
# Constants
# =============================================================================

MODELS: dict[str, str] = {
    "sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "opus": "anthropic/claude-opus-4-5-20251101",
    "haiku": "anthropic/claude-haiku-4-5-20251001",
    "gpt": "gpt-5.2-codex",
}

DEFAULT_EVAL_TIMEOUT = 90

# =============================================================================
# Policy Factories
# =============================================================================


def make_vero_policy(
    model: str,
    task_name: str,
    enable_sub_agents: bool = True,
    enable_context_store: bool = False,
    use_resources_only: bool = False,
    orchestrator_mode: bool = False,
    disable_per_split_evaluation: bool = True,
    instructions_template: str = "instructions/few_shot_instructions.j2",
    prompt_template: str = "prompts/simple_prompt.j2",
    evaluation_parameters: BaseEvaluationParameters | None = None,
    **policy_kwargs: Any,
) -> Policy:
    """Create a Policy with a VeroAgent."""
    from agents import Agent as OAIAgent
    from agents import ModelSettings
    from vero.tools.sub_agent import SubAgentTool

    tools = default_tool_sets()

    if enable_context_store:
        tools.append(ContextStore())
    if use_resources_only:
        tools = [t for t in tools if not isinstance(t, FileWrite)]
        tools.append(ResourceControl())
    if orchestrator_mode:
        tools = [
            t
            for t in tools
            if not isinstance(
                t,
                (
                    BashTool,
                    DatasetViewer,
                    ExperimentViewer,
                    FileRead,
                    FileWrite,
                    Grep,
                    ResourceControl,
                    WebFetch,
                    WebSearch,
                ),
            )
        ]
    if not enable_sub_agents:
        tools = [t for t in tools if not isinstance(t, SubAgentTool)]
    if disable_per_split_evaluation:
        for t in tools:
            if isinstance(t, ExperimentRunnerTool):
                t.exclude_tools = ["evaluate_commit"]

    from agents import ModelRetrySettings, retry_policies

    agent = VeroAgent(
        oai_agent=OAIAgent(
            name="VeroAgent",
            model=get_model(model),
            model_settings=ModelSettings(
                include_usage=True,
                retry=ModelRetrySettings(
                    max_retries=4,
                    backoff={
                        "initial_delay": 0.5,
                        "max_delay": 5.0,
                        "multiplier": 2.0,
                        "jitter": True,
                    },
                    policy=retry_policies.any(
                        retry_policies.provider_suggested(),
                        retry_policies.retry_after(),
                        retry_policies.network_error(),
                        retry_policies.http_status([408, 409, 429, 500, 502, 503, 504]),
                    ),
                ),
            ),
        ),
        tool_sets=tools,
    )

    task = load_task(task_name)
    kwargs: dict[str, Any] = dict(
        project_path=task.project_path,
        dataset=task.dataset_path,
        task=task.task,
        prompt_kwargs={"batch_size": task.batch_size, "score_threshold": task.score_threshold},
        agent=agent,
        instructions_template=instructions_template,
        prompt_template=prompt_template,
        evaluation_parameters=evaluation_parameters or BaseEvaluationParameters(timeout=DEFAULT_EVAL_TIMEOUT),
        isolate=True,
    )
    # Budget: explicit list takes precedence over convenience fields
    if task.budget is not None:
        kwargs["budget"] = task.budget
    else:
        kwargs.update(
            train_budget=task.train_budget,
            validation_budget=task.validation_budget,
            train_sample_budget=task.train_sample_budget,
            validation_sample_budget=task.validation_sample_budget,
        )
    kwargs.update(policy_kwargs)
    return Policy(**kwargs)


def make_claude_code_policy(
    model: str,
    task_name: str,
    use_pure: bool = False,
    enable_context_store: bool = False,
    disable_per_split_evaluation: bool = True,
    instructions_template: str = "instructions/few_shot_instructions.j2",
    prompt_template: str = "prompts/simple_prompt.j2",
    evaluation_parameters: BaseEvaluationParameters | None = None,
    **policy_kwargs: Any,
) -> Policy:
    """Create a Policy with a ClaudeCodeAgent."""
    tool_sets = [DatasetViewer(), ExperimentRunnerTool(), ExperimentViewer()]
    if enable_context_store:
        tool_sets.append(ContextStore())
    if disable_per_split_evaluation:
        for t in tool_sets:
            if isinstance(t, ExperimentRunnerTool):
                t.exclude_tools = ["evaluate_commit"]

    from claude_agent_sdk import ClaudeAgentOptions

    claude_model = model.split("/")[-1] if "/" in model else model

    agent = ClaudeCodeAgent(
        options=ClaudeAgentOptions(model=claude_model, permission_mode="bypassPermissions"),
        tool_sets=[] if use_pure else tool_sets,
        enable_hooks=not use_pure,
        output_format=BestVersion if use_pure else None,
    )

    task = load_task(task_name)
    kwargs: dict[str, Any] = dict(
        project_path=task.project_path,
        dataset=task.dataset_path,
        task=task.task,
        prompt_kwargs={"batch_size": task.batch_size, "score_threshold": task.score_threshold},
        agent=agent,
        instructions_template=instructions_template,
        prompt_template=prompt_template,
        evaluation_parameters=evaluation_parameters or BaseEvaluationParameters(timeout=DEFAULT_EVAL_TIMEOUT),
        isolate=True,
    )
    # Budget: explicit list takes precedence over convenience fields
    if task.budget is not None:
        kwargs["budget"] = task.budget
    else:
        kwargs.update(
            train_budget=task.train_budget,
            validation_budget=task.validation_budget,
            train_sample_budget=task.train_sample_budget,
            validation_sample_budget=task.validation_sample_budget,
        )
    if use_pure:
        from vero.artifacts import RawDatasetArtifact, SkillsArtifact

        kwargs["filesystem_accesses"] = []
        kwargs["artifacts"] = [RawDatasetArtifact(), SkillsArtifact()]
    kwargs.update(policy_kwargs)

    return Policy(**kwargs)


# =============================================================================
# Manifest
# =============================================================================


def write_session_manifest(
    policy: Policy,
    best_commit: str | None = None,
    batch_id: str | None = None,
    config_name: str | None = None,
) -> None:
    """Write session mapping to local manifest file for backup/querying."""
    import fcntl

    manifest_path = DEFAULT_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "session_id": policy.session_id,
        "timestamp": datetime.now().isoformat(),
        "batch_id": batch_id,
        "config_name": config_name,
        "task": policy.task,
        "agent_type": type(policy.agent).__name__,
        "best_commit": best_commit,
    }

    with open(manifest_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    print(f"Session manifest written to: {manifest_path}")


# =============================================================================
# Optimization
# =============================================================================


class RunOptimizationOutput(NamedTuple):
    session_id: str
    best_commit: str | None


async def run_optimization(
    policy: Policy,
    batch_id: str | None = None,
    config_name: str | None = None,
    push_to_origin: bool = False,
    skip_initial_eval: bool = False,
    eval_split: str = "test",
) -> RunOptimizationOutput:
    """Run optimization and handle benchmarking-specific post-processing."""

    # Thread batch_id into policy metadata for wandb tracking
    if batch_id:
        policy.metadata["batch_id"] = batch_id
    if config_name:
        policy.metadata["config_name"] = config_name

    best = await policy.run(skip_initial_eval=skip_initial_eval, eval_split=eval_split)

    # Push to origin if requested
    if push_to_origin and policy.session.workspace:
        print(f"\n--- Pushing branch {policy.session.base_branch} to origin ---")
        try:
            import subprocess as _sp

            _sp.run(
                ["git", "push", "origin", policy.session.base_branch],
                cwd=policy.session.workspace.root,
                capture_output=True, check=True,
            )
            print(f"Successfully pushed {policy.session.base_branch} to origin")
        except Exception as e:
            print(f"Warning: Failed to push to origin: {e}")

    # Write session manifest
    write_session_manifest(
        policy, best.commit, batch_id=batch_id, config_name=config_name
    )

    return RunOptimizationOutput(session_id=policy.session_id, best_commit=best.commit)


# =============================================================================
# Baseline Evaluation
# =============================================================================


def print_baseline_stats(df: pd.DataFrame, model: str) -> None:
    """Print summary statistics for a model's baseline results."""
    print(f"\n=== {model} ===")
    print(f"  Samples: {len(df)}")
    if "score" in df.columns:
        scores = df["score"].dropna()
        print(f"  Mean score: {scores.mean():.4f}")
        print(f"  Std score:  {scores.std():.4f}")
        print(f"  Min score:  {scores.min():.4f}")
        print(f"  Max score:  {scores.max():.4f}")
        print(f"  Null scores: {df['score'].isna().sum()}")
    if "error" in df.columns:
        print(f"  Errors: {df['error'].notna().sum()}")


async def run_baseline(
    project_path: str | Path,
    dataset: str | Path,
    task: str,
    models: list[str],
    split: str = "test",
    commit: str | None = None,
    sample_ids: list[str] | None = None,
    output_path: Path | None = None,
    create_temporary_copy: bool = False,
) -> pd.DataFrame:
    """Run baseline evaluations for multiple models (no optimization loop)."""
    dataset_path = Path(dataset)
    dataset_id = dataset_path.stem

    if commit is None:
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "main"], cwd=project_path,
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    run_evaluation_kwargs = dict(
        project_path=str(project_path),
        dataset_path=str(dataset_path),
        split=split,
        commit=commit,
        task=task,
        sample_ids=sample_ids,
        create_temporary_copy=create_temporary_copy,
    )

    print(f"Task: {dataset_id}")
    print(f"Split: {split}")
    print(f"Commit: {commit}")
    print(f"Models: {models}")

    all_dfs = []
    for model in models:
        print(f"\n--- Evaluating model: {model} ---")
        try:
            result = await run_evaluation(
                task_params={"model": model}, **run_evaluation_kwargs
            )
            df = result.sample_results_df()
            df["model"] = model
            df["task"] = dataset_id
            df["split"] = split
            df["commit"] = commit
            print_baseline_stats(df, model)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error evaluating {model}: {e}")

    if not all_dfs:
        print("No successful evaluations.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_RESULTS_DIR / f"{dataset_id}_{split}_{timestamp}.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_path)
    print(f"\nResults saved to: {output_path}")
    return combined_df


# =============================================================================
# CLI
# =============================================================================


def _build_evaluation_parameters(args: argparse.Namespace) -> BaseEvaluationParameters:
    """Build evaluation parameters from CLI args."""
    overrides = {}
    if args.run_inference_in_thread:
        overrides["run_inference"] = True
    if args.run_evaluation_in_thread:
        overrides["run_evaluation"] = True

    return BaseEvaluationParameters(
        max_concurrency=args.evaluation_concurrency,
        timeout=args.evaluation_timeout,
        run_in_thread_overrides=overrides if overrides else {},
    )


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments for optimization subcommands."""
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-5-20250929",
        help="Model to use",
    )
    parser.add_argument(
        "--enable-wandb", action="store_true", help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vero-benchmarking",
        help="Wandb project name",
    )
    parser.add_argument(
        "--max-turns", type=int, default=200, help="Maximum turns for optimization loop"
    )
    parser.add_argument(
        "--git-ref", type=str, default="main", help="Branch or commit to start from"
    )
    parser.add_argument(
        "--instructions-template", type=str, help="Path to instructions template"
    )
    parser.add_argument("--prompt-template", type=str, help="Path to prompt template")
    parser.add_argument(
        "--evaluation-concurrency",
        type=int,
        default=100,
        help="Concurrency for evaluation",
    )
    parser.add_argument(
        "--evaluation-timeout",
        type=int,
        default=180,
        help="Timeout per sample in seconds",
    )
    parser.add_argument(
        "--run-inference-in-thread",
        action="store_true",
        help="Run inference in separate threads",
    )
    parser.add_argument(
        "--run-evaluation-in-thread",
        action="store_true",
        help="Run evaluation in separate threads",
    )
    parser.add_argument(
        "--sgp-account-id",
        type=str,
        default=None,
        help="SGP agents SDK tracing account ID",
    )
    parser.add_argument(
        "--session-id", type=str, default=None, help="Session ID (UUID)"
    )
    parser.add_argument("--batch-id", type=str, default=None, help="Batch identifier")
    parser.add_argument(
        "--push-to-origin",
        action="store_true",
        help="Push best commit branch to origin",
    )
    parser.add_argument(
        "--enable-per-split-evaluation", action="store_true", default=False,
        help="Enable per-split evaluation tool (default: disabled)"
    )
    parser.add_argument(
        "--skip-initial-eval",
        action="store_true",
        help="Skip initial baseline evaluation",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file for the optimizer process (LLM API keys, etc.)",
    )
    parser.add_argument(
        "--subprocess-env-file",
        type=str,
        default=None,
        help="Path to .env file for evaluation subprocesses",
    )


def add_vero_args(parser: argparse.ArgumentParser) -> None:
    """Add Vero-specific arguments."""
    add_shared_args(parser)
    parser.add_argument("--enable-sub-agents", action="store_true", default=True)
    parser.add_argument("--disable-sub-agents", action="store_true")
    parser.add_argument("--enable-context-store", action="store_true")
    parser.add_argument("--use-resources-only", action="store_true")
    parser.add_argument("--orchestrator-mode", action="store_true")


def add_claude_code_args(parser: argparse.ArgumentParser) -> None:
    """Add Claude Code-specific arguments."""
    add_shared_args(parser)
    parser.add_argument("--permission-mode", type=str, default="bypassPermissions")
    parser.add_argument("--use-pure", action="store_true")
    parser.add_argument("--enable-context-store", action="store_true")


def add_baseline_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the baseline subcommand."""
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--models", type=str, nargs="+", required=True, help="List of model names"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "validation", "test"]
    )
    parser.add_argument(
        "--commit", type=str, default=None, help="Git commit to evaluate"
    )
    parser.add_argument("--sample-ids", type=str, nargs="*", default=None)
    parser.add_argument(
        "--output", type=str, default=None, help="Output parquet file path"
    )
    parser.add_argument("--create-temporary-worktree", action="store_true")


def run_vero_command(args: argparse.Namespace) -> None:
    """Handle the vero subcommand."""
    policy = make_vero_policy(
        model=args.model,
        task_name=args.task,
        enable_sub_agents=args.enable_sub_agents and not args.disable_sub_agents,
        enable_context_store=args.enable_context_store,
        use_resources_only=args.use_resources_only,
        orchestrator_mode=args.orchestrator_mode,
        disable_per_split_evaluation=not args.enable_per_split_evaluation,
        instructions_template=args.instructions_template or "instructions/few_shot_instructions.j2",
        prompt_template=args.prompt_template or "prompts/simple_prompt.j2",
        max_turns=args.max_turns,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        ref=args.git_ref,
        evaluation_parameters=_build_evaluation_parameters(args),
        session_id=args.session_id,
        optimizer_env_file=args.env_file,
        subprocess_env_vars=args.subprocess_env_file,
    )
    asyncio.run(
        run_optimization(
            policy,
            batch_id=args.batch_id,
            push_to_origin=args.push_to_origin,
            skip_initial_eval=args.skip_initial_eval,
        )
    )


def run_claude_code_command(args: argparse.Namespace) -> None:
    """Handle the claude-code subcommand."""
    policy = make_claude_code_policy(
        model=args.model,
        task_name=args.task,
        use_pure=args.use_pure,
        enable_context_store=args.enable_context_store,
        disable_per_split_evaluation=not args.enable_per_split_evaluation,
        instructions_template=args.instructions_template or "instructions/few_shot_instructions.j2",
        prompt_template=args.prompt_template or "prompts/claude_code_prompt.j2",
        max_turns=args.max_turns,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        ref=args.git_ref,
        evaluation_parameters=_build_evaluation_parameters(args),
        session_id=args.session_id,
        optimizer_env_file=args.env_file,
        subprocess_env_vars=args.subprocess_env_file,
    )
    asyncio.run(
        run_optimization(
            policy,
            batch_id=args.batch_id,
            push_to_origin=args.push_to_origin,
            skip_initial_eval=args.skip_initial_eval,
        )
    )


def run_baseline_command(args: argparse.Namespace) -> None:
    """Handle the baseline subcommand."""
    task = load_task(args.task)
    print(f"Loaded task: {task}")

    output_path = Path(args.output) if args.output else None

    asyncio.run(
        run_baseline(
            project_path=task.project_path,
            dataset=task.dataset_path,
            task=task.task,
            models=args.models,
            split=args.split,
            commit=args.commit,
            sample_ids=args.sample_ids,
            output_path=output_path,
            create_temporary_copy=args.create_temporary_copy,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Vero benchmarking runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    vero_parser = subparsers.add_parser(
        "vero", help="Run Vero optimization (VeroAgent)"
    )
    add_vero_args(vero_parser)

    claude_code_parser = subparsers.add_parser(
        "claude-code", help="Run Claude Code optimization (ClaudeCodeAgent)"
    )
    add_claude_code_args(claude_code_parser)

    baseline_parser = subparsers.add_parser("baseline", help="Run baseline evaluations")
    add_baseline_args(baseline_parser)

    args = parser.parse_args()

    # Setup SGP tracing if configured
    if hasattr(args, "sgp_account_id") and args.sgp_account_id:
        from vero.logging import setup_sgp_agents_sdk_tracing

        setup_sgp_agents_sdk_tracing(account_id=args.sgp_account_id)

    if args.command == "vero":
        run_vero_command(args)
    elif args.command == "claude-code":
        run_claude_code_command(args)
    elif args.command == "baseline":
        run_baseline_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
