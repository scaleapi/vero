"""
Batch evaluation runner - run evaluations from a DataFrame specification.

Usage (file mode):
    python -m vero_benchmarking.eval \
        --input evaluations.csv \
        --prefix baseline_eval \
        --n-iterations 3

Usage (directory mode - for resumable runs):
    python -m vero_benchmarking.eval \
        --input /path/to/eval_dir \
        --n-iterations 3

    Directory must contain manifest.csv with columns: task, model, commit, split
    Results (summary.parquet, sample_results/) will be saved in the same directory.

Usage (dry run - see what would run):
    python -m vero_benchmarking.eval \
        --input /path/to/eval_dir \
        --dry-run

Input CSV/manifest must have columns: task, model, commit, split
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from vero.evaluator import run_evaluation

from vero_benchmarking.constants import DEFAULT_RESULTS_DIR
from vero_benchmarking.tasks import load_task


def generate_output_dir(prefix: str, base_dir: Path | None = None) -> Path:
    """Generate output directory with prefix and random suffix."""
    if base_dir is None:
        base_dir = DEFAULT_RESULTS_DIR

    suffix = secrets.token_hex(3)  # 6 character hex string
    output_dir = base_dir / f"{prefix}_{suffix}"
    return output_dir


def get_completed_keys(summary_path: Path) -> set[tuple[str, str, str, str]]:
    """Load completed evaluation keys from summary file."""
    if not summary_path.exists():
        return set()

    existing_summary = pd.read_parquet(summary_path)
    return set(
        zip(
            existing_summary["task"],
            existing_summary["model"],
            existing_summary["commit"],
            existing_summary["split"],
        )
    )


def print_dry_run_summary(
    input_df: pd.DataFrame,
    output_dir: Path,
    n_iterations: int,
) -> None:
    """Print a summary of what would be run without actually running."""
    summary_path = output_dir / "summary.parquet"
    completed_keys = get_completed_keys(summary_path)
    has_model_alias = "model_alias" in input_df.columns

    print("\n" + "=" * 60)
    print("DRY RUN SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Iterations per evaluation: {n_iterations}")
    print(f"Total evaluations in manifest: {len(input_df)}")
    print(f"Already completed: {len(completed_keys)}")
    print()

    pending = []
    skipped = []

    for _, row in input_df.iterrows():
        task = row["task"]
        model = row["model"]
        commit = row["commit"]
        split = row["split"]
        model_alias = row["model_alias"] if has_model_alias else model
        key = (task, model_alias, commit, split)

        entry = f"{task} | {model_alias} | {commit[:8]} | {split}"
        if key in completed_keys:
            skipped.append(entry)
        else:
            pending.append(entry)

    if skipped:
        print(f"SKIPPED ({len(skipped)} - already complete):")
        for entry in skipped:
            print(f"  [SKIP] {entry}")
        print()

    if pending:
        print(f"PENDING ({len(pending)} - will run):")
        for entry in pending:
            print(f"  [RUN]  {entry}")
        print()
    else:
        print("Nothing to run - all evaluations complete!")
        print()

    print("=" * 60)
    print(f"Summary: {len(pending)} to run, {len(skipped)} to skip")
    print("=" * 60)


def resolve_input_and_output(
    input_path: str,
    output_dir_arg: str | None,
    prefix: str,
) -> tuple[pd.DataFrame, Path]:
    """
    Resolve input DataFrame and output directory from input path.

    Supports two modes:
    1. File mode: input_path is a CSV/parquet file
       - Creates new output_dir or uses --output-dir
    2. Directory mode: input_path is a directory containing manifest.csv
       - Uses the same directory for output (resume-friendly)

    Args:
        input_path: Path to input file or directory
        output_dir_arg: Explicit output directory (overrides default)
        prefix: Prefix for generated output directory name

    Returns:
        Tuple of (input_df, output_dir)
    """
    path = Path(input_path)

    if path.is_dir():
        # Directory mode - look for manifest.csv, use same dir for output
        manifest_path = path / "manifest.csv"
        if not manifest_path.exists():
            raise ValueError(f"No manifest.csv found in {path}")
        input_df = pd.read_csv(manifest_path)
        output_dir = path
        print(f"Directory mode: using {path} for input and output")
    else:
        # File mode - load file, create/use output_dir
        if not path.exists():
            raise ValueError(f"Input file not found: {path}")

        if path.suffix == ".csv":
            input_df = pd.read_csv(path)
        elif path.suffix == ".parquet":
            input_df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}")

        if output_dir_arg:
            output_dir = Path(output_dir_arg)
        else:
            output_dir = generate_output_dir(prefix)
        print(f"File mode: input from {path}, output to {output_dir}")

    return input_df, output_dir


def get_eval_filename(
    task: str, model: str, commit: str, split: str, iteration: int
) -> str:
    """Generate filename for sample results."""
    # Sanitize model name (replace / with _)
    safe_model = model.replace("/", "_")
    # Use short commit hash
    short_commit = commit[:8] if len(commit) > 8 else commit
    return f"{task}_{safe_model}_{short_commit}_{split}_iter{iteration}.parquet"


async def run_single_evaluation(
    task_name: str,
    model: str,
    commit: str,
    split: str,
    hooks: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    use_copy: bool = True,
) -> dict[str, Any]:
    """Run a single evaluation and return results."""
    task = load_task(task_name)

    # Build extra dict with model and any additional params
    eval_extra = {"model": model}
    if extra:
        eval_extra.update(extra)

    result = await run_evaluation(
        project_path=str(task.project_path),
        dataset=str(task.dataset_path),
        split=split,
        commit=commit,
        task=task.task,
        task_params=eval_extra,
        create_temporary_copy=use_copy,
        hooks=hooks,
    )

    sample_df = result.sample_results_df()

    assert sample_df is not None, "Sample DF is empty!"

    # Extract metrics
    # SampleResult.as_pandas_series() adds 'is_error' column which checks:
    # error, eval_error, score is None, or error_traceback
    num_samples = len(sample_df)
    error_count = sample_df["is_error"].sum() if "is_error" in sample_df.columns else 0
    error_rate = error_count / num_samples if num_samples > 0 else 0.0

    # Calculate score - TaskResult.score is the standard column
    if "score" in sample_df.columns:
        score = sample_df["score"].mean()
    else:
        score = None

    return {
        "score": score,
        "num_samples": num_samples,
        "error_rate": error_rate,
        "sample_df": sample_df,
    }


async def run_batch_evaluations(
    input_df: pd.DataFrame,
    output_dir: Path,
    n_iterations: int = 1,
    continue_on_error: bool = True,
    hooks: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    use_copy: bool = True,
) -> pd.DataFrame:
    """
    Run batch evaluations from a DataFrame specification.

    Args:
        input_df: DataFrame with columns: task, model, commit, split
        output_dir: Directory to save results
        n_iterations: Number of iterations per evaluation
        continue_on_error: Whether to continue on evaluation errors
        hooks: List of hook names to execute (e.g., ["configure_litellm"])
        extra: Extra parameters to pass to evaluations (merged with model)
        use_copy: Whether to create temporary copies for each commit

    Returns:
        Summary DataFrame with aggregated results
    """
    # Validate input columns
    required_cols = {"task", "model", "commit", "split"}
    if not required_cols.issubset(input_df.columns):
        missing = required_cols - set(input_df.columns)
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    # Check for optional model_alias column
    has_model_alias = "model_alias" in input_df.columns

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_results_dir = output_dir / "sample_results"
    sample_results_dir.mkdir(exist_ok=True)

    summary_path = output_dir / "summary.parquet"

    # Load existing summary if resuming
    completed_keys = get_completed_keys(summary_path)
    if completed_keys:
        existing_summary = pd.read_parquet(summary_path)
        print(f"Resuming: {len(completed_keys)} evaluations already complete")
    else:
        existing_summary = None

    results = []
    total = len(input_df)

    for idx, row in input_df.iterrows():
        task = row["task"]
        model = row["model"]  # actual model string for execution
        commit = row["commit"]
        split = row["split"]
        # model_alias for tracking - defaults to model if not present
        model_alias = row["model_alias"] if has_model_alias else model

        key = (task, model_alias, commit, split)

        # Skip if already complete
        if key in completed_keys:
            print(
                f"[{idx + 1}/{total}] SKIP {task} | {model_alias} | {commit[:8]} | {split} (already complete)"
            )
            continue

        print(f"\n[{idx + 1}/{total}] {task} | {model_alias} | {commit[:8]} | {split}")
        print("=" * 60)

        scores = []
        num_samples_list = []
        error_rates = []
        iteration_files = []
        timestamps = []
        status = "complete"

        for i in range(n_iterations):
            print(f"  Iteration {i + 1}/{n_iterations}...")
            timestamp = datetime.now().isoformat()

            try:
                eval_result = await run_single_evaluation(
                    task,
                    model,
                    commit,
                    split,
                    hooks=hooks,
                    extra=extra,
                    use_copy=use_copy,
                )

                scores.append(eval_result["score"])
                num_samples_list.append(eval_result["num_samples"])
                error_rates.append(eval_result["error_rate"])
                timestamps.append(timestamp)

                # Save sample results
                filename = get_eval_filename(task, model, commit, split, i + 1)
                filepath = sample_results_dir / filename
                eval_result["sample_df"].to_parquet(filepath)
                iteration_files.append(str(filepath.relative_to(output_dir)))

                print(
                    f"    Score: {eval_result['score']:.4f}, Samples: {eval_result['num_samples']}, Errors: {eval_result['error_rate']:.2%}"
                )

            except Exception as e:
                print(f"    ERROR: {e}")
                if not continue_on_error:
                    raise
                status = "partial" if scores else "failed"
                timestamps.append(timestamp)

        # Compute aggregates
        if scores:
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
            mean_error_rate = statistics.mean(error_rates)
        else:
            mean_score = None
            std_score = None
            mean_error_rate = None

        result_row = {
            "task": task,
            "model": model_alias,  # use alias for tracking/display
            "commit": commit,
            "split": split,
            "scores": scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "num_samples": num_samples_list,
            "error_rates": error_rates,
            "mean_error_rate": mean_error_rate,
            "iteration_files": iteration_files,
            "timestamps": timestamps,
            "status": status,
            "n_iterations_completed": len(scores),
            "n_iterations_requested": n_iterations,
        }
        results.append(result_row)

        # Incremental save
        current_df = pd.DataFrame(results)
        if existing_summary is not None:
            current_df = pd.concat([existing_summary, current_df], ignore_index=True)
        current_df.to_parquet(summary_path)
        print(f"  Saved to {summary_path}")

    # Final summary
    if existing_summary is not None:
        final_df = pd.concat(
            [existing_summary, pd.DataFrame(results)], ignore_index=True
        )
    else:
        final_df = pd.DataFrame(results)

    final_df.to_parquet(summary_path)

    print("\n" + "=" * 60)
    print(f"COMPLETE: {len(final_df)} evaluations")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return final_df


def main():
    parser = argparse.ArgumentParser(
        description="Run batch evaluations from a CSV/parquet specification"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (CSV/parquet) or directory containing manifest.csv",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="batch_eval",
        help="Prefix for output directory name (default: batch_eval). Ignored in directory mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory path (overrides --prefix). Ignored in directory mode.",
    )
    parser.add_argument(
        "--n-iterations",
        "-n",
        type=int,
        default=1,
        help="Number of iterations per evaluation (default: 1)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running on evaluation errors",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running",
    )
    parser.add_argument(
        "--hook",
        action="append",
        dest="hooks",
        default=[],
        help="Hook name to execute (can be specified multiple times, e.g., --hook configure_litellm)",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default=None,
        help="JSON string of extra parameters to pass to evaluations",
    )
    parser.add_argument(
        "--no-worktree",
        action="store_true",
        help="Don't create temporary worktrees (run in current working directory)",
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        dest="skip_models",
        default=[],
        help="Skip models matching this substring (can be specified multiple times, e.g., --skip-model qwen)",
    )
    parser.add_argument(
        "--interleave-models",
        action="store_true",
        help="Interleave runs by model to spread API load (Claude, Gemini, GPT, Claude, ...)",
    )

    args = parser.parse_args()

    # Resolve input and output paths
    input_df, output_dir = resolve_input_and_output(
        args.input, args.output_dir, args.prefix
    )

    print(f"Loaded {len(input_df)} evaluation specs")
    print(f"Output directory: {output_dir}")

    # Filter out skipped models
    has_model_alias = "model_alias" in input_df.columns
    model_col = "model_alias" if has_model_alias else "model"
    if args.skip_models:
        original_count = len(input_df)
        for pattern in args.skip_models:
            input_df = input_df[~input_df[model_col].str.contains(pattern, case=False)]
        print(
            f"Skipping models matching {args.skip_models}: {original_count} -> {len(input_df)} specs"
        )

    # Interleave models to spread API load
    if args.interleave_models:
        # Assign a round-robin index within each model group
        input_df = input_df.copy()
        input_df["_model_rank"] = input_df.groupby(model_col).cumcount()
        input_df = input_df.sort_values(["_model_rank", model_col]).drop(
            columns=["_model_rank"]
        )
        input_df = input_df.reset_index(drop=True)
        print("Interleaving models to spread API load")

    # Parse extra JSON if provided
    extra = None
    if args.extra:
        import json

        extra = json.loads(args.extra)

    # Dry run mode - just show what would be run
    if args.dry_run:
        print_dry_run_summary(input_df, output_dir, args.n_iterations)
        return

    # Run evaluations
    asyncio.run(
        run_batch_evaluations(
            input_df=input_df,
            output_dir=output_dir,
            n_iterations=args.n_iterations,
            continue_on_error=args.continue_on_error,
            hooks=args.hooks or None,
            extra=extra,
            use_copy=not args.no_worktree,
        )
    )


if __name__ == "__main__":
    main()
