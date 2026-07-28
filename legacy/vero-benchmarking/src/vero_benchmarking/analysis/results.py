"""Extract and process benchmark results from Weights & Biases.

This module extracts optimization run data from W&B, processes history metrics,
and produces standardized DataFrames for downstream analysis and plotting.

Typical usage:
    import wandb
    api = wandb.Api()
    runs = list(api.runs("your-project"))
    df = build_run_df_with_history(runs)
    extract_primary_fields(df)
    df = add_performance_metrics(df, task_to_split_map)
    df = filter_quality(df)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm


# =============================================================================
# Column definitions
# =============================================================================

DISPLAY_COLS = [
    "run_id",
    "optimizer_scaffold",
    "policy_type",
    "task",
    "model",
    "session_id",
    "base_commit",
    "final_commit",
    "initial_score",
    "best_score",
    "num_evals",
    "initial_commit",
    "best_commit",
    "initial_error_rate",
    "best_error_rate",
]

DEFAULT_TASK_TO_SPLIT = {
    "math": "test_history",
    "gpqa": "test_history",
    "simple_qa": "test_history",
    "gaia": "validation_history",
    "retail": "test_history",
}


# =============================================================================
# W&B extraction helpers
# =============================================================================


def default_columns() -> list[str]:
    return ["score", "num_samples", "candidate_commit", "error_rate"]


def get_default_columns_map(prefix: str) -> dict[str, str]:
    cols = default_columns()
    return {f"{prefix}/{c}": c for c in cols}


def extract_history_metrics(
    hist_df: pd.DataFrame, column_map: dict[str, str] | str
) -> list[dict]:
    """Extract metrics for a given prefix (train/validation/test) as list of dicts."""
    if isinstance(column_map, str):
        column_map = get_default_columns_map(column_map)

    available = [c for c in column_map if c in hist_df.columns]
    if not available:
        return []
    subset = hist_df[available].dropna(how="all")
    records = []
    for _, row in subset.iterrows():
        record = {}
        for col, key in column_map.items():
            record[key] = row.get(col)
        if any(pd.notna(v) for v in record.values()):
            records.append(record)
    return records


def get_nested(config: dict, *keys, default: Any = None) -> Any:
    """Deep dict getter for nested W&B config."""
    val = config
    for k in keys:
        try:
            val = val.get(k)
        except (KeyError, AttributeError):
            return default
    return val


# =============================================================================
# DataFrame construction
# =============================================================================


def build_run_df_with_history(runs: list) -> pd.DataFrame:
    """Build a DataFrame with one row per W&B run, including history metrics.

    Args:
        runs: List of wandb.Run objects.

    Returns:
        DataFrame with columns: run_id, name, state, created_at, config, summary,
        best_results, train_history, validation_history, test_history.
    """
    data = []
    for run in tqdm(runs, desc="Loading runs"):
        hist_df = run.history()
        row = {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "config": dict(run.config),
            "summary": dict(run.summary),
            "best_results": dict(run.summary.get("best_results", {})),
            "train_history": extract_history_metrics(hist_df, "train"),
            "validation_history": extract_history_metrics(hist_df, "validation"),
            "test_history": extract_history_metrics(hist_df, "test"),
        }
        data.append(row)

    return pd.DataFrame(data)


def extract_primary_fields(df: pd.DataFrame) -> None:
    """Extract nested config fields to top-level columns (modifies df in place).

    Extracts: base_branch, base_commit, final_commit, model, session_id,
    optimizer_scaffold, policy_type, task.
    """
    extractions = {
        "base_branch": ("summary", "config", "base_branch"),
        "base_commit": ("summary", "config", "base_commit"),
        "final_commit": ("summary", "config", "final_commit"),
        "model": ("summary", "config", "model"),
        "session_id": ("summary", "config", "session_id"),
        "optimizer_scaffold": ("config", "vero-benchmarking-config", "name"),
        "policy_type": ("config", "vero-benchmarking-config", "policy_type"),
        "task": ("config", "vero-benchmarking-config", "task", "task"),
    }

    for col, keys in extractions.items():
        src = keys[0]
        df[col] = df[src].apply(lambda x, k=keys[1:]: get_nested(x, *k))


# =============================================================================
# Performance extraction
# =============================================================================


def extract_performance(row: pd.Series) -> dict:
    """Extract initial/best scores and commits from a run's history.

    Requires 'performance_dimension' and 'base_commit' columns.
    """
    col = row["performance_dimension"]
    num_evals = len(row[col])

    if num_evals < 2:
        return {
            "initial_score": None,
            "best_score": None,
            "num_evals": num_evals,
            "initial_commit": None,
            "best_commit": None,
            "initial_error_rate": None,
            "best_error_rate": None,
        }

    base_commit = row["base_commit"]
    initial_score = None
    initial_commit = None
    best_score = None
    best_commit = None
    best_error_rate = None
    initial_error_rate = None

    for d in row[col]:
        commit = d.get("candidate_commit")
        if commit == base_commit:
            if initial_score is None or d["score"] > initial_score:
                initial_score = d["score"]
                initial_commit = commit
                initial_error_rate = d["error_rate"]
        elif commit is not None:
            if best_score is None or d["score"] > best_score:
                best_score = d["score"]
                best_commit = commit
                best_error_rate = d["error_rate"]

    return {
        "initial_score": initial_score,
        "best_score": best_score,
        "num_evals": num_evals,
        "initial_commit": initial_commit,
        "best_commit": best_commit,
        "initial_error_rate": initial_error_rate,
        "best_error_rate": best_error_rate,
    }


def check_row_quality(
    row: pd.Series, error_rate_threshold: float = 0.15
) -> list[str]:
    """Check quality of a run row, returning list of issue tags."""
    tags = []
    if pd.isna(row["base_commit"]):
        tags.append("initial_commit_missing")
    if pd.isna(row["final_commit"]):
        tags.append("final_commit_missing")

    history = row[row["performance_dimension"]]
    if len(history) < 2:
        tags.append("insufficient_history")

    if row["best_error_rate"] and row["best_error_rate"] > error_rate_threshold:
        tags.append("high_best_error_rate")
    if row["initial_error_rate"] and row["initial_error_rate"] > error_rate_threshold:
        tags.append("high_initial_error_rate")

    return tags


def add_performance_metrics(
    df: pd.DataFrame,
    task_to_split_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add performance metrics and quality tags to the DataFrame.

    Filters to tasks in the split map, extracts initial/best scores,
    and flags bad runs.

    Args:
        df: DataFrame from build_run_df_with_history + extract_primary_fields.
        task_to_split_map: Maps task name to history column (e.g. "test_history").
            Defaults to DEFAULT_TASK_TO_SPLIT.

    Returns:
        DataFrame with added columns: performance_dimension, initial_score,
        best_score, num_evals, quality_tags, bad_run, etc.
    """
    if task_to_split_map is None:
        task_to_split_map = DEFAULT_TASK_TO_SPLIT

    df = df[df["task"].isin(task_to_split_map)].copy()
    df["performance_dimension"] = df["task"].map(task_to_split_map)

    perf_cols = df.apply(extract_performance, axis=1, result_type="expand")
    for col in perf_cols.columns:
        df[col] = perf_cols[col]

    df["quality_tags"] = df.apply(check_row_quality, axis=1)
    df["bad_run"] = df["quality_tags"].apply(bool)

    return df


def filter_quality(
    df: pd.DataFrame,
    max_per_group: int = 3,
) -> pd.DataFrame:
    """Filter out bad runs and keep only the most recent per group.

    Args:
        df: DataFrame with bad_run column.
        max_per_group: Keep at most N runs per (optimizer_scaffold, task, model).

    Returns:
        Filtered DataFrame with lift column added.
    """
    filtered = df[~df["bad_run"]].copy()

    # Fill missing model
    filtered["model"] = filtered["model"].fillna("anthropic/claude-sonnet-4-5-20250929")

    # Keep only most recent N runs per group
    filtered = (
        filtered.sort_values("created_at", ascending=False)
        .groupby(["optimizer_scaffold", "task", "model"])
        .head(max_per_group)
    )

    # Add derived metrics
    filtered["avg_initial_score_by_task"] = filtered.groupby("task")[
        "initial_score"
    ].transform("mean")
    filtered["lift"] = filtered["best_score"] - filtered["initial_score"]

    return filtered
