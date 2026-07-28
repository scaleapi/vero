"""Data loading and filtering for analysis module."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from vero.core.sessions import VERO_SESSIONS_DIR

DatasetSplitT = Literal["train", "test", "validation"]

from vero_benchmarking.constants import DEFAULT_RESULTS_DIR

from .config import DEFAULT_SCAFFOLDS


def load_analyses() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all analysis DataFrames and metadata.

    Returns:
        Tuple of (analyses_dict, metadata_df) where:
        - analyses_dict: Dict mapping session_id -> analysis DataFrame
        - metadata_df: DataFrame with session metadata (task, scaffold, model, etc.)
    """
    # Load metadata
    metadata_path = DEFAULT_RESULTS_DIR / "benchmark_results.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path, index_col=0)

    # Load analyses from cache
    analyses = {}
    for session_id in metadata["session_id"]:
        cache_path = VERO_SESSIONS_DIR / session_id / "analysis_df.csv"
        if cache_path.exists():
            analyses[session_id] = pd.read_csv(cache_path)

    return analyses, metadata


def filter_data(
    analyses: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    scaffolds: set[str] | None = None,
    tasks: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Filter analyses and metadata.

    Args:
        analyses: Dict of session_id -> analysis DataFrame
        metadata: Metadata DataFrame
        scaffolds: Scaffold names to include. None = use DEFAULT_SCAFFOLDS.
        tasks: Task names to include. None = no filtering.

    Returns:
        Filtered (analyses, metadata) tuple
    """
    filtered_metadata = metadata

    if scaffolds is None:
        scaffolds = DEFAULT_SCAFFOLDS
    filtered_metadata = filtered_metadata[
        filtered_metadata["optimizer_scaffold"].isin(scaffolds)
    ]

    if tasks is not None:
        filtered_metadata = filtered_metadata[filtered_metadata["task"].isin(tasks)]

    filtered_session_ids = set(filtered_metadata["session_id"])
    filtered_analyses = {k: v for k, v in analyses.items() if k in filtered_session_ids}

    return filtered_analyses, filtered_metadata


def _get_score_column(score_type: DatasetSplitT) -> str:
    """Get the column name for a score type."""
    return f"{score_type}_mean_score"


def get_session_score(
    analysis_df: pd.DataFrame,
    score_type: DatasetSplitT,
    phase: Literal["initial", "final", "best"],
    fallback: DatasetSplitT | None = None,
) -> float:
    """Get score for a session at a specific phase.

    Args:
        analysis_df: Analysis DataFrame for a session
        score_type: Which score column to use ("train", "validation", "test")
        phase: Which phase to get score from:
            - "initial": first phase
            - "final": last phase
            - "best": highest score across all phases
        fallback: If score_type is NaN, try this instead. If None, raise ValueError.

    Returns:
        Score value

    Raises:
        ValueError: If score is NaN and no fallback provided (or fallback also NaN)
    """
    col = _get_score_column(score_type)

    if col not in analysis_df.columns:
        if fallback is not None:
            return get_session_score(analysis_df, fallback, phase, fallback=None)
        raise ValueError(f"Score column {col} not found and no fallback provided")

    if phase == "initial":
        score = analysis_df[col].iloc[0]
    elif phase == "final":
        score = analysis_df[col].iloc[-1]
    elif phase == "best":
        score = analysis_df[col].max()
    else:
        raise ValueError(f"Invalid phase: {phase}")

    if pd.isna(score):
        if fallback is not None:
            return get_session_score(analysis_df, fallback, phase, fallback=None)
        raise ValueError(
            f"Score is NaN for {score_type}/{phase} and no fallback provided"
        )

    return float(score)


def get_session_improvement(
    analysis_df: pd.DataFrame,
    score_type: DatasetSplitT,
    fallback: DatasetSplitT | None = None,
) -> float:
    """Get improvement (best - initial) for a session.

    Args:
        analysis_df: Analysis DataFrame for a session
        score_type: Which score column to use
        fallback: If score_type is NaN, try this instead. If None, raise ValueError.

    Returns:
        Improvement value (best_score - initial_score)

    Raises:
        ValueError: If scores are NaN and no fallback provided
    """
    initial = get_session_score(analysis_df, score_type, "initial", fallback)
    best = get_session_score(analysis_df, score_type, "best", fallback)
    return best - initial


def rank_sessions_by_improvement(
    analyses: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    score_type: DatasetSplitT,
    fallback: DatasetSplitT | None = None,
    task: str | None = None,
) -> pd.DataFrame:
    """Rank sessions by improvement.

    Args:
        analyses: Dict of session_id -> analysis DataFrame
        metadata: Metadata DataFrame
        score_type: Which score column to use
        fallback: If score_type is NaN, try this instead. If None, skip session.
        task: Filter to specific task (optional)

    Returns:
        DataFrame with columns [session_id, task, scaffold, initial_score,
        best_score, improvement] sorted by improvement descending
    """
    if task is not None:
        _, metadata = filter_data(analyses, metadata, scaffolds=None, tasks={task})

    records = []
    for _, row in metadata.iterrows():
        session_id = row["session_id"]
        if session_id not in analyses:
            continue

        analysis_df = analyses[session_id]
        try:
            initial = get_session_score(analysis_df, score_type, "initial", fallback)
            best = get_session_score(analysis_df, score_type, "best", fallback)
            improvement = best - initial
        except ValueError:
            # Skip sessions where we can't compute scores
            continue

        records.append(
            {
                "session_id": session_id,
                "task": row.get("task"),
                "scaffold": row.get("optimizer_scaffold"),
                "model": row.get("model"),
                "initial_score": initial,
                "best_score": best,
                "improvement": improvement,
            }
        )

    result = pd.DataFrame(records)
    if len(result) > 0:
        result = result.sort_values("improvement", ascending=False).reset_index(
            drop=True
        )

    return result


def compute_optimal_discovery(
    analyses: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    score_type: DatasetSplitT,
    fallback: DatasetSplitT | None = None,
) -> pd.DataFrame:
    """Compute normalized phase where optimal score is first achieved.

    Optimal = first phase achieving max score (excluding initial phase 0).

    Args:
        analyses: Dict of session_id -> analysis DataFrame
        metadata: Metadata DataFrame
        score_type: Which score column to use
        fallback: If score_type is NaN, try this instead. If None, skip session.

    Returns:
        DataFrame with columns:
        - session_id
        - task
        - scaffold
        - optimal_phase: phase index where optimal found
        - total_phases: total number of phases
        - optimal_phase_normalized: optimal_phase / (total_phases - 1)
    """
    score_col = f"{score_type}_mean_score"
    fallback_col = f"{fallback}_mean_score" if fallback else None

    task_map = dict(zip(metadata["session_id"], metadata["task"]))
    scaffold_map = dict(zip(metadata["session_id"], metadata["optimizer_scaffold"]))

    records = []
    for session_id, df in analyses.items():
        task = task_map.get(session_id)
        scaffold = scaffold_map.get(session_id)

        if task is None or scaffold is None:
            continue

        # Get scores (excluding phase 0)
        non_initial = df[df["phase_index"] > 0].copy()
        if len(non_initial) == 0:
            continue

        # Try primary score column, fallback if needed
        if score_col in non_initial.columns:
            scores = non_initial[score_col]
        elif fallback_col and fallback_col in non_initial.columns:
            scores = non_initial[fallback_col]
        else:
            continue

        # Skip if all NaN
        if scores.isna().all():
            if fallback_col and fallback_col in non_initial.columns:
                scores = non_initial[fallback_col]
                if scores.isna().all():
                    continue
            else:
                continue

        # Find first phase achieving max score
        max_score = scores.max()
        optimal_idx = scores.eq(max_score).idxmax()
        optimal_phase = int(non_initial.loc[optimal_idx, "phase_index"])
        total_phases = int(df["phase_index"].max()) + 1

        # Normalize: 0 = found at phase 1, 1 = found at last phase
        if total_phases > 1:
            optimal_normalized = (
                (optimal_phase - 1) / (total_phases - 2) if total_phases > 2 else 0.0
            )
        else:
            optimal_normalized = 0.0

        records.append(
            {
                "session_id": session_id,
                "task": task,
                "scaffold": scaffold,
                "optimal_phase": optimal_phase,
                "total_phases": total_phases,
                "optimal_phase_normalized": optimal_normalized,
            }
        )

    return pd.DataFrame(records)
