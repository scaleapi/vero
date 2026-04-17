"""Tag parsing and analysis for interpret module."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
from pydantic import TypeAdapter
from vero.traces.analysis.analyzer import ChangeTag, PrimaryType

# Tag categories derived from PrimaryType Literal
TAG_CATEGORIES = {pt: pt.title() for pt in get_args(PrimaryType)}

# TypeAdapter for parsing list of tags
_TagsAdapter = TypeAdapter(list[ChangeTag])


def parse_tags_from_row(tags_value: Any) -> list[ChangeTag]:
    """Parse tags column from DataFrame row.

    Expects tags to be a JSON string of serialized ChangeTag objects.

    Args:
        tags_value: Value from the 'tags' column (JSON string)

    Returns:
        List of ChangeTag instances
    """
    if tags_value is None or (isinstance(tags_value, float) and pd.isna(tags_value)):
        return []

    if isinstance(tags_value, str):
        tags_value = json.loads(tags_value)

    return _TagsAdapter.validate_python(tags_value)


def _compute_entropy(counts: dict[str, int]) -> float:
    """Compute Shannon entropy from counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def compute_tag_distribution(
    analyses: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    group_by: Literal["scaffold", "task"],
    phases: list[int] | None = None,
) -> pd.DataFrame:
    """Compute tag probability distribution by phase.

    Args:
        analyses: Dict of session_id -> analysis DataFrame
        metadata: Metadata DataFrame
        group_by: Group results by "scaffold" or "task"
        phases: List of phase indices to include. None = auto-detect (1-7).

    Returns:
        DataFrame with columns:
        - group: scaffold name or task name
        - phase: phase index
        - primary_type: tag category
        - count: raw count
        - probability: normalized probability within phase
        - entropy: Shannon entropy of distribution for that phase
    """
    if phases is None:
        phases = list(range(1, 8))  # Default to phases 1-7

    # Build group mapping
    if group_by == "scaffold":
        group_col = "optimizer_scaffold"
    else:
        group_col = "task"

    group_map = dict(zip(metadata["session_id"], metadata[group_col]))

    # Collect tag counts: group -> phase -> primary_type -> count
    group_phase_counts: dict[str, dict[int, Counter]] = {}

    for session_id, df in analyses.items():
        group = group_map.get(session_id)
        if group is None:
            continue

        if group not in group_phase_counts:
            group_phase_counts[group] = {p: Counter() for p in phases}

        for _, row in df.iterrows():
            phase = row.get("phase_index")
            if phase not in phases:
                continue

            tags = parse_tags_from_row(row.get("tags"))
            for tag in tags:
                group_phase_counts[group][phase][tag.primary_type] += 1

    # Convert to DataFrame
    records = []
    for group, phase_counts in group_phase_counts.items():
        for phase, counts in phase_counts.items():
            total = sum(counts.values())
            entropy = _compute_entropy(counts)

            for primary_type in TAG_CATEGORIES:
                count = counts.get(primary_type, 0)
                prob = count / total if total > 0 else 0.0

                records.append(
                    {
                        "group": group,
                        "phase": phase,
                        "primary_type": primary_type,
                        "count": count,
                        "probability": prob,
                        "entropy": entropy,
                    }
                )

    return pd.DataFrame(records)


def compute_subtype_distribution(
    analyses: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    group_by: Literal["scaffold", "task"] = "scaffold",
) -> pd.DataFrame:
    """Compute subtype counts by primary_type and group.

    Args:
        analyses: Dict of session_id -> analysis DataFrame
        metadata: Metadata DataFrame
        group_by: Group results by "scaffold" or "task"

    Returns:
        DataFrame with columns:
        - group: scaffold name or task name
        - primary_type: tag category
        - sub_type: specific subtype
        - count: raw count
    """
    # Build group mapping
    if group_by == "scaffold":
        group_col = "optimizer_scaffold"
    else:
        group_col = "task"

    group_map = dict(zip(metadata["session_id"], metadata[group_col]))

    # Collect counts: group -> primary_type -> sub_type -> count
    counts: dict[str, dict[str, Counter]] = {}

    for session_id, df in analyses.items():
        group = group_map.get(session_id)
        if group is None:
            continue

        if group not in counts:
            counts[group] = {pt: Counter() for pt in TAG_CATEGORIES}

        for _, row in df.iterrows():
            tags = parse_tags_from_row(row.get("tags"))
            for tag in tags:
                sub_type = tag.sub_type or "unspecified"
                counts[group][tag.primary_type][sub_type] += 1

    # Convert to DataFrame
    records = []
    for group, type_counts in counts.items():
        for primary_type, subtype_counts in type_counts.items():
            for sub_type, count in subtype_counts.items():
                records.append(
                    {
                        "group": group,
                        "primary_type": primary_type,
                        "sub_type": sub_type,
                        "count": count,
                    }
                )

    return pd.DataFrame(records)
