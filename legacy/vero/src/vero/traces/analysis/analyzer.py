"""LLM-based trace analysis for optimization sessions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from vero.traces.analysis.collator import TraceAnalysisPayload
from vero.workspace.git import GitWorkspace

# =============================================================================
# Change Tag Model - Flat structure compatible with OpenAI structured output
# =============================================================================

PrimaryType = Literal["prompt", "tool", "workflow", "config", "dependency", "other"]
Action = Literal["added", "modified", "deleted", "refactored"]
SubType = Literal[
    # Prompt sub-types
    "system_prompt",
    "user_prompt",
    "few_shot",
    "formatting",
    "context",
    "persona",
    "constraints",
    # Tool sub-types
    "search",
    "browser",
    "database",
    "file_ops",
    "memory",
    "code_execution",
    "math",
    "api",
    "subagent",
    "human_input",
    "messaging",
    "parsing",
    "validation",
    "transformation",
    "error_handling",
    "logging",
    # Workflow sub-types
    "orchestration",
    "control_flow",
    "retry",
    "parallelization",
    "sampling",
    "planning",
    "reflection",
    "verification",
    "multi_agent",
    # Config sub-types
    "model_settings",
    "function_parameters",
    "thresholds",
    "environment",
    "timeouts",
    "resource_limits",
    # Dependency sub-types
    "package",
    "import",
    "version",
    "external_service",
    # Generic fallback
    "other",
]


class ChangeTag(BaseModel):
    """Structured tag describing a code change.

    Primary types and their typical sub-types:
    - prompt: system_prompt, user_prompt, few_shot, instructions, formatting, context, persona, constraints
    - tool: search, browser, database, file_ops, memory, code_execution, math, api, subagent, human_input, messaging, parsing, validation, transformation, error_handling, logging
    - workflow: orchestration, control_flow, retry, parallelization, state_management, termination, planning, reflection
    - config: model_settings, parameters, thresholds, environment, timeouts, resource_limits
    - dependency: package, import, version, external_service
    - other: other (or any descriptive sub_type)
    """

    primary_type: PrimaryType = Field(
        description="Primary category: prompt, tool, workflow, config, dependency, or other"
    )
    action: Action = Field(
        description="Action taken: added, modified, deleted, or refactored"
    )
    sub_type: Optional[SubType] = Field(
        default=None,
        description="Specific sub-category within the primary type (see docstring for guidance)",
    )
    descriptor: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Brief description of the change, e.g., 'added retry on API timeout'",
    )


# =============================================================================
# Phase Analysis Model
# =============================================================================


class PhaseAnalysis(BaseModel):
    """Structured analysis of an optimization phase."""

    description: str = Field(
        description="A detailed description of the changes made in this phase"
    )
    short_summary: str = Field(description="A very brief summary, MAX 5 WORDS")
    tags: list[ChangeTag] = Field(description="Categorized changes made in this phase")


# =============================================================================
# Default Prompt Template
# =============================================================================

DEFAULT_PHASE_ANALYSIS_PROMPT = """You are analyzing a single phase of an LLM coding agent that is tasked with optimizing another LLM agent to perform a specific task.

The agent makes code changes (shown as git diffs) and then evaluates the results through experiments.

## Commit History

{commit_info}

## Agent's Trace

{trace_items}

## Experiment Results

{experiment_info}

---

Analyze this data and provide:

1. **short_summary**: MAXIMUM 5 WORDS - be extremely concise (e.g., "Simplified GPQA prompt", "Added retry logic")

2. **tags**: A list of structured change tags. For each distinct change, provide:
   - **primary_type**: One of: prompt, tool, workflow, config, dependency, other
   - **action**: One of: added, modified, deleted, refactored
   - **sub_type**: Specific category based on primary_type:
     - prompt: system_prompt, user_prompt, few_shot, instructions, formatting, context, persona, constraints, other
     - tool: search, browser, database, file_ops, memory, code_execution, math, api, subagent, human_input, messaging, parsing, validation, transformation, error_handling, logging, other
     - workflow: orchestration, control_flow, retry, parallelization, state_management, termination, planning, reflection, other
     - config: model_settings, parameters, thresholds, environment, timeouts, resource_limits, other
     - dependency: package, import, version, external_service, other
   - **descriptor**: Optional brief description of this specific change

3. **description**: A detailed description of what the agent did in this phase, including its intent and actions
"""


# =============================================================================
# Formatting Helpers
# =============================================================================


def format_commit_info(phase_info: dict[str, Any]) -> str:
    """Format commit diffs for the prompt (raw diffs only)."""
    diffs = []
    for cd in phase_info.get("commit_diffs", []):
        diff = cd.get("diff", "")
        if diff:
            diffs.append(f"{diff}")
    return "\n\n".join(diffs) if diffs else "No commits in this phase"


def format_trace_items(phase_info: dict[str, Any], max_items: int = 50) -> str:
    """Format trace items for the prompt as JSON."""
    trace_items = phase_info.get("trace_items", [])[:max_items]
    if not trace_items:
        return "No trace items"

    return json.dumps(trace_items, indent=2)


def format_experiment_info(phase_info: dict[str, Any]) -> str:
    """Format experiment scores for the prompt."""
    experiment_info_parts = []
    for score in phase_info.get("experiment_scores", []):
        experiment_info_parts.append(
            f"- Commit {score['commit'][:8]}: "
            f"split={score.get('split')}, "
            f"mean_score={score.get('mean_score')}, "
            f"error_rate={score.get('error_rate')}, "
            f"num_samples={score.get('num_samples')}"
        )
    return (
        "\n".join(experiment_info_parts)
        if experiment_info_parts
        else "No experiments in this phase"
    )


# =============================================================================
# DataFrame Conversion
# =============================================================================


def _extract_experiment_by_split(experiment_scores: list[dict]) -> dict[str, Any]:
    """Extract experiment metrics organized by split (train/test/validation)."""
    result = {}
    for split in ["train", "test", "validation"]:
        exp = next((e for e in experiment_scores if e.get("split") == split), None)
        if exp:
            result[f"{split}_mean_score"] = exp.get("mean_score")
            result[f"{split}_error_rate"] = exp.get("error_rate")
            result[f"{split}_num_samples"] = exp.get("num_samples")
            result[f"{split}_dataset"] = exp.get("dataset")
        else:
            result[f"{split}_mean_score"] = None
            result[f"{split}_error_rate"] = None
            result[f"{split}_num_samples"] = None
            result[f"{split}_dataset"] = None
    return result


def _serialize_tags(tags: list[Any]) -> str:
    """Serialize ChangeTag objects to JSON string for CSV storage."""
    serialized = []
    for tag in tags:
        if isinstance(tag, BaseModel):
            serialized.append(tag.model_dump())
        else:
            serialized.append(tag)
    return json.dumps(serialized)


def _result_to_row(result: dict[str, Any]) -> dict[str, Any]:
    """Convert an analysis result to a flat row for DataFrame.

    Dynamically extracts all fields from the structured output model.
    Tags are serialized to JSON string for CSV storage.
    """
    analysis = result["analysis"]

    # Start with metadata
    row = {
        "phase_index": result["phase_index"],
        "final_commit": result["final_commit"],
        "num_commits": result["num_commits"],
        "num_trace_items": result["num_trace_items"],
    }

    # Dynamically add all fields from the analysis model
    if isinstance(analysis, BaseModel):
        for field_name in analysis.model_fields:
            value = getattr(analysis, field_name)
            # Serialize tags to JSON string for CSV compatibility
            if field_name == "tags" and isinstance(value, list):
                value = _serialize_tags(value)
            row[field_name] = value
    elif isinstance(analysis, dict):
        row.update(analysis)

    # Add experiment metrics by split
    row.update(_extract_experiment_by_split(result["experiment_scores"]))

    return row


# =============================================================================
# Trace Analyzer
# =============================================================================


class TraceAnalyzer:
    """LLM-based analyzer for optimization session traces.

    Args:
        model: OpenAI model to use for analysis
        output_model: Pydantic model for structured output (default: PhaseAnalysis)
        prompt_template: Custom prompt template with {commit_info}, {trace_items}, {experiment_info} placeholders
        client: Optional AsyncOpenAI client (creates new one if not provided)
        max_trace_items: Maximum number of trace items to include in prompt
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        output_model: type[BaseModel] = PhaseAnalysis,
        prompt_template: str = DEFAULT_PHASE_ANALYSIS_PROMPT,
        client: AsyncOpenAI | None = None,
        max_trace_items: int = 50,
    ):
        self.model = model
        self.output_model = output_model
        self.prompt_template = prompt_template
        self.client = client or AsyncOpenAI()
        self.max_trace_items = max_trace_items

    async def analyze_phase(
        self, phase_info: dict[str, Any], drop_items_from_info: bool = True
    ) -> dict[str, Any]:
        """Analyze a single phase using the LLM.

        Args:
            phase_info: Phase info dict from TraceAnalysisPayload.get_phase_info()

        Returns:
            Dict with analysis results and metadata
        """
        # Extract metadata before potentially dropping
        phase_index = phase_info.get("phase_index")
        final_commit = phase_info.get("final_commit", "")[:8]
        experiment_scores = phase_info.get("experiment_scores", [])

        if drop_items_from_info:
            phase_info = phase_info.copy()
            phase_info.pop("tool_calls", None)
            phase_info.pop("experiment_scores", None)

        phase = phase_info.get("phase", {})
        if phase.get("is_initial", False):
            return {
                "phase_index": phase_index,
                "final_commit": final_commit,
                "analysis": None,
                "experiment_scores": experiment_scores,
                "num_commits": len(phase_info.get("commit_diffs", [])),
                "num_trace_items": len(phase_info.get("trace_items", [])),
            }

        prompt = self.prompt_template.format(
            commit_info=format_commit_info(phase_info),
            trace_items=format_trace_items(phase_info, self.max_trace_items),
            experiment_info=format_experiment_info(phase_info),
        )

        response = await self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            text_format=self.output_model,
            temperature=0.0,
        )

        return {
            "phase_index": phase_index,
            "final_commit": final_commit,
            "analysis": response.output_parsed,
            "experiment_scores": experiment_scores,
            "num_commits": len(phase_info.get("commit_diffs", [])),
            "num_trace_items": len(phase_info.get("trace_items", [])),
        }

    async def analyze_session(
        self,
        session_id: str,
        project_path: Path | str,
        max_concurrency: int = 5,
        show_progress: bool = True,
        return_payload: bool = False,
        use_cache: bool = False,
        save_to_cache: bool = False,
        drop_items_from_info: bool = True,
    ) -> pd.DataFrame | tuple[TraceAnalysisPayload, pd.DataFrame]:
        """Analyze all phases in a session with concurrent LLM calls.

        Args:
            session_id: Session UUID
            project_path: Path to the project/repo
            max_concurrency: Maximum concurrent LLM calls
            show_progress: Whether to show a progress bar
            return_payload: Whether to return the payload used in the analysis
            use_cache: If True, load from cache if analysis_df.csv exists in session dir
            save_to_cache: If True, save results to analysis_df.csv in session dir
            drop_items_from_info: If True, drop trace items from the info dict

        Returns:
            DataFrame with one row per phase, or (payload, DataFrame) if return_payload=True
        """
        project_path = Path(project_path)
        from vero.core.sessions import get_vero_home_dir
        session_dir = get_vero_home_dir() / "sessions" / session_id
        cache_df_path = session_dir / "analysis_df.csv"

        # Try loading from cache
        if use_cache and cache_df_path.exists():
            print(f"Loading analysis from cache: {cache_df_path}")
            df = pd.read_csv(cache_df_path)
            if return_payload:
                payload = await TraceAnalysisPayload.from_session_id(
                    session_id, project_path=project_path
                )
                return payload, df
            return df

        # Load payload and workspace
        workspace = await GitWorkspace.create(project_path)
        payload = await TraceAnalysisPayload.from_session_id(
            session_id, project_path=project_path
        )

        # Delegate to analyze_payload
        df = await self.analyze_payload(
            payload,
            workspace,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
            drop_items_from_info=drop_items_from_info,
        )

        # Save to cache if requested
        if save_to_cache:
            df.to_csv(cache_df_path, index=False)

        if return_payload:
            return payload, df

        return df

    async def analyze_payload(
        self,
        payload: TraceAnalysisPayload,
        workspace: GitWorkspace,
        max_concurrency: int = 5,
        show_progress: bool = True,
        drop_items_from_info: bool = True,
    ) -> pd.DataFrame:
        """Analyze all phases in an existing payload.

        Args:
            payload: TraceAnalysisPayload to analyze
            workspace: GitWorkspace for the project
            max_concurrency: Maximum concurrent LLM calls
            show_progress: Whether to show a progress bar
            drop_items_from_info: If True, drop trace items from the info dict
        Returns:
            DataFrame with one row per phase
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def analyze_with_semaphore(phase_index: int) -> dict[str, Any] | None:
            phase_info = await payload.get_phase_info(phase_index, workspace)
            if not phase_info:
                return None
            async with semaphore:
                return await self.analyze_phase(
                    phase_info, drop_items_from_info=drop_items_from_info
                )

        tasks = [analyze_with_semaphore(i) for i in range(len(payload.phases))]

        if show_progress:
            results = await tqdm.gather(
                *tasks,
                desc="Analyzing phases",
                total=len(tasks),
            )
        else:
            results = await asyncio.gather(*tasks)

        rows = [_result_to_row(r) for r in results if r is not None]
        return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================


def plot_session_scores(
    df: pd.DataFrame,
    title: str = "Optimization Session Progress",
    figsize: tuple[int, int] = (14, 8),
    show_annotations: bool = True,
    annotation_fontsize: int = 8,
    show_best_so_far: bool = True,
    max_annotation_chars: int = 30,
) -> Any:
    """Plot optimization progress with phase annotations.

    Args:
        df: DataFrame from TraceAnalyzer.analyze_session()
        title: Plot title
        figsize: Figure size (width, height)
        show_annotations: Whether to show short_summary annotations
        annotation_fontsize: Font size for annotations
        show_best_so_far: Whether to show "best so far" line (uses validation, falls back to train)
        max_annotation_chars: Maximum characters for annotation text before truncation

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)

    x = df["phase_index"].values

    # Plot lines for each split (train=green, validation=yellow, test=red)
    splits = [
        ("train_mean_score", "Train", "tab:green", "-"),
        ("validation_mean_score", "Validation", "gold", "--"),
        ("test_mean_score", "Test", "tab:red", "-."),
    ]

    for col, label, color, linestyle in splits:
        if col in df.columns:
            y = df[col].values
            mask = ~pd.isna(y)
            if mask.any():
                ax.plot(
                    x[mask],
                    y[mask],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markersize=6,
                    linewidth=2,
                )

    # Plot best-so-far line (subtle dotted black)
    if show_best_so_far:
        # Build score series: use validation if available, else train
        scores = []
        for _, row in df.iterrows():
            val_score = row.get("validation_mean_score")
            train_score = row.get("train_mean_score")
            if pd.notna(val_score):
                scores.append(val_score)
            elif pd.notna(train_score):
                scores.append(train_score)
            else:
                scores.append(np.nan)

        scores = np.array(scores)
        best_so_far = np.zeros_like(scores, dtype=float)
        current_best = -np.inf
        for i, val in enumerate(scores):
            if not pd.isna(val):
                current_best = max(current_best, val)
            best_so_far[i] = current_best if current_best > -np.inf else np.nan

        mask = ~np.isnan(best_so_far)
        if mask.any():
            ax.step(
                x[mask],
                best_so_far[mask],
                label="Best So Far",
                color="black",
                linewidth=1,
                linestyle=":",
                alpha=0.6,
                where="post",
            )

    # Add annotations between indices
    if show_annotations and "short_summary" in df.columns:
        # Get y-axis limits for positioning
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        for idx, row in df.iterrows():
            # Skip if short_summary is missing or empty
            summary = row.get("short_summary")
            if pd.isna(summary) or (isinstance(summary, str) and not summary.strip()):
                continue

            # Truncate if too long
            if len(summary) > max_annotation_chars:
                summary = summary[: max_annotation_chars - 3] + "..."

            # Position annotation between previous index and this one (centered at -0.5)
            x_pos = row["phase_index"] - 0.5

            # Alternate y positions (top/bottom of plot area)
            if idx % 2 == 0:
                y_pos = y_min + y_range * 0.02
                va = "bottom"
            else:
                y_pos = y_max - y_range * 0.02
                va = "top"

            ax.text(
                x_pos,
                y_pos,
                summary,
                fontsize=annotation_fontsize,
                ha="center",
                va=va,
                rotation=90,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="lightgray",
                ),
            )

    ax.set_xlabel("Commit Index", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    plt.tight_layout()
    return fig


def plot_session_scores_with_table(
    df: pd.DataFrame,
    title: str = "Optimization Session Progress",
    figsize: tuple[int, int] = (14, 8),
    show_best_so_far: bool = True,
    annotation_fontsize: int = 8,
    wrap_width: int = 15,
) -> Any:
    """Plot optimization progress with text box annotations on the plot.

    Annotations are placed as horizontal text boxes between commit indices,
    with text wrapping to fit.

    Args:
        df: DataFrame from TraceAnalyzer.analyze_session()
        title: Plot title
        figsize: Figure size (width, height)
        show_best_so_far: Whether to show "best so far" line
        annotation_fontsize: Font size for annotation text
        wrap_width: Number of characters before wrapping to new line

    Returns:
        matplotlib Figure object
    """
    import textwrap

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    x = df["phase_index"].values

    # Plot lines (train=blue, validation=gold, test=purple)
    splits = [
        ("train_mean_score", "Train", "#1f77b4", "-"),  # blue
        ("validation_mean_score", "Validation", "#f0b800", "--"),  # gold
        ("test_mean_score", "Test", "#9467bd", "-."),  # purple
    ]

    for col, label, color, linestyle in splits:
        if col in df.columns:
            y = df[col].values
            mask = ~pd.isna(y)
            if mask.any():
                ax.plot(
                    x[mask],
                    y[mask],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markersize=8,
                    linewidth=2,
                    zorder=3,
                )

    # Best-so-far line (more visible)
    if show_best_so_far:
        scores = []
        for _, row in df.iterrows():
            val_score = row.get("validation_mean_score")
            train_score = row.get("train_mean_score")
            if pd.notna(val_score):
                scores.append(val_score)
            elif pd.notna(train_score):
                scores.append(train_score)
            else:
                scores.append(np.nan)

        scores = np.array(scores)
        best_so_far = np.zeros_like(scores, dtype=float)
        current_best = -np.inf
        for i, val in enumerate(scores):
            if not pd.isna(val):
                current_best = max(current_best, val)
            best_so_far[i] = current_best if current_best > -np.inf else np.nan

        mask = ~np.isnan(best_so_far)
        if mask.any():
            ax.step(
                x[mask],
                best_so_far[mask],
                label="Best So Far",
                color="#333333",
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
                where="post",
                zorder=2,
            )

    ax.set_xlabel("Phase Index", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_xticks(x)

    # Add text box annotations between indices
    if "short_summary" in df.columns:
        import matplotlib.colors as mcolors

        y_min, y_max = ax.get_ylim()

        # Compute score deltas for coloring (prefer validation, fallback to train)
        val_scores = (
            df["validation_mean_score"].values
            if "validation_mean_score" in df.columns
            else None
        )
        train_scores = (
            df["train_mean_score"].values if "train_mean_score" in df.columns else None
        )

        # Use validation if it has any non-NaN values, otherwise use train
        if val_scores is not None and not pd.isna(val_scores).all():
            scores_for_delta = val_scores
        elif train_scores is not None:
            scores_for_delta = train_scores
        else:
            scores_for_delta = None

        deltas = []
        if scores_for_delta is not None:
            for i in range(len(scores_for_delta)):
                if (
                    i == 0
                    or pd.isna(scores_for_delta[i])
                    or pd.isna(scores_for_delta[i - 1])
                ):
                    deltas.append(0.0)
                else:
                    deltas.append(scores_for_delta[i] - scores_for_delta[i - 1])
        else:
            deltas = [0.0] * len(df)

        # Find max absolute delta for normalization
        max_abs_delta = max(abs(d) for d in deltas) if deltas else 1.0
        if max_abs_delta == 0:
            max_abs_delta = 1.0

        def delta_to_color(delta: float) -> str:
            """Map delta to red-green spectrum. Green=improvement, Red=regression."""
            # Normalize to [-1, 1]
            norm = delta / max_abs_delta
            # Clamp
            norm = max(-1.0, min(1.0, norm))

            if norm >= 0:
                # Green spectrum: white to green
                intensity = norm
                r = 1.0 - intensity * 0.6
                g = 1.0 - intensity * 0.1
                b = 1.0 - intensity * 0.6
            else:
                # Red spectrum: white to red
                intensity = -norm
                r = 1.0 - intensity * 0.1
                g = 1.0 - intensity * 0.6
                b = 1.0 - intensity * 0.6

            return mcolors.to_hex([r, g, b])

        # First pass: collect all annotations and find max dimensions
        annotations = []
        max_lines = 0

        for idx, row in df.iterrows():
            summary = row.get("short_summary")
            if pd.isna(summary) or (isinstance(summary, str) and not summary.strip()):
                continue

            wrapped_lines = textwrap.wrap(str(summary), width=wrap_width)
            wrapped = "\n".join(wrapped_lines)
            x_pos = row["phase_index"] - 0.5
            phase_idx = int(row["phase_index"])
            color = delta_to_color(deltas[phase_idx])

            annotations.append((x_pos, wrapped, len(wrapped_lines), color))
            max_lines = max(max_lines, len(wrapped_lines))

        # Pad all annotations to have the same number of lines
        padded_annotations = []
        for x_pos, wrapped, num_lines, color in annotations:
            lines_to_add = max_lines - num_lines
            top_pad = lines_to_add // 2
            bottom_pad = lines_to_add - top_pad
            padded = "\n" * top_pad + wrapped + "\n" * bottom_pad
            padded_annotations.append((x_pos, padded, color))

        # Fixed y position for all annotations (bottom of plot)
        y_range = y_max - y_min
        y_pos = y_min + y_range * 0.12

        # Second pass: draw all annotations with uniform sizing and delta-based colors
        for x_pos, padded_text, box_color in padded_annotations:
            ax.text(
                x_pos,
                y_pos,
                padded_text,
                fontsize=annotation_fontsize,
                ha="center",
                va="center",
                family="sans-serif",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=box_color,
                    edgecolor="gray",
                    alpha=0.95,
                ),
                zorder=4,
            )

    plt.tight_layout()
    return fig
