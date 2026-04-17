"""Plotting functions for analysis module."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, SCAFFOLD_ALIASES, TASK_ALIASES
from .tags import TAG_CATEGORIES

# Color palette for tag categories
TAG_COLORS = {
    "prompt": "#1f77b4",  # blue
    "tool": "#ff7f0e",  # orange
    "workflow": "#2ca02c",  # green
    "config": "#d62728",  # red
    "dependency": "#9467bd",  # purple
    "other": "#7f7f7f",  # gray
}

# Hatching patterns for scaffolds
SCAFFOLD_HATCHES = {
    "vero-cookbook": "",  # solid
    "vero-orchestrator-cookbook": "//",
    "vero-prompts-only": "xx",
}


def plot_tag_probability_by_phase(
    tag_dist: pd.DataFrame,
    group_by: Literal["scaffold", "task"],
    figsize: tuple[int, int] = (14, 8),
    show_entropy: bool = True,
) -> plt.Figure:
    """Plot tag probability distribution by phase.

    Creates a single plot with grouped bars for each phase.
    Each group (scaffold/task) is differentiated by hatching pattern.
    Bars are stacked by tag category.

    Args:
        tag_dist: DataFrame from compute_tag_distribution()
        group_by: How data is grouped ("scaffold" or "task")
        figsize: Figure size
        show_entropy: Whether to show entropy values at top of bars

    Returns:
        Matplotlib Figure
    """
    groups = sorted(tag_dist["group"].unique())
    phases = sorted(tag_dist["phase"].unique())
    n_groups = len(groups)
    n_phases = len(phases)

    # Get alias and hatch mappings
    aliases = SCAFFOLD_ALIASES if group_by == "scaffold" else TASK_ALIASES

    # Generate hatching patterns for groups
    hatch_patterns = ["", "//", "xx", "\\\\", "..", "oo", "**"]
    if group_by == "scaffold":
        group_hatches = {
            g: SCAFFOLD_HATCHES.get(g, hatch_patterns[i % len(hatch_patterns)])
            for i, g in enumerate(groups)
        }
    else:
        group_hatches = {g: hatch_patterns[i % len(hatch_patterns)] for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=figsize)

    # Bar width and positions
    bar_width = 0.8 / n_groups
    phase_positions = np.arange(n_phases)

    # Track bars for legend
    tag_handles = {}
    group_handles = {}

    for g_idx, group in enumerate(groups):
        group_data = tag_dist[tag_dist["group"] == group]
        x_offset = (g_idx - (n_groups - 1) / 2) * bar_width

        # Build stacked bars for this group
        bottom = np.zeros(n_phases)

        for primary_type in TAG_CATEGORIES:
            heights = []
            for phase in phases:
                phase_data = group_data[
                    (group_data["phase"] == phase) & (group_data["primary_type"] == primary_type)
                ]
                prob = phase_data["probability"].values[0] if len(phase_data) > 0 else 0
                heights.append(prob)

            bars = ax.bar(
                phase_positions + x_offset,
                heights,
                bar_width,
                bottom=bottom,
                color=TAG_COLORS[primary_type],
                hatch=group_hatches[group],
                edgecolor="black",
                linewidth=0.5,
            )
            bottom += np.array(heights)

            # Track first bar of each tag type for legend
            if primary_type not in tag_handles:
                tag_handles[primary_type] = bars[0]

        # Create a dummy bar for group legend (with hatch)
        group_handles[group] = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="black",
            hatch=group_hatches[group],
            linewidth=1,
        )

        # Add entropy annotations
        if show_entropy:
            for p_idx, phase in enumerate(phases):
                phase_data = group_data[group_data["phase"] == phase]
                if len(phase_data) > 0:
                    entropy = phase_data["entropy"].iloc[0]
                    # Avoid displaying -0.0
                    if abs(entropy) < 0.05:
                        entropy = 0.0
                    ax.text(
                        phase_positions[p_idx] + x_offset,
                        bottom[p_idx] + 0.02,
                        f"{entropy:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                        color="black",
                    )

    # Styling
    ax.set_xlabel("Optimization Phase", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_xticks(phase_positions)
    ax.set_xticklabels(phases)
    ax.set_ylim(0, 1.3 if show_entropy else 1.05)

    # Create legends
    # Change type legend (colors) - multi-column, inside plot area
    tag_legend_handles = [tag_handles[pt] for pt in TAG_CATEGORIES if pt in tag_handles]
    tag_legend_labels = [TAG_CATEGORIES[pt] for pt in TAG_CATEGORIES if pt in tag_handles]
    legend1 = ax.legend(
        tag_legend_handles,
        tag_legend_labels,
        loc="upper left",
        title="Change Type",
        fontsize=9,
        ncol=2,
        frameon=True,
    )
    ax.add_artist(legend1)

    # Group legend (hatching)
    group_legend_handles = [group_handles[g] for g in groups]
    group_legend_labels = [aliases.get(g, g) for g in groups]
    ax.legend(
        group_legend_handles,
        group_legend_labels,
        loc="upper right",
        title=group_by.title(),
        fontsize=9,
    )

    fig.tight_layout()

    return fig


def plot_entropy_by_phase(
    tag_dist: pd.DataFrame,
    group_by: Literal["scaffold", "task"],
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot entropy vs phase as a line plot.

    Args:
        tag_dist: DataFrame from compute_tag_distribution()
        group_by: How data is grouped ("scaffold" or "task")
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    groups = sorted(tag_dist["group"].unique())
    phases = sorted(tag_dist["phase"].unique())

    # Get alias mapping
    aliases = SCAFFOLD_ALIASES if group_by == "scaffold" else TASK_ALIASES

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette for lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    for g_idx, group in enumerate(groups):
        group_data = tag_dist[tag_dist["group"] == group]

        # Get entropy for each phase (take first row per phase since entropy is same for all tag types)
        entropies = []
        for phase in phases:
            phase_data = group_data[group_data["phase"] == phase]
            if len(phase_data) > 0:
                entropies.append(phase_data["entropy"].iloc[0])
            else:
                entropies.append(np.nan)

        display_name = aliases.get(group, group)
        ax.plot(phases, entropies, marker="o", label=display_name, color=colors[g_idx], linewidth=2)

    ax.set_xlabel("Optimization Phase", fontsize=11)
    ax.set_ylabel("Entropy", fontsize=11)
    ax.set_xticks(phases)
    ax.legend(title=group_by.title(), fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_subtype_distribution(
    subtype_dist: pd.DataFrame,
    group_by: Literal["scaffold", "task"] = "scaffold",
    figsize: tuple[int, int] = (12, 7),
    top_n_subtypes: int = 3,
    primary_types_to_show: list[str] | None = None,
) -> plt.Figure:
    """Plot subtype counts by primary_type with grouped bars.

    X-axis: primary_type
    Bars: one per group (scaffold/task), stacked by subtype
    Y-axis: raw counts

    Args:
        subtype_dist: DataFrame from compute_subtype_distribution()
        group_by: How data is grouped ("scaffold" or "task")
        figsize: Figure size
        top_n_subtypes: Max subtypes to show per primary_type (others grouped as "misc")
        primary_types_to_show: Which primary types to include. None = ["prompt", "tool", "workflow"]

    Returns:
        Matplotlib Figure
    """
    if primary_types_to_show is None:
        primary_types_to_show = ["prompt", "tool", "workflow"]

    groups = sorted(subtype_dist["group"].unique())
    n_groups = len(groups)

    # Get alias and hatch mappings
    aliases = SCAFFOLD_ALIASES if group_by == "scaffold" else TASK_ALIASES
    hatch_patterns = ["", "//", "xx", "\\\\", "..", "oo", "**"]
    if group_by == "scaffold":
        group_hatches = {
            g: SCAFFOLD_HATCHES.get(g, hatch_patterns[i % len(hatch_patterns)])
            for i, g in enumerate(groups)
        }
    else:
        group_hatches = {g: hatch_patterns[i % len(hatch_patterns)] for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=figsize)

    # Bar positioning - more space between groups
    bar_width = 0.25
    type_positions = np.arange(len(primary_types_to_show)) * 1.2  # More spacing between types

    # Get top N subtypes per primary_type (globally across groups)
    top_subtypes_per_type = {}
    for pt in primary_types_to_show:
        pt_data = subtype_dist[subtype_dist["primary_type"] == pt]
        top_subtypes = (
            pt_data.groupby("sub_type")["count"].sum().nlargest(top_n_subtypes).index.tolist()
        )
        top_subtypes_per_type[pt] = top_subtypes  # Keep as list for ordering

    # Assign unique colors per primary_type's subtypes
    # Use distinct color palettes per primary type (5 shades each)
    color_palettes = {
        "prompt": ["#08519c", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef"],  # Blues
        "tool": ["#d94701", "#fd8d3c", "#fdae6b", "#fdd0a2", "#feedde"],  # Oranges
        "workflow": ["#238b45", "#41ab5d", "#74c476", "#a1d99b", "#c7e9c0"],  # Greens
    }

    subtype_colors = {}
    subtype_to_primary = {}  # Track which primary type each subtype belongs to
    for pt in primary_types_to_show:
        palette = color_palettes.get(pt, ["#999999", "#bbbbbb", "#dddddd"])
        for i, st in enumerate(top_subtypes_per_type[pt]):
            subtype_colors[st] = palette[i % len(palette)]
            subtype_to_primary[st] = pt
    subtype_colors["misc"] = "#cccccc"  # Gray for misc

    # Track handles for legend
    subtype_handles = {}
    group_handles = {}

    for g_idx, group in enumerate(groups):
        group_data = subtype_dist[subtype_dist["group"] == group]
        x_offset = (g_idx - (n_groups - 1) / 2) * bar_width

        for pt_idx, primary_type in enumerate(primary_types_to_show):
            pt_data = group_data[group_data["primary_type"] == primary_type]

            # Get subtypes, bucket non-top as "misc"
            subtype_counts = pt_data.set_index("sub_type")["count"].to_dict()
            top_subtypes = set(top_subtypes_per_type[primary_type])

            bucketed_counts = {}
            misc_count = 0
            for st, count in subtype_counts.items():
                if st in top_subtypes:
                    bucketed_counts[st] = count
                else:
                    misc_count += count
            if misc_count > 0:
                bucketed_counts["misc"] = misc_count

            # Normalize to probabilities
            total = sum(bucketed_counts.values())
            if total == 0:
                continue

            # Stack subtypes (top ones first, misc last)
            bottom = 0
            for sub_type in sorted(
                bucketed_counts.keys(), key=lambda x: (x == "misc", -bucketed_counts.get(x, 0))
            ):
                count = bucketed_counts[sub_type]
                if count == 0:
                    continue

                prob = count / total
                color = subtype_colors.get(sub_type, "#cccccc")
                bar = ax.bar(
                    type_positions[pt_idx] + x_offset,
                    prob,
                    bar_width,
                    bottom=bottom,
                    color=color,
                    hatch=group_hatches[group],
                    edgecolor="black",
                    linewidth=0.5,
                )
                bottom += prob

                if sub_type not in subtype_handles:
                    subtype_handles[sub_type] = bar[0]

        # Group legend handle
        group_handles[group] = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="black",
            hatch=group_hatches[group],
            linewidth=1,
        )

    # Styling
    ax.set_xlabel("Change Type", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_xticks(type_positions)
    ax.set_xticklabels([TAG_CATEGORIES[pt] for pt in primary_types_to_show])
    ax.set_ylim(0, 1.05)

    # Subtype legend - columns: Prompt | Tool | Workflow
    # With ncol=3, matplotlib fills row-by-row from the list
    # So we interleave: [p1, t1, w1, p2, t2, w2, p3, t3, w3]
    # to get columns: col1=p1,p2,p3  col2=t1,t2,t3  col3=w1,w2,w3
    max_subtypes = max(len(top_subtypes_per_type[pt]) for pt in primary_types_to_show)

    legend_handles = []
    legend_labels = []
    for pt in primary_types_to_show:
        subtypes = top_subtypes_per_type[pt]
        for i in range(max_subtypes):
            if i < len(subtypes):
                st = subtypes[i]
                color = subtype_colors.get(st, "#cccccc")
                patch = plt.Rectangle(
                    (0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5
                )
                legend_handles.append(patch)
                legend_labels.append(st)
            else:
                legend_handles.append(
                    plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor="none", linewidth=0)
                )
                legend_labels.append(" ")

    legend1 = ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        title="Prompt         Tool         Workflow",
        fontsize=8,
        ncol=3,
        frameon=True,
        columnspacing=0.8,
        handlelength=1.5,
        handleheight=1.0,
    )
    ax.add_artist(legend1)

    # Group legend - below subtype legend
    group_legend_handles = [group_handles[g] for g in groups]
    group_legend_labels = [aliases.get(g, g) for g in groups]
    _ = ax.legend(
        group_legend_handles,
        group_legend_labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.65),
        title=group_by.title(),
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout()
    return fig


def plot_optimal_discovery(
    discovery_df: pd.DataFrame,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot optimal discovery phase by task.

    Shows when the optimal score is first achieved during optimization,
    normalized to [0, 1] where 0 = early and 1 = late.

    Args:
        discovery_df: DataFrame from compute_optimal_discovery()
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate by task
    task_stats = (
        discovery_df.groupby("task")["optimal_phase_normalized"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    tasks = task_stats["task"].values
    means = task_stats["mean"].values
    stds = task_stats["std"].values
    counts = task_stats["count"].values

    # Sort by mean
    sort_idx = np.argsort(means)
    tasks = tasks[sort_idx]
    means = means[sort_idx]
    stds = stds[sort_idx]
    counts = counts[sort_idx]

    # Get display names
    display_names = [TASK_ALIASES.get(t, t) for t in tasks]

    # Bar chart with error bars - different color per task
    x = np.arange(len(tasks))
    colors = [plt.cm.tab10(i % 10) for i in range(len(tasks))]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.8)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds[i] + 0.03,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_ylabel("Normalized Discovery Phase")
    ax.set_ylim(0, 1.1)
    ax.set_title("When is Optimal Score First Achieved?")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Midpoint")

    fig.tight_layout()
    return fig


def plot_umap_trajectories(
    umap_df: pd.DataFrame,
    metadata: pd.DataFrame,
    color_by: Literal["task", "improvement"] = "task",
    figsize: tuple[int, int] = (10, 10),
    show_arrows: bool = False,
) -> plt.Figure:
    """Plot UMAP trajectories of cumulative diff embeddings.

    Each trajectory shows the semantic evolution of changes from
    base commit to final commit.

    Args:
        umap_df: DataFrame from reduce_to_umap()
        metadata: Metadata DataFrame for color mapping
        color_by: Color trajectories by "task" or "improvement"
        figsize: Figure size
        show_arrows: Whether to show direction arrows

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Build session metadata mapping
    task_map = dict(zip(metadata["session_id"], metadata["task"]))

    # Get unique tasks for coloring
    tasks = list(set(task_map.values()))
    task_colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    task_color_map = dict(zip(tasks, task_colors))

    # Plot each trajectory
    for session_id in umap_df["session_id"].unique():
        session_data = umap_df[umap_df["session_id"] == session_id].sort_values("phase")

        if len(session_data) < 2:
            continue

        task = task_map.get(session_id, "unknown")
        color = task_color_map.get(task, "gray")

        x = session_data["x"].values
        y = session_data["y"].values

        # Plot line
        ax.plot(x, y, color=color, alpha=0.3, linewidth=1)

        # Start point (yellow)
        ax.scatter(x[0], y[0], color="gold", s=50, zorder=5, edgecolors="black", linewidths=0.5)

        # End point (purple)
        ax.scatter(x[-1], y[-1], color="purple", s=50, zorder=5, edgecolors="black", linewidths=0.5)

        # Intermediate points
        if len(x) > 2:
            ax.scatter(
                x[1:-1], y[1:-1], color=color, s=20, alpha=0.5, edgecolors="white", linewidths=0.3
            )

    # Legend for tasks
    for task, color in task_color_map.items():
        display_name = TASK_ALIASES.get(task, task)
        ax.scatter([], [], color=color, label=display_name, s=50)

    # Legend for start/end
    ax.scatter([], [], color="gold", label="Start (Base)", s=50, edgecolors="black")
    ax.scatter([], [], color="purple", label="End (Final)", s=50, edgecolors="black")

    ax.legend(loc="upper right", title="Task / Phase")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Optimization Trajectories in Semantic Space")

    fig.tight_layout()
    return fig


def plot_umap_trajectories_by_task(
    umap_df: pd.DataFrame,
    metadata: pd.DataFrame,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Plot UMAP trajectories with one subplot per task.

    Points are colored by normalized phase (0=start/yellow, 1=end/purple).

    Args:
        umap_df: DataFrame from reduce_to_umap()
        metadata: Metadata DataFrame
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Build session metadata mapping
    task_map = dict(zip(metadata["session_id"], metadata["task"]))

    # Get unique tasks
    tasks = sorted(set(task_map.values()))
    n_tasks = len(tasks)

    # Create subplots - arrange in 2 rows
    n_cols = (n_tasks + 1) // 2
    n_rows = 2 if n_tasks > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Colormap for phase progression (yellow -> purple)
    cmap = plt.cm.plasma
    norm = Normalize(vmin=0, vmax=1)

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        display_name = TASK_ALIASES.get(task, task)

        # Get sessions for this task
        task_sessions = [sid for sid, t in task_map.items() if t == task]

        for session_id in task_sessions:
            session_data = umap_df[umap_df["session_id"] == session_id].sort_values("phase")

            if len(session_data) < 2:
                continue

            x = session_data["x"].values
            y = session_data["y"].values
            phases = session_data["phase"].values

            # Normalize phases to [0, 1]
            max_phase = phases.max()
            if max_phase > 0:
                norm_phases = phases / max_phase
            else:
                norm_phases = np.zeros_like(phases)

            # Plot line (gray, thin)
            ax.plot(x, y, color="gray", alpha=0.3, linewidth=0.8)

            # Plot all points colored by normalized phase
            _ = ax.scatter(
                x,
                y,
                c=norm_phases,
                cmap=cmap,
                norm=norm,
                s=40,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
            )

        ax.set_title(display_name, fontsize=12)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)

    # Hide unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)

    # Add horizontal colorbar at bottom
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized Phase (0=start, 1=end)", fontsize=10)

    fig.suptitle("Optimization Trajectories by Task", fontsize=14)
    fig.subplots_adjust(bottom=0.12)
    return fig


def plot_final_clusters(
    cluster_df: pd.DataFrame,
    metadata: pd.DataFrame,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot clustered final embeddings with cluster as color and task as shape.

    Args:
        cluster_df: DataFrame from cluster_final_embeddings()
        metadata: Metadata DataFrame
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Merge with metadata
    task_map = dict(zip(metadata["session_id"], metadata["task"]))
    cluster_df = cluster_df.copy()
    cluster_df["task"] = cluster_df["session_id"].map(task_map)

    # Define markers for tasks
    tasks = sorted(cluster_df["task"].dropna().unique())
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]  # circle, square, triangle, diamond, etc.
    task_markers = {task: markers[i % len(markers)] for i, task in enumerate(tasks)}

    # Define colors for clusters
    clusters = sorted(cluster_df["cluster"].unique())
    n_clusters = len(clusters)
    cluster_colors = {c: plt.cm.tab10(i / max(n_clusters, 1)) for i, c in enumerate(clusters)}

    # Plot each task × cluster combination
    for task in tasks:
        for cluster in clusters:
            mask = (cluster_df["task"] == task) & (cluster_df["cluster"] == cluster)
            if not mask.any():
                continue
            ax.scatter(
                cluster_df.loc[mask, "x"],
                cluster_df.loc[mask, "y"],
                c=[cluster_colors[cluster]],
                marker=task_markers[task],
                s=100,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

    # Create legends
    # Task legend (shapes)
    task_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=task_markers[t],
            color="gray",
            linestyle="",
            markersize=10,
            label=TASK_ALIASES.get(t, t),
        )
        for t in tasks
    ]
    legend1 = ax.legend(handles=task_handles, loc="upper left", title="Task", fontsize=9)
    ax.add_artist(legend1)

    # Cluster legend (colors)
    cluster_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=cluster_colors[c],
            linestyle="",
            markersize=10,
            label=f"Cluster {c}" if c >= 0 else "Noise",
        )
        for c in clusters
    ]
    ax.legend(handles=cluster_handles, loc="upper right", title="Cluster", fontsize=9)

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title("Final Optimization States (color=cluster, shape=task)", fontsize=12)

    fig.tight_layout()
    return fig


def generate_paper_figures(
    analyses: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    output_dir: Path | None = None,
    project_path: Path | str | None = None,
) -> dict[str, plt.Figure]:
    """Generate all paper figures.

    Args:
        analyses: Dict of session_id -> analysis DataFrame
        metadata: Metadata DataFrame
        output_dir: Directory to save figures (default: FIGURES_DIR)
        project_path: Path to git repo (needed for UMAP figure)

    Returns:
        Dict mapping figure name -> Figure object
    """
    from .data import compute_optimal_discovery
    from .tags import compute_tag_distribution

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Figure 1: Tag probability by scaffold
    print("Generating tag distribution by scaffold...")
    tag_dist_scaffold = compute_tag_distribution(analyses, metadata, group_by="scaffold")
    fig1 = plot_tag_probability_by_phase(tag_dist_scaffold, group_by="scaffold")
    fig1.savefig(output_dir / "tag_prob_by_scaffold.png", dpi=150, bbox_inches="tight")
    figures["tag_prob_by_scaffold"] = fig1

    # Figure 2: Tag probability by task
    print("Generating tag distribution by task...")
    tag_dist_task = compute_tag_distribution(analyses, metadata, group_by="task")
    fig2 = plot_tag_probability_by_phase(tag_dist_task, group_by="task")
    fig2.savefig(output_dir / "tag_prob_by_task.png", dpi=150, bbox_inches="tight")
    figures["tag_prob_by_task"] = fig2

    # Figure 3: Entropy by phase (by task)
    print("Generating entropy by phase plot...")
    fig3 = plot_entropy_by_phase(tag_dist_task, group_by="task")
    fig3.savefig(output_dir / "entropy_by_phase_task.png", dpi=150, bbox_inches="tight")
    figures["entropy_by_phase_task"] = fig3

    # Figure 4: Subtype distribution
    print("Generating subtype distribution plot...")
    from .tags import compute_subtype_distribution

    subtype_dist = compute_subtype_distribution(analyses, metadata, group_by="scaffold")
    fig4 = plot_subtype_distribution(subtype_dist, group_by="scaffold", top_n_subtypes=5)
    fig4.savefig(output_dir / "subtype_distribution.png", dpi=150, bbox_inches="tight")
    figures["subtype_distribution"] = fig4

    # Figure 5: Optimal discovery
    print("Generating optimal discovery plot...")
    discovery_df = compute_optimal_discovery(
        analyses, metadata, score_type="validation", fallback="train"
    )
    fig5 = plot_optimal_discovery(discovery_df)
    fig5.savefig(output_dir / "optimal_discovery.png", dpi=150, bbox_inches="tight")
    figures["optimal_discovery"] = fig5

    # Figures 6-8: UMAP/embedding-based figures (requires embeddings)
    if project_path is not None:
        from .embeddings import cluster_final_embeddings, load_or_compute_embeddings, reduce_to_umap

        print("Computing/loading embeddings for UMAP...")
        session_ids = list(analyses.keys())
        embeddings = load_or_compute_embeddings(session_ids, project_path)

        print("Reducing to UMAP...")
        umap_df = reduce_to_umap(embeddings)

        # Figure 6: Basic UMAP trajectories
        print("Generating UMAP trajectory plot...")
        fig6 = plot_umap_trajectories(umap_df, metadata)
        fig6.savefig(output_dir / "umap_trajectories.png", dpi=150, bbox_inches="tight")
        figures["umap_trajectories"] = fig6

        # Figure 7: UMAP trajectories by task
        print("Generating UMAP trajectories by task plot...")
        fig7 = plot_umap_trajectories_by_task(umap_df, metadata)
        fig7.savefig(output_dir / "umap_trajectories_by_task.png", dpi=150, bbox_inches="tight")
        figures["umap_trajectories_by_task"] = fig7

        # Figure 8: Final clusters
        print("Generating final clusters plot...")
        cluster_df = cluster_final_embeddings(embeddings, n_clusters=5)
        fig8 = plot_final_clusters(cluster_df, metadata)
        fig8.savefig(output_dir / "final_clusters.png", dpi=150, bbox_inches="tight")
        figures["final_clusters"] = fig8
    else:
        print("Skipping UMAP/embedding figures (no project_path provided)")

    # Example trajectories (from CSV)
    if project_path is not None:
        print("Generating example trajectory plots...")
        example_figures = generate_example_trajectories(project_path, output_dir)
        figures.update(example_figures)
    else:
        print("Skipping example trajectories (no project_path provided)")

    print(f"Saved {len(figures)} figures to {output_dir}")
    return figures


async def _generate_example_trajectories_async(
    project_path: Path | str,
    output_dir: Path,
) -> dict[str, plt.Figure]:
    """Async helper to generate example trajectory plots."""
    import pandas as pd
    from vero.traces.analysis import TraceAnalyzer, plot_session_scores_with_table

    # Load example sessions CSV
    examples_csv = output_dir.parent / "example_trajectories.csv"
    if not examples_csv.exists():
        print(f"  No example_trajectories.csv found at {examples_csv}")
        return {}

    examples = pd.read_csv(examples_csv)
    analyzer = TraceAnalyzer()
    traj_output_dir = output_dir.parent / "example_trajectories"
    traj_output_dir.mkdir(parents=True, exist_ok=True)

    # Tasks where test scores should be excluded
    exclude_test_for = ["gaia"]

    figures = {}
    for _, row in examples.iterrows():
        session_id = row["session_id"]
        label = row["label"]
        task = row["task"]

        print(f"  Processing: {label} ({task})")

        try:
            payload, analysis = await analyzer.analyze_session(
                session_id=session_id,
                project_path=str(project_path),
                return_payload=True,
                use_cache=True,
            )

            # Drop test columns for certain tasks
            if task in exclude_test_for:
                test_cols = [c for c in analysis.columns if c.startswith("test_")]
                analysis = analysis.drop(columns=test_cols)

            fig = plot_session_scores_with_table(analysis, title=f"{label} ({task})")
            filename = f"{session_id[:8]}_{label.replace(' ', '_').replace('+', 'and')}.png"
            fig.savefig(traj_output_dir / filename, dpi=150, bbox_inches="tight")
            figures[f"traj_{session_id[:8]}"] = fig
            plt.close(fig)
        except Exception as e:
            print(f"  Error processing {session_id}: {e}")

    return figures


def generate_example_trajectories(
    project_path: Path | str,
    output_dir: Path,
) -> dict[str, plt.Figure]:
    """Generate example trajectory plots from example_trajectories.csv."""
    import asyncio

    return asyncio.run(_generate_example_trajectories_async(project_path, output_dir))
