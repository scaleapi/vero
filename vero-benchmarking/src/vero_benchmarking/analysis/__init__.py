"""Analysis module for VeRO benchmark results.

Provides tools for loading, analyzing, and visualizing optimization
trajectories from VeRO benchmark runs.

Usage:
    from vero_benchmarking.analysis import (
        load_analyses,
        compute_tag_distribution,
        plot_tag_probability_by_phase,
        generate_paper_figures,
        batch_analyze_sessions,
        build_run_df_with_history,
    )
"""

# Configuration
from .config import (
    DEFAULT_SCAFFOLDS,
    EMBEDDINGS_CACHE_DIR,
    FIGURES_DIR,
    MODEL_ALIASES,
    SCAFFOLD_ALIASES,
    TASK_ALIASES,
)

# Data loading and filtering
from .data import (
    compute_optimal_discovery,
    filter_data,
    get_session_improvement,
    get_session_score,
    load_analyses,
    rank_sessions_by_improvement,
)

# Embeddings
from .embeddings import (
    compute_cumulative_diff_embeddings,
    load_or_compute_embeddings,
    reduce_to_umap,
)

# Plotting
from .plots import (
    generate_paper_figures,
    plot_entropy_by_phase,
    plot_optimal_discovery,
    plot_subtype_distribution,
    plot_tag_probability_by_phase,
    plot_umap_trajectories,
)

# W&B results extraction
from .results import (
    add_performance_metrics,
    build_run_df_with_history,
    extract_primary_fields,
    filter_quality,
)

# Batch session analysis
from .sessions import batch_analyze_sessions

# Tag analysis
from .tags import compute_subtype_distribution, compute_tag_distribution, parse_tags_from_row

__all__ = [
    # Config
    "FIGURES_DIR",
    "EMBEDDINGS_CACHE_DIR",
    "DEFAULT_SCAFFOLDS",
    "SCAFFOLD_ALIASES",
    "MODEL_ALIASES",
    "TASK_ALIASES",
    # Data
    "load_analyses",
    "filter_data",
    "get_session_score",
    "get_session_improvement",
    "rank_sessions_by_improvement",
    "compute_optimal_discovery",
    # Results (W&B extraction)
    "build_run_df_with_history",
    "extract_primary_fields",
    "add_performance_metrics",
    "filter_quality",
    # Sessions (batch analysis)
    "batch_analyze_sessions",
    # Tags
    "parse_tags_from_row",
    "compute_tag_distribution",
    "compute_subtype_distribution",
    # Embeddings
    "compute_cumulative_diff_embeddings",
    "load_or_compute_embeddings",
    "reduce_to_umap",
    # Plots
    "plot_tag_probability_by_phase",
    "plot_subtype_distribution",
    "plot_entropy_by_phase",
    "plot_optimal_discovery",
    "plot_umap_trajectories",
    "generate_paper_figures",
]
