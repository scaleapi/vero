"""CLI entry point for analysis module.

Usage:
    python -m vero_benchmarking.analysis [--project-path PATH]
"""

import argparse
from pathlib import Path

from vero_benchmarking.utils import get_path_to_vero_agents

from .config import DEFAULT_SCAFFOLDS, FIGURES_DIR
from .data import filter_data, load_analyses
from .plots import generate_paper_figures


def main():
    parser = argparse.ArgumentParser(description="Generate analysis figures for VeRO paper")
    parser.add_argument(
        "--project-path",
        type=Path,
        default=None,
        help="Path to vero-agents repo (for UMAP embeddings). Default: auto-detect",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help=f"Output directory for figures. Default: {FIGURES_DIR}",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP figure (faster, no embedding computation)",
    )
    args = parser.parse_args()

    # Auto-detect project path
    project_path = args.project_path
    if project_path is None and not args.skip_umap:
        try:
            project_path = get_path_to_vero_agents()
            print(f"Auto-detected project path: {project_path}")
        except Exception:
            print("Could not auto-detect project path. Skipping UMAP figure.")
            project_path = None

    # Load and filter data
    print("Loading analyses...")
    analyses, metadata = load_analyses()
    print(f"Loaded {len(analyses)} sessions")

    print(f"Filtering to scaffolds: {DEFAULT_SCAFFOLDS}")
    analyses, metadata = filter_data(analyses, metadata)
    print(f"Filtered to {len(analyses)} sessions")

    # Generate figures
    figures = generate_paper_figures(
        analyses,
        metadata,
        output_dir=args.output_dir,
        project_path=project_path if not args.skip_umap else None,
    )

    print(f"\nGenerated {len(figures)} figures:")
    for name in figures:
        print(f"  - {name}.png")


if __name__ == "__main__":
    main()
