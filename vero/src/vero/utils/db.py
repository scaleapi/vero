"""Database utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.tree import Tree

    from vero.core.db.database import ExperimentDatabase


def render_candidate_graph(db: ExperimentDatabase) -> Tree:
    """Renders a DAG of candidates based on commit-parent commit relationships using Rich Tree.

    Args:
        db: The experiment database to render.

    Returns:
        A Rich Tree object representing the candidate graph.
    """
    from rich.tree import Tree

    from vero.core.dataset import DefaultSplitNames
    from vero.core.db.candidate import Candidate

    if not db.candidates:
        tree = Tree("[bold red]No candidates in database[/bold red]")
        return tree

    # Build parent-child relationships
    children_map: dict[str, list[Candidate]] = {}
    roots: list[Candidate] = []

    for candidate in db.candidates.values():
        if candidate.parent_commit is None:
            roots.append(candidate)
        else:
            # Find parent by commit hash
            parent_id = candidate.parent_commit[:10]
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(candidate)

    # Sort roots by creation time
    roots.sort(key=lambda c: c.created_at)

    # Sort children by creation time for each parent
    for children in children_map.values():
        children.sort(key=lambda c: c.created_at)

    # Create the rich tree
    tree = Tree("[bold cyan]🌳 Candidate Graph[/bold cyan]", guide_style="dim")

    def render_node(parent_tree: Tree, candidate: Candidate):
        """Recursively render a node and its children."""
        # Get training split experiments for this candidate
        candidate_experiments = [
            experiment
            for experiment in db.get_experiments()
            if experiment.run.candidate == candidate
            and experiment.run.dataset_subset.split == DefaultSplitNames.train
        ]

        # Build node info with rich formatting
        node_info = f"[yellow]{candidate.commit}[/yellow]"
        if candidate_experiments:
            scores = [exp.result.score() for exp in candidate_experiments]
            scores = [score for score in scores if score is not None]
            error_rates = [exp.result.error_rate() for exp in candidate_experiments]
            if scores:
                avg_score = sum(scores) / len(scores)
                avg_error = sum(error_rates) / len(error_rates)
                node_info += f" [dim]([/dim][green]score: {avg_score:.3f}[/green][dim],[/dim] [red]error: {avg_error:.3f}[/red]"
                if len(candidate_experiments) > 1:
                    node_info += f"[dim], n={len(candidate_experiments)}[/dim]"
                node_info += "[dim])[/dim]"

        # Add the current node
        branch = parent_tree.add(node_info)

        # Render children
        children = children_map.get(candidate.commit, [])
        for child in children:
            render_node(branch, child)

    # Render all roots
    if not roots:
        warning_branch = tree.add(
            "[bold yellow]⚠️  Warning: No root candidates found (all candidates have parents)[/bold yellow]"
        )
        # Find orphaned candidates
        for candidate in db.candidates.values():
            render_node(warning_branch, candidate)
    else:
        for root in roots:
            render_node(tree, root)

    return tree
