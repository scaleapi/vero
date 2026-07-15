"""Database utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.tree import Tree

    from vero.evaluation import EvaluationDatabase


def render_candidate_graph(db: EvaluationDatabase) -> Tree:
    """Renders a DAG of candidates based on commit-parent commit relationships using Rich Tree.

    Args:
        db: The experiment database to render.

    Returns:
        A Rich Tree object representing the candidate graph.
    """
    from rich.tree import Tree

    from vero.core.db.candidate import Candidate

    if not db.candidates:
        tree = Tree("[bold red]No candidates in database[/bold red]")
        return tree

    # Build parent-child relationships
    children_map: dict[str, list[Candidate]] = {}
    roots: list[Candidate] = []
    known_commits = {candidate.commit for candidate in db.candidates.values()}

    for candidate in db.candidates.values():
        if candidate.parent_commit is None:
            roots.append(candidate)
        else:
            # Historical sessions sometimes stored abbreviated hashes. Resolve
            # an unambiguous prefix while keeping full canonical hashes intact.
            matches = [
                commit
                for commit in known_commits
                if commit.startswith(candidate.parent_commit)
                or candidate.parent_commit.startswith(commit)
            ]
            parent_id = matches[0] if len(matches) == 1 else candidate.parent_commit
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
        candidate_evaluations = [
            evaluation
            for evaluation in db.get_evaluations()
            if evaluation.request.candidate == candidate
        ]

        # Build node info with rich formatting
        node_info = f"[yellow]{candidate.commit}[/yellow]"
        if candidate_evaluations:
            scored = [
                evaluation
                for evaluation in candidate_evaluations
                if evaluation.objective is not None
                and evaluation.objective.value is not None
            ]
            error_rates = [
                evaluation.report.metrics["error_rate"]
                for evaluation in candidate_evaluations
                if "error_rate" in evaluation.report.metrics
            ]
            if scored:
                latest = max(scored, key=lambda evaluation: evaluation.completed_at)
                assert latest.objective is not None
                assert latest.objective.value is not None
                metric = (
                    latest.objective_spec.selector.metric
                    if latest.objective_spec is not None
                    else "objective"
                )
                node_info += (
                    f" [dim]([/dim][green]{metric}: "
                    f"{latest.objective.value:.3f}[/green]"
                )
                if error_rates:
                    avg_error = sum(error_rates) / len(error_rates)
                    node_info += f"[dim],[/dim] [red]error: {avg_error:.3f}[/red]"
                if len(candidate_evaluations) > 1:
                    node_info += f"[dim], n={len(candidate_evaluations)}[/dim]"
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
