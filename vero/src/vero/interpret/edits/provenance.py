"""Whose defect was it? Derived from the seed, not asked of a model.

Asking a model was tried and fails systematically. An edit shown in isolation
carries no history, so the model sees code being repaired inside the optimizer's own
agent file and answers "own" almost every time: on the first pass it returned 452
own against 3 seed corpus-wide, and labelled 21 of 22 swe-atlas submission fixes as
self-inflicted when those demonstrably repair a defect in the seed's answer parser
that 15 of 20 cells independently patched.

The question is not a judgement. If the code being repaired is still exactly as the
seed wrote it, the defect came with the seed; if an earlier candidate in the same
cell had already rewritten it, the optimizer is repairing itself. That is two tree
lookups.
"""

from __future__ import annotations

import ast

from vero.interpret.artifacts.harbor.repo import CandidateRepo
from vero.interpret.labeling.taxonomy import Provenance


def _symbol_source(source: str, symbol: str) -> str | None:
    """Source text of one qualified symbol, or None if absent."""
    if not source or symbol in ("<module>", "<file>"):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    want = symbol.split(".")

    def find(node: ast.AST, path: list[str]) -> ast.AST | None:
        if not path:
            return node
        for child in ast.iter_child_nodes(node):
            name = getattr(child, "name", None)
            if name == path[0]:
                return find(child, path[1:])
            if isinstance(child, (ast.Assign, ast.AnnAssign)) and len(path) == 1:
                targets = (
                    [child.target] if isinstance(child, ast.AnnAssign) else child.targets
                )
                if any(getattr(t, "id", None) == path[0] for t in targets):
                    return child
        return None

    found = find(tree, want)
    if found is None:
        return None
    try:
        return ast.unparse(found)
    except Exception:
        return None


def provenance_of(
    repo: CandidateRepo,
    seed_sha: str,
    parent_sha: str,
    path: str,
    symbol: str,
) -> Provenance:
    """SEED if the repaired code is untouched since the seed, OWN if not."""
    if not parent_sha or not seed_sha or parent_sha.startswith(seed_sha[:12]):
        return Provenance.SEED

    seed_src = repo.show_file(seed_sha, path)
    parent_src = repo.show_file(parent_sha, path)
    if not seed_src:
        # The file did not exist in the seed, so whatever is being fixed is the
        # optimizer's own work by construction.
        return Provenance.OWN
    if not parent_src:
        return Provenance.UNKNOWN

    seed_sym = _symbol_source(seed_src, symbol)
    parent_sym = _symbol_source(parent_src, symbol)
    if seed_sym is None or parent_sym is None:
        # Fall back to whole-file comparison: coarser, but still decided by content
        # rather than by guess.
        return Provenance.SEED if seed_src == parent_src else Provenance.OWN
    return Provenance.SEED if seed_sym == parent_sym else Provenance.OWN
