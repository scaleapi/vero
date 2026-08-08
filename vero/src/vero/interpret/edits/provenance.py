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

from vero.interpret.artifacts.harbor.repo import CandidateRepo
from vero.interpret.edits.locus import symbol_source
from vero.interpret.labeling.taxonomy import Provenance


def provenance_between(seed_src: str, parent_src: str, symbol: str) -> Provenance:
    """SEED if `symbol` is untouched between these two sources, OWN if not.

    Takes sources rather than shas because `decompose` has already read both files to
    build the symbol map and the before/after values; going back to git per symbol
    would be one subprocess per row for bytes already in memory. `provenance_of`
    below is the same decision for callers holding only shas.
    """
    if not seed_src:
        # The file did not exist in the seed, so whatever is being fixed is the
        # optimizer's own work by construction.
        return Provenance.OWN
    if not parent_src:
        return Provenance.UNKNOWN

    seed_sym = symbol_source(seed_src, symbol)
    parent_sym = symbol_source(parent_src, symbol)
    if seed_sym is None or parent_sym is None:
        # Fall back to whole-file comparison: coarser, but still decided by content
        # rather than by guess.
        return Provenance.SEED if seed_src == parent_src else Provenance.OWN
    return Provenance.SEED if seed_sym == parent_sym else Provenance.OWN


def provenance_of(
    repo: CandidateRepo,
    seed_sha: str,
    parent_sha: str,
    path: str,
    symbol: str,
) -> Provenance:
    """SEED if the repaired code is untouched since the seed, OWN if not."""
    if not parent_sha or not seed_sha:
        return Provenance.UNKNOWN
    if parent_sha.startswith(seed_sha[:12]):
        # Editing the seed itself: nothing else has touched this code yet.
        return Provenance.SEED
    return provenance_between(
        repo.show_file(seed_sha, path), repo.show_file(parent_sha, path), symbol
    )
