"""Split a candidate into symbol-scoped edits.

The unit of analysis is one symbol touched by one candidate, not the candidate. A
single commit routinely bundles a genuine bug fix with unrelated tuning under one
subject line, so labelling per candidate assigns a single category to several
distinct modifications and the interesting one gets buried.

Everything here is deterministic. No model is consulted, so the resulting table is
reproducible and can be diffed between runs.
"""

from __future__ import annotations

import re
from collections import defaultdict

from vero.interpret.artifacts.harbor.repo import CandidateRepo
from vero.interpret.edits import locus
from vero.interpret.edits.provenance import provenance_between, provenance_of
from vero.interpret.labeling.taxonomy import Provenance
from vero.interpret.models import Candidate, Edit, SymbolKind

_HUNK_SPLIT = re.compile(r"^(@@ .*?@@.*)$", re.M)
# __pycache__ is compiled noise. .gitignore is NOT skipped: several cells shipped a
# candidate whose only change was one, and "shipped something inert" is a finding.
_SKIP = ("__pycache__",)
# The stored diff was capped at 8000 chars, which silently truncated the largest
# rewrites before any labeller could see them. Storage is cheap; keep the whole thing.
MAX_STORED_DIFF = 60_000
MAX_STORED_SOURCE = 20_000


def _per_symbol_diff(diff: str, keep: set[int], *, deletions: bool = False) -> str:
    """Hunks of `diff` whose post-image lines fall in `keep`.

    Deletion-only (`+start,0`) hunks match no post-image line, so they belong to no
    symbol; `deletions` puts them on the module row, which is where the file's
    removal count is carried, instead of dropping them from every stored diff.
    """
    parts = _HUNK_SPLIT.split(diff)
    if len(parts) < 2:
        return ""
    chunks: list[str] = []
    for header, body in zip(parts[1::2], parts[2::2]):
        match = locus._HUNK.match(header + "\n")
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or 1)
        if count == 0:
            if deletions:
                chunks.append(header + body.rstrip("\n"))
            continue
        if keep & set(range(start, start + count)):
            chunks.append(header + body.rstrip("\n"))
    return "\n".join(chunks)[:MAX_STORED_DIFF]


def decompose(
    repo: CandidateRepo,
    cell_key: str,
    candidate: Candidate,
    *,
    seed_sha: str | None = None,
    prior_symbols: set[tuple[str, str]] | None = None,
) -> list[Edit]:
    """Symbol-scoped edits introduced by `candidate` relative to its parent."""
    if candidate.is_seed or candidate.parent_sha is None:
        return []

    edits: list[Edit] = []
    for path in candidate.files:
        if any(s in path for s in _SKIP):
            continue

        diff = repo.diff(candidate.parent_sha, candidate.sha, path=path, context=0)
        if not diff.strip():
            continue

        if not path.endswith(".py"):
            added = sum(
                1 for line in diff.splitlines() if line.startswith("+") and line[1:2] != "+"
            )
            removed = sum(
                1 for line in diff.splitlines() if line.startswith("-") and line[1:2] != "-"
            )
            # Whole-file comparison; there is no symbol to resolve in a non-Python file.
            prov = provenance_of(repo, seed_sha or "", candidate.parent_sha, path, "<file>")
            edits.append(
                _edit(cell_key, candidate, path, "<file>", SymbolKind.NON_PYTHON,
                      added, removed, diff[:MAX_STORED_DIFF], None, None, None, True, 0,
                      prov.value)
            )
            continue

        after_src = repo.show_file(candidate.sha, path)
        before_src = repo.show_file(candidate.parent_sha, path)
        seed_src = repo.show_file(seed_sha, path) if seed_sha else ""
        mapping = locus.symbol_map(after_src)

        grouped: dict[tuple[str, SymbolKind], set[int]] = defaultdict(set)
        for line in locus.changed_lines(diff):
            symbol, kind = mapping.get(line, ("<module>", SymbolKind.MODULE))
            grouped[(symbol, kind)].add(line)

        # Deletions vanish from the post-image, so a symbol removed outright has no
        # line to map. Attribute the whole file's removals to <module> rather than
        # dropping them: "removed the audit pass" is a modification worth counting.
        removed_total = sum(
            1 for line in diff.splitlines() if line.startswith("-") and line[1:2] != "-"
        )
        added_total = sum(
            1 for line in diff.splitlines() if line.startswith("+") and line[1:2] != "+"
        )
        attributed_added = sum(len(v) for v in grouped.values())
        # Removals ride on the module row (see below), so that row has to exist
        # whenever the file lost lines — not only when nothing else was attributed.
        # A commit that adds a helper and deletes another function outright has a
        # populated `grouped` and would otherwise drop the deletion entirely.
        if removed_total and ("<module>", SymbolKind.MODULE) not in grouped:
            grouped[("<module>", SymbolKind.MODULE)] = set()

        for (symbol, kind), lines in grouped.items():
            before = after = None
            if kind is SymbolKind.SCALAR_CONST:
                before = locus.scalar_value(before_src, symbol)
                after = locus.scalar_value(after_src, symbol)
                if before == after:
                    continue  # touched by reflow, not retuned
            share = len(lines)
            # Context and history, so the labeller is not reading changed lines blind.
            after_source = locus.symbol_source(after_src, symbol)
            in_seed = bool(seed_src) and locus.symbol_source(seed_src, symbol) is not None
            touches = len([1 for k in (prior_symbols or set()) if k == (path, symbol)])
            # Derived here, not asked of the labeller: shown one edit in isolation the
            # model answers "own" almost everywhere. Only the tree comparison knows
            # whether this code is still as the seed left it.
            if not seed_sha:
                prov = Provenance.UNKNOWN
            elif candidate.parent_sha.startswith(seed_sha[:12]):
                prov = Provenance.SEED
            else:
                prov = provenance_between(seed_src, before_src, symbol)
            edits.append(
                _edit(
                    cell_key,
                    candidate,
                    path,
                    symbol,
                    kind,
                    share,
                    # Removals cannot be attributed per symbol; carry the file total
                    # on the module row so the count is never silently lost.
                    removed_total if symbol == "<module>" else 0,
                    _per_symbol_diff(diff, lines, deletions=symbol == "<module>")
                    or diff[:MAX_STORED_DIFF],
                    before,
                    after,
                    after_source[:MAX_STORED_SOURCE] if after_source else None,
                    in_seed,
                    touches,
                    prov.value,
                )
            )
        if attributed_added < added_total and grouped:
            pass  # unattributed remainder is comment/blank churn; not an edit
    return edits


def _edit(
    cell_key: str,
    candidate: Candidate,
    path: str,
    symbol: str,
    kind: SymbolKind,
    added: int,
    removed: int,
    diff: str,
    before: str | None,
    after: str | None,
    after_source: str | None,
    in_seed: bool,
    prior_touches: int,
    provenance: str,
) -> Edit:
    return Edit(
        id=Edit.make_id(cell_key, candidate.sha, path, symbol, diff),
        cell_key=cell_key,
        candidate_sha=candidate.sha,
        path=path,
        symbol=symbol,
        symbol_kind=kind,
        added=added,
        removed=removed,
        before_value=before,
        after_value=after,
        diff=diff,
        after_source=after_source,
        in_seed=in_seed,
        prior_touches=prior_touches,
        provenance=provenance,
    )
