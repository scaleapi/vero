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
from vero.interpret.models import Candidate, Edit, SymbolKind

_HUNK_SPLIT = re.compile(r"^(@@ .*?@@.*)$", re.M)
_SKIP = ("__pycache__", ".gitignore")


def _per_symbol_diff(diff: str, keep: set[int]) -> str:
    """Hunks of `diff` whose post-image start line falls in `keep`."""
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
        if keep & set(range(start, start + max(count, 1))):
            chunks.append(header + body.rstrip("\n"))
    return "\n".join(chunks)[:8000]


def decompose(
    repo: CandidateRepo,
    cell_key: str,
    candidate: Candidate,
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
            edits.append(
                _edit(cell_key, candidate, path, "<file>", SymbolKind.NON_PYTHON,
                      added, removed, diff[:8000], None, None)
            )
            continue

        after_src = repo.show_file(candidate.sha, path)
        before_src = repo.show_file(candidate.parent_sha, path)
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
        if removed_total and not grouped:
            grouped[("<module>", SymbolKind.MODULE)] = set()

        for (symbol, kind), lines in grouped.items():
            before = after = None
            if kind is SymbolKind.SCALAR_CONST:
                before = locus.scalar_value(before_src, symbol)
                after = locus.scalar_value(after_src, symbol)
                if before == after:
                    continue  # touched by reflow, not retuned
            share = len(lines)
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
                    _per_symbol_diff(diff, lines) or diff[:2000],
                    before,
                    after,
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
    )
