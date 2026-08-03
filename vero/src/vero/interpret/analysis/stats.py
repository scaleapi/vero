"""Aggregations behind the figures. Pure functions over labelled edits.

Two rules run through all of these.

Prevalence counts **cells, not edits**. Cells produced between 1 and 18 candidates,
so an edit-weighted count answers "which cells were prolific" while pretending to
answer "what did optimizers try".

Diversity needs a **null**. A mean pairwise Jaccard distance of 0.5 is
uninterpretable on its own: it could mean cells explore genuinely different
repertoires, or simply that each drew a few roles from the same skewed marginal. The
permutation null holds each cell's repertoire *size* and the corpus-wide role
frequencies fixed and reshuffles which cell got what, so the comparison isolates
whether cells differ beyond chance.
"""

from __future__ import annotations

import itertools
import random
import statistics as st
from collections import Counter, defaultdict

BENCH_ORDER = [
    "browsecomp-plus",
    "officeqa",
    "swe-atlas-qna",
    "terminal-bench",
    "gaia-shell",
]

# gaia-shell's seed is an empty shell, so every role is "present" by construction
# rather than by choice. It is shown but never pooled with the rest.
CONSTRUCTED_SEED = {"gaia-shell"}


def cell_roles(rows: list[dict]) -> dict[str, set[str]]:
    """cell_key -> the set of roles it ever touched."""
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        out[r["cell_key"]].add(r["role"])
    return dict(out)


def benchmark_cells(rows: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        out[r["cell_key"].split("/")[1]].add(r["cell_key"])
    return dict(out)


def prevalence(rows: list[dict]) -> tuple[list[str], dict[str, dict[str, tuple[int, int]]]]:
    """role -> benchmark -> (cells that used it, cells in the benchmark)."""
    roles_by_cell = cell_roles(rows)
    cells_by_bench = benchmark_cells(rows)
    roles = sorted({r for s in roles_by_cell.values() for r in s})
    table: dict[str, dict[str, tuple[int, int]]] = {}
    for role in roles:
        table[role] = {}
        for bench, cells in cells_by_bench.items():
            hit = sum(1 for c in cells if role in roles_by_cell.get(c, set()))
            table[role][bench] = (hit, len(cells))
    # Order roles by how universal they are, so the figure reads top-down.
    roles.sort(key=lambda r: -sum(h / t for h, t in table[r].values()))
    return roles, table


def rarefaction(rows: list[dict], *, trials: int = 200, seed: int = 0) -> dict[str, list[float]]:
    """Mean distinct roles discovered after k cells, averaged over orderings.

    A curve that flattens says the k-th optimizer tried nothing the first k-1 had
    not already tried; one still climbing at k=20 says the repertoire is not
    exhausted by the sample.
    """
    rng = random.Random(seed)
    roles_by_cell = cell_roles(rows)
    out: dict[str, list[float]] = {}
    for bench, cells in benchmark_cells(rows).items():
        members = sorted(cells)
        totals = [0.0] * len(members)
        for _ in range(trials):
            rng.shuffle(members)
            seen: set[str] = set()
            for i, cell in enumerate(members):
                seen |= roles_by_cell.get(cell, set())
                totals[i] += len(seen)
        out[bench] = [t / trials for t in totals]
    return out


def jaccard(rows: list[dict], *, trials: int = 500, seed: int = 0) -> dict[str, dict]:
    """Observed mean pairwise distance per benchmark, against a permutation null."""
    rng = random.Random(seed)
    roles_by_cell = cell_roles(rows)
    out: dict[str, dict] = {}
    for bench, cells in benchmark_cells(rows).items():
        sets = [roles_by_cell.get(c, set()) for c in sorted(cells)]
        sets = [s for s in sets if s]
        if len(sets) < 3:
            continue
        observed = st.mean(
            1 - len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)
        )
        # Null: keep each cell's repertoire size and the corpus role frequencies,
        # reshuffle the assignment.
        pool: list[str] = []
        for s in sets:
            pool.extend(s)
        freq = Counter(pool)
        vocab = list(freq)
        weights = [freq[v] for v in vocab]
        null: list[float] = []
        for _ in range(trials):
            drawn = []
            for s in sets:
                picked: set[str] = set()
                while len(picked) < len(s):
                    picked.add(rng.choices(vocab, weights=weights, k=1)[0])
                drawn.append(picked)
            null.append(
                st.mean(
                    1 - len(a & b) / len(a | b) for a, b in itertools.combinations(drawn, 2)
                )
            )
        null.sort()
        lo, hi = null[int(0.025 * len(null))], null[int(0.975 * len(null)) - 1]
        out[bench] = {
            "observed": observed,
            "null_mean": st.mean(null),
            "null_lo": lo,
            "null_hi": hi,
            "n_cells": len(sets),
            # Below the null: cells are MORE alike than chance -> convergence.
            "verdict": "converged" if observed < lo else ("diverged" if observed > hi else "as chance"),
        }
    return out


def action_by_role(rows: list[dict], *, top_roles: int = 10) -> tuple[list[str], list[str], dict]:
    counts: dict[tuple[str, str], int] = Counter()
    for r in rows:
        counts[(r["role"], r["action"])] += 1
    role_totals = Counter()
    for (role, _), n in counts.items():
        role_totals[role] += n
    roles = [r for r, _ in role_totals.most_common(top_roles)]
    actions = [a for a, _ in Counter(r["action"] for r in rows).most_common()]
    return roles, actions, {k: v for k, v in counts.items() if k[0] in roles}


def tuning_direction(rows: list[dict], edits: dict[str, dict], *, top: int = 12) -> list[tuple[str, int, int]]:
    """(symbol, ups, downs) for scalar constants that actually changed."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        edit = edits.get(r["edit_id"])
        if not edit or edit["symbol_kind"] != "scalar_const":
            continue
        if r["direction"] == "up":
            counts[edit["symbol"]][0] += 1
        elif r["direction"] == "down":
            counts[edit["symbol"]][1] += 1
    ranked = sorted(counts.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
    return [(s, u, d) for s, (u, d) in ranked[:top]]


def provenance_of_fixes(rows: list[dict]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["action"] != "fix":
            continue
        out[r["cell_key"].split("/")[1]][r["provenance"]] += 1
    return dict(out)


def hint_agreement(rows: list[dict]) -> dict[str, int]:
    """How often the model's role matched the deterministic hint, where audited.

    Disagreements are recorded in `mechanism` as a "[hint=… model=…]" prefix, which
    is the only place both readings survive.
    """
    audited = [r for r in rows if r["hinted"] and r["mechanism"].startswith("[hint=")]
    hinted = [r for r in rows if r["hinted"]]
    return {
        "hinted": len(hinted),
        "disagreements": len(audited),
        "model_decided": len(rows) - len(hinted),
        "total": len(rows),
    }
