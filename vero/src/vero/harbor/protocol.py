"""Wire types for the eval sidecar's HTTP frontend, and the redaction that
projects a full Experiment down to what the agent may see.

`EvalRequest` (the request) lives in `vero.evaluation.engine` — it is shared with
the in-process tool. The *response* types here are sidecar-specific: they are
aggregate-safe by construction (never per-sample), because per-sample detail is
delivered as files on the agent-readable volume, gated by split tier.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from vero.core.budget import SplitBudget
from vero.core.dataset.base import SplitAccess, SplitAccessLevel
from vero.core.db.database import Experiment

logger = logging.getLogger(__name__)


@dataclass
class EvalSummary:
    """Aggregate-safe response to an agent evaluate call.

    Carries no per-sample data. Per-sample detail (for visible splits) and
    summary stats (for partial splits) are written to the agent-readable volume
    at `result_path`; nothing is written there for no_access splits.
    """

    commit: str
    split: str
    dataset_id: str
    n_samples: int
    mean_score: float | None
    result_path: str | None  # where on the agent volume to read detail (None if nothing written)
    budget_remaining: SplitBudget | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.budget_remaining is not None:
            d["budget_remaining"] = asdict(self.budget_remaining)
        return d


@dataclass
class StatusSummary:
    """Response to a status call. `submit_enabled` (not the verifier-internal
    selection strategy) is what the agent needs to know."""

    submit_enabled: bool
    # per (split, dataset_id): tier + whether the agent may evaluate it + remaining budget
    splits: list[dict] = field(default_factory=list)
    # the seeded baseline sha and whether its one budget-free reference eval
    # is still available (None/False when the task has no recorded baseline)
    base_commit: str | None = None
    free_baseline_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def tier_for_split(split: str, split_accesses: list[SplitAccess]) -> SplitAccessLevel:
    """Resolve a split's visibility tier.

    Fails CLOSED: an unlisted split is treated as ``no_access`` (not
    agent-evaluable, nothing written to the agent volume). A split with no
    declared access is a misconfiguration, not an invitation to expose labels,
    so we log a warning and deny rather than default to the most permissive tier.
    """
    for sa in split_accesses:
        if sa.split == split:
            return sa.access
    logger.warning(
        "Split %r has no declared access level; failing closed to no_access. "
        "Declare it explicitly in split_accesses if the agent should see it.",
        split,
    )
    return SplitAccessLevel.no_access


def summarize_experiment(
    experiment: Experiment,
    *,
    result_path: str | None,
    budget_remaining: SplitBudget | None = None,
) -> EvalSummary:
    """Project a full Experiment to an aggregate-safe EvalSummary."""
    return EvalSummary(
        commit=experiment.run.candidate.commit,
        split=experiment.run.dataset_subset.split,
        dataset_id=experiment.run.dataset_subset.dataset_id,
        n_samples=len(experiment.result.sample_results),
        mean_score=experiment.result.score(),
        result_path=result_path,
        budget_remaining=budget_remaining,
    )


def build_status(
    *,
    submit_enabled: bool,
    budget: dict[tuple[str, str], SplitBudget],
    split_accesses: list[SplitAccess],
    base_commit: str | None = None,
    free_baseline_available: bool = False,
    k_anonymity_floor: int = 1,
) -> StatusSummary:
    """Build the agent-facing status from the budget ledger + split tiers.

    Only budgeted (split, dataset_id) pairs are listed — those are exactly what
    the agent may evaluate. no_access splits are not in the agent ledger.
    """
    splits = []
    for (split, dataset_id), b in budget.items():
        tier = tier_for_split(split, split_accesses)
        splits.append(
            {
                "split": split,
                "dataset_id": dataset_id,
                "tier": str(tier),
                "agent_evaluable": tier != SplitAccessLevel.no_access,
                # Subset evals below this sample count are rejected on
                # non_viewable splits (full-split evals always pass). Advertised
                # so an agent plans subset probes instead of burning budget on
                # requests the sidecar will refuse.
                "min_subset_samples": (
                    k_anonymity_floor
                    if tier == SplitAccessLevel.non_viewable
                    else 1
                ),
                "remaining_sample_budget": b.remaining_sample_budget,
                "remaining_run_budget": b.remaining_run_budget,
            }
        )
    return StatusSummary(
        submit_enabled=submit_enabled,
        splits=splits,
        base_commit=base_commit,
        free_baseline_available=free_baseline_available,
    )
