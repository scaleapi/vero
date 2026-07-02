"""Per-split evaluation budgets and the ledger that meters them.

``SplitBudget`` is the public, stateful budget for one (split, dataset_id) pair.
``BudgetLedger`` owns a set of them — the keys also form the allowlist of
evaluable combinations. The ledger is in-memory by default (the in-process
``ExperimentRunnerTool``); with a ``persist_path`` it flushes every mutation to
durable JSON under a single-writer lock (the Harbor eval sidecar).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from vero.exceptions import ExperimentBudgetExceeded, InvalidSplitError

logger = logging.getLogger(__name__)


@dataclass
class SplitBudget:
    """A stateful object that tracks the remaining budget for running experiments."""

    split: str
    dataset_id: str = ""
    total_sample_budget: int | None = None
    remaining_sample_budget: int | None = field(init=False)
    total_run_budget: int | None = None
    remaining_run_budget: int | None = field(init=False)
    max_samples_per_run: int | None = None

    def __repr__(self) -> str:
        repr_items = [
            ("split", self.split),
            ("dataset_id", self.dataset_id),
            ("total_sample_budget", self.total_sample_budget),
            ("total_run_budget", self.total_run_budget),
        ]
        repr_items = [item for item in repr_items if item[1] is not None]
        return (
            f"SplitBudget({', '.join([f'{item[0]}={item[1]}' for item in repr_items])})"
        )

    def __post_init__(self):
        assert (
            self.total_sample_budget is not None or self.total_run_budget is not None
        ), "Either total sample budget or total run budget must be provided."
        self.remaining_sample_budget = self.total_sample_budget
        self.remaining_run_budget = self.total_run_budget

        assert (
            isinstance(self.total_sample_budget, int)
            or self.total_sample_budget is None
        )
        assert isinstance(self.total_run_budget, int) or self.total_run_budget is None
        assert (
            isinstance(self.max_samples_per_run, int)
            or self.max_samples_per_run is None
        )

    def has_run_budget(self) -> bool:
        return self.remaining_run_budget is None or self.remaining_run_budget > 0

    def decrement_run_budget(self) -> None:
        if self.remaining_run_budget is not None:
            self.remaining_run_budget -= 1

    def has_sample_budget(self, num_samples: int) -> bool:
        return (
            self.remaining_sample_budget is None
            or self.remaining_sample_budget >= num_samples
        )

    def decrement_sample_budget(self, num_samples: int) -> None:
        if self.remaining_sample_budget is not None:
            self.remaining_sample_budget -= num_samples

    def exceeds_per_run_budget(self, num_samples: int) -> bool:
        return (
            self.max_samples_per_run is not None
            and num_samples > self.max_samples_per_run
        )


class BudgetLedger:
    """Meters evaluation budget across (split, dataset_id) pairs.

    The keys are also the allowlist of evaluable combinations: a pair with no
    budget entry is rejected by ``validate``.

    In-memory by default. Pass ``persist_path`` for the durable, crash-safe
    variant used by the Harbor sidecar — every mutation is flushed under a
    single-writer lock, and ``reserve`` checks-and-decrements atomically before a
    run so concurrent callers cannot overspend. Budget is never refunded on error.
    """

    def __init__(
        self,
        budgets: list[SplitBudget] | None = None,
        *,
        persist_path: Path | str | None = None,
    ):
        self._budgets: dict[tuple[str, str], SplitBudget] = {
            (b.split, b.dataset_id): b for b in (budgets or [])
        }
        self.persist_path = Path(persist_path) if persist_path else None
        self._lock = asyncio.Lock()

    def validate(self, dataset_id: str, split: str) -> None:
        """Raise if (split, dataset_id) is not an allowed combination."""
        if (split, dataset_id) not in self._budgets:
            allowed_keys = list(self._budgets.keys())
            raise InvalidSplitError(
                f"No split budget found for the combination (dataset_id={dataset_id}, split={split}) "
                f"either because it does not exist or because it is not allowed. "
                f"Allowed combinations: {allowed_keys}"
            )

    def get(self, dataset_id: str, split: str) -> SplitBudget:
        """Return the budget for a pair (validates membership first)."""
        self.validate(dataset_id, split)
        return self._budgets[(split, dataset_id)]

    def check(self, dataset_id: str, split: str, num_samples: int) -> None:
        """Raise ExperimentBudgetExceeded if the request would exceed the budget."""
        budget = self.get(dataset_id, split)
        if not budget.has_run_budget():
            raise ExperimentBudgetExceeded(
                f"No runs left for the {split} split of the {dataset_id} dataset."
            )
        if not budget.has_sample_budget(num_samples):
            raise ExperimentBudgetExceeded(
                f"Requested {num_samples} samples for the {split} split of the {dataset_id} dataset, "
                f"but the remaining sample budget only allows for {budget.remaining_sample_budget} samples."
            )
        if budget.exceeds_per_run_budget(num_samples):
            raise ExperimentBudgetExceeded(
                f"Requested {num_samples} samples for the {split} split of the {dataset_id} dataset, "
                f"but only {budget.max_samples_per_run} are allowed per run."
            )

    def _decrement(self, dataset_id: str, split: str, num_samples: int) -> SplitBudget:
        """In-memory decrement only (no flush). Caller is responsible for durability."""
        budget = self.get(dataset_id, split)
        budget.decrement_sample_budget(num_samples)
        budget.decrement_run_budget()
        return budget

    def record(self, dataset_id: str, split: str, num_samples: int) -> SplitBudget:
        """Decrement the budget for a completed (or attempted) run and flush."""
        budget = self._decrement(dataset_id, split, num_samples)
        self._flush()
        return budget

    async def reserve(
        self, dataset_id: str, split: str, num_samples: int
    ) -> SplitBudget:
        """Atomically check + record before a run (durable, single-writer).

        Raises InvalidSplitError / ExperimentBudgetExceeded *before* decrementing,
        so a rejected request costs nothing; a reserved request is never refunded.
        """
        async with self._lock:
            self.check(dataset_id, split, num_samples)
            budget = self._decrement(dataset_id, split, num_samples)
            # Flush off the event loop (to_thread) but still under the lock: the
            # to_thread call keeps the synchronous write from blocking the loop
            # (the actual bug), while holding the lock keeps concurrent flushes
            # from racing on the shared temp file and preserves decrement/flush
            # ordering so a crash never persists more remaining budget than was
            # committed.
            if self.persist_path is not None:
                await asyncio.to_thread(self._flush)
        return budget

    def status(self) -> dict[tuple[str, str], SplitBudget]:
        """Return all budgets keyed by (split, dataset_id)."""
        return dict(self._budgets)

    def _flush(self) -> None:
        if self.persist_path is None:
            return
        data = [
            {
                "split": b.split,
                "dataset_id": b.dataset_id,
                "total_sample_budget": b.total_sample_budget,
                "remaining_sample_budget": b.remaining_sample_budget,
                "total_run_budget": b.total_run_budget,
                "remaining_run_budget": b.remaining_run_budget,
                "max_samples_per_run": b.max_samples_per_run,
            }
            for b in self._budgets.values()
        ]
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.persist_path)
