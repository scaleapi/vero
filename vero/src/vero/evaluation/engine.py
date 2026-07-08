"""EvaluationEngine: the shared evaluation core.

Wraps the :class:`~vero.evaluator.Evaluator` with budget metering and the
dataset/split allowlist. It is the single eval path used by both the in-process
``ExperimentRunnerTool`` (in-memory budget) and the Harbor eval sidecar (durable
budget + HTTP frontend). It returns the **full** ``Experiment`` — redaction,
write-routing, and human/wire formatting are the frontend's job, not the core's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vero.core.budget import BudgetLedger, SplitBudget
from vero.core.dataset import (
    SplitAccess,
    SplitAccessLevel,
    resolve_split_access,
)
from vero.core.evaluation import BaseEvaluationParameters
from vero.exceptions import InvalidSplitError

if TYPE_CHECKING:
    from vero.core.db.database import Experiment, ExperimentDatabase
    from vero.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


@dataclass
class EvalRequest:
    """A request to evaluate a commit on a dataset split.

    Also the agent-facing wire payload in the Harbor case. ``task`` is
    not a field — it is fixed config bound on the service, not agent-chosen.
    """

    dataset_id: str
    split: str
    commit: str | None = None  # None -> resolved by the caller (e.g. agent repo HEAD)
    sample_ids: list[int] | None = None
    num_samples: int | None = None


class EvaluationEngine:
    """Resolve samples -> meter budget -> run the Evaluator -> full Experiment."""

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        budget: BudgetLedger,
        default_task: str | None = None,
        db: ExperimentDatabase | None = None,
        run_constraints: BaseEvaluationParameters | None = None,
        session_id: str | None = None,
        vero_home: Path | None = None,
        split_accesses: list[SplitAccess] | None = None,
    ):
        self.evaluator = evaluator
        self.budget = budget
        self.default_task = default_task
        self.db = db
        self.run_constraints = run_constraints or BaseEvaluationParameters()
        self.session_id = session_id
        self.vero_home = vero_home
        self.split_accesses = split_accesses

    @classmethod
    def from_session(cls, session) -> EvaluationEngine:
        """Build a service from a bound Session (mirrors ExperimentRunnerTool.bind)."""
        from copy import deepcopy

        return cls(
            evaluator=session.evaluator,
            budget=BudgetLedger(deepcopy(session.budget)),
            default_task=session.task,
            db=session.db,
            run_constraints=session.evaluation_parameters,
            session_id=session.session_id,
            vero_home=session.vero_home,
            split_accesses=session.split_accesses,
        )

    # ------------------------------------------------------------------
    # Dataset / sample resolution (lifted from ExperimentRunnerTool)
    # ------------------------------------------------------------------

    def _get_dataset_info(self, dataset_id: str):
        from vero.core.dataset import DatasetInfo
        from vero.core.dataset.store import load_dataset

        sessions_dir = self.vero_home / "sessions" if self.vero_home else None
        dataset_cache = self.vero_home / "datasets" if self.vero_home else None
        dataset = load_dataset(sessions_dir, dataset_cache, self.session_id, dataset_id)
        return DatasetInfo(
            id=dataset_id,
            splits={split: len(dataset[split]) for split in dataset},
            features={split: list(dataset[split].features) for split in dataset},
        )

    def _get_samples_from_split(
        self, dataset_id: str, split: str, num_samples: int
    ) -> list[int] | None:
        """First-N sample ids, or None when N covers (or exceeds) the whole split."""
        split_size = self._get_dataset_info(dataset_id).splits[split]
        num_samples = min(num_samples, split_size)
        if num_samples >= split_size:
            return None
        return list(range(num_samples))

    def _validate_and_count_samples(
        self, dataset_id: str, split: str, sample_ids: list[int] | None = None
    ) -> int:
        """Validate sample ids are in range; return the count (full split if None)."""
        split_size = self._get_dataset_info(dataset_id).splits[split]
        if sample_ids is None:
            return split_size
        invalid = [s for s in sample_ids if s < 0 or s >= split_size]
        if invalid:
            raise ValueError(
                f"The provided sample ids are outside the range of the split "
                f"[0, {split_size - 1}]: {invalid}"
            )
        return len(sample_ids)

    def resolve_samples(self, req: EvalRequest) -> tuple[list[int] | None, int]:
        """Resolve (sample_ids, count) for a request. Raises on invalid combos."""
        if req.sample_ids is not None and req.num_samples is not None:
            raise ValueError(
                "Cannot specify both sample_ids and num_samples. "
                "Use sample_ids for specific samples, or num_samples for the first N samples."
            )
        sample_ids = req.sample_ids
        if req.num_samples is not None:
            sample_ids = self._get_samples_from_split(
                req.dataset_id, req.split, req.num_samples
            )
        count = self._validate_and_count_samples(req.dataset_id, req.split, sample_ids)
        return sample_ids, count

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self, req: EvalRequest, *, admin: bool = False, free: bool = False
    ) -> Experiment:
        """Meter (unless admin or free) and run one evaluation; return the full
        Experiment.

        ``admin`` and ``free`` are distinct authorities and must stay separate:
        ``admin`` grants ACCESS (bypasses the tier gate and the ledger), while
        ``free`` only waives the BUDGET debit for an otherwise ordinary agent
        eval (the sidecar's one free baseline eval). A free eval still runs as
        the agent: conflating the two let the free-baseline path evaluate
        ``no_access`` splits and return their aggregate score to the agent.

        ``no_access`` gating is EXPLICIT and fail-closed: when ``split_accesses``
        is configured, the split's tier is resolved (an unlisted split defaults
        to ``no_access``) and a ``no_access`` split is hard-rejected for non-admin
        callers *before* the budget ledger is consulted. Admin bypasses both the
        tier gate and the ledger and may evaluate anything.
        """
        if not admin and self.split_accesses is not None:
            tier = resolve_split_access(req.split, self.split_accesses)
            if tier == SplitAccessLevel.no_access:
                raise InvalidSplitError(
                    f"Split '{req.split}' of dataset '{req.dataset_id}' is no_access "
                    f"and cannot be evaluated by the agent."
                )
        sample_ids, n = self.resolve_samples(req)
        if not admin and not free:
            await self.budget.reserve(req.dataset_id, req.split, n)
        return await self.evaluator.evaluate(
            commit=req.commit,
            dataset_id=req.dataset_id,
            split=req.split,
            task=self.default_task,
            sample_ids=sample_ids,
            db=self.db,
            evaluation_parameters=self.run_constraints,
        )

    async def evaluate_admin(
        self,
        *,
        task: str,
        dataset_id: str,
        split: str,
        commit: str,
        sample_ids: list[int] | None = None,
    ) -> Experiment:
        """Admin/verifier evaluation: explicit ``task``, no budget, no allowlist.

        Unlike :meth:`evaluate` (which is bound to ``default_task`` and metered),
        this scores an arbitrary ``(task, dataset_id, split)`` — including held-out
        tasks/splits the agent never had access to. Used by the verifier to score
        the selected commit on its configured targets.
        """
        return await self.evaluator.evaluate(
            commit=commit,
            dataset_id=dataset_id,
            split=split,
            task=task,
            sample_ids=sample_ids,
            db=self.db,
            evaluation_parameters=self.run_constraints,
        )

    def status(self) -> dict[tuple[str, str], SplitBudget]:
        """Remaining budget per (split, dataset_id)."""
        return self.budget.status()
