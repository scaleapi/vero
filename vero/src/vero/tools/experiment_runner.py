from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NoReturn

from vero.core.budget import BudgetLedger, SplitBudget
from vero.core.db.database import Experiment, ExperimentDatabase
from vero.core.evaluation import BaseEvaluationParameters
from vero.evaluation.evaluator import Evaluator
from vero.exceptions import (
    ExperimentBudgetExceeded,
    ExperimentRunFailedError,
)
from vero.evaluation.engine import EvalRequest, EvaluationEngine
from vero.tools.utils import is_tool

logger = logging.getLogger(__name__)

# SplitBudget moved to vero.core.budget; re-exported here for the public import path.
__all__ = ["ExperimentRunnerTool", "SplitBudget"]


def _default_on_fatal(msg: str) -> NoReturn:
    raise RuntimeError(msg)


@dataclass
class ExperimentRunnerTool:
    """Run target agents on tasks and get performance metrics."""

    exclude_tools: list[str] = field(default_factory=list)
    on_fatal: Callable[[str], NoReturn] = field(default=_default_on_fatal)

    # Runtime fields — set during bind()
    evaluator: Evaluator | None = None
    split_budgets: list[SplitBudget] | None = None
    run_constraints: BaseEvaluationParameters = field(
        default_factory=BaseEvaluationParameters
    )
    _task: str | None = None
    db: ExperimentDatabase | None = None
    _vero_home: Path | None = None
    _session_id: str | None = None
    # The shared evaluation core. This tool is a thin frontend over it (formats
    # results for the LLM, owns on_fatal); the Harbor sidecar is the other frontend.
    engine: EvaluationEngine | None = field(default=None, repr=False)

    def __post_init__(self):
        self._build_engine()

    def _build_engine(self) -> None:
        self.engine = EvaluationEngine(
            evaluator=self.evaluator,
            budget=BudgetLedger(self.split_budgets or []),
            default_task=self._task,
            db=self.db,
            run_constraints=self.run_constraints,
            session_id=self._session_id,
            vero_home=self._vero_home,
        )

    def bind(self, session) -> None:
        from copy import deepcopy

        self.evaluator = session.evaluator
        self.split_budgets = deepcopy(session.budget)
        self.db = session.db
        self._session_id = session.session_id
        self._vero_home = session.vero_home
        self.run_constraints = session.evaluation_parameters
        self._task = session.task
        self._build_engine()

    @property
    def _budget_ledger(self) -> BudgetLedger:
        return self.engine.budget

    @property
    def _budget_map(self) -> dict[tuple[str, str], SplitBudget]:
        """Back-compat view of the budget ledger, keyed (split, dataset_id).

        Returns the ledger's live SplitBudget objects (mutations propagate).
        """
        return self.engine.budget.status()

    def _get_dataset_info(self, dataset_id: str):
        """Get dataset info from the store (delegates to the shared service)."""
        return self.engine._get_dataset_info(dataset_id)

    async def _resolve_commit(self, commit: str) -> str:
        """Resolve a commit reference to its full hash.

        Args:
            commit: A commit reference (hash, short hash, HEAD, branch name, etc.)

        Returns:
            The full 40-character commit hash

        Raises:
            ValueError: If the commit reference cannot be resolved
        """
        from vero.workspace.git import GitWorkspace

        try:
            workspace = self.evaluator.workspace
            if isinstance(workspace, GitWorkspace):
                return await workspace.resolve_ref(commit)
            return commit
        except Exception as e:
            raise ValueError(
                f"Cannot resolve commit '{commit}': {e}. "
                f"Make sure the commit exists in the repository."
            )

    def _get_samples_from_split(
        self, dataset_id: str, split: str, num_samples: int
    ) -> list[int] | None:
        """First-N sample ids, or None for the whole split (delegates to the service)."""
        return self.engine._get_samples_from_split(dataset_id, split, num_samples)

    def _validate_and_count_samples(
        self, dataset_id: str, split: str, sample_ids: list[int] | None = None
    ) -> int:
        """Validate + count samples (delegates to the service)."""
        return self.engine._validate_and_count_samples(dataset_id, split, sample_ids)

    def _validate_split_access(self, dataset_id: str, split: str) -> None:
        """Validate that the split and dataset combination is allowed."""
        self._budget_ledger.validate(dataset_id, split)

    def _check_budget(
        self, dataset_id: str, split: str, requested_num_samples: int
    ) -> None:
        """Check that the budget allows for the requested number of samples."""
        self._budget_ledger.check(dataset_id, split, requested_num_samples)

    def _update_budget(self, dataset_id: str, split: str, num_samples: int) -> str:
        """Decrement the budget for a given dataset and split; return a status message."""
        budget = self._budget_ledger.record(dataset_id, split, num_samples)

        info = ""
        if budget.total_sample_budget is not None:
            info += f"Used {num_samples} samples from the total {budget.total_sample_budget} sample budget. Remaining sample budget: {budget.remaining_sample_budget}. "
        if budget.remaining_run_budget is not None:
            info += f"Used 1 run from the total {budget.total_run_budget} run budget. Remaining runs: {budget.remaining_run_budget}"

        return info

    async def _evaluate_commit(
        self,
        commit: str,
        dataset_id: str,
        split: str,
        sample_ids: list[int] | None = None,
    ) -> Experiment:
        """Run one evaluation via the shared EvaluationEngine.

        Uses ``admin=True`` so the service does not meter the budget — this tool
        owns budgeting via ``_check_budget``/``_update_budget`` (check-before,
        decrement-after) to preserve its existing semantics.
        """
        req = EvalRequest(
            dataset_id=dataset_id, split=split, commit=commit, sample_ids=sample_ids
        )
        try:
            return await self.engine.evaluate(req, admin=True)
        except ExperimentRunFailedError as e:
            if e.returncode >= 3:
                self.on_fatal(str(e))
            raise

    @is_tool
    async def check_remaining_experiment_budget(
        self, dataset_id: str, split: str
    ) -> str:
        """Get the remaining budget for a given dataset and split.

        Args:
            dataset_id: The id of the dataset.
            split: The split of the dataset.

        Returns:
            A string containing the remaining budget for the given dataset and split.
        """
        budget = self._budget_ledger.get(dataset_id, split)

        info = ""
        if budget.total_sample_budget is not None:
            info += f"Remaining sample budget: {budget.remaining_sample_budget} / {budget.total_sample_budget} samples. "
        if budget.remaining_run_budget is not None:
            info += f"Remaining run budget: {budget.remaining_run_budget} / {budget.total_run_budget} runs."
        return info

    @is_tool
    async def evaluate_commit(
        self,
        commit: str,
        dataset_id: str,
        split: str,
        sample_ids: list[int] | None = None,
        num_samples: int | None = None,
    ) -> str:
        """Evaluate a version of the codebase specified by a Git commit on a subset of a dataset.
        Use num_samples to evaluate the first N samples, or sample_ids for specific samples.
        If both are None, the full split is evaluated.

        Args:
            commit: The Git commit to evaluate.
            dataset_id: The id of the dataset to evaluate on.
            split: The split of the dataset to evaluate on.
            sample_ids: Specific sample ids to evaluate. Cannot be used with num_samples.
            num_samples: Evaluate the first N samples. Cannot be used with sample_ids.

        Returns:
            A string containing the results of the evaluation.
        """

        # Validate that only one of sample_ids or num_samples is provided
        if sample_ids is not None and num_samples is not None:
            raise ValueError(
                "Cannot specify both sample_ids and num_samples. "
                "Use sample_ids for specific samples, or num_samples for the first N samples."
            )

        # If number of samples is provided, sample the appropriate number of samples
        if num_samples is not None:
            sample_ids = self._get_samples_from_split(dataset_id, split, num_samples)

        # Count the number of samples that will be decremented from the budget
        requested_num_samples = self._validate_and_count_samples(
            dataset_id, split, sample_ids
        )

        # Check that the budget allows for the requested number of samples
        self._check_budget(dataset_id, split, requested_num_samples)

        # Evaluate the commit
        try:
            experiment = await self._evaluate_commit(
                commit=commit,
                dataset_id=dataset_id,
                split=split,
                sample_ids=sample_ids,
            )
        except Exception as e:
            raise e
        finally:
            # Update the budget regardless of whether the experiment was successful or not
            update_info = self._update_budget(dataset_id, split, requested_num_samples)

        # Construct the message for the llm
        message = f"Experiment ID {experiment.id} completed with status {experiment.result.status}. "
        experiment_summary_json = experiment.as_pandas_series().to_json(indent=2)
        return f"{message}{update_info}\n```json\n{experiment_summary_json}\n```"

    @is_tool
    async def evaluate_commit_on_all_splits(
        self,
        commit: str,
        dataset_id: str,
    ) -> list[str]:
        """Evaluate a version of the codebase specified by a Git commit on all accessible splits of a dataset.

        Args:
            commit: The Git commit to evaluate.
            dataset_id: The id of the dataset to evaluate on.

        Returns:
            A list of strings containing the results of the evaluation on each split.
        """

        accessible_splits = [
            split
            for (split, ds_id) in self._budget_ledger.status().keys()
            if ds_id == dataset_id
        ]

        logger.info(
            f"Evaluating commit {commit} on dataset {dataset_id} with accessible splits: {accessible_splits}"
        )

        if not accessible_splits:
            raise ValueError(
                f"No splits found for dataset {dataset_id}. Ensure the dataset_id is correct."
            )

        total_requested_num_samples = 0

        results = {}

        for split in accessible_splits:
            full_split_size = self._validate_and_count_samples(dataset_id, split)
            budget = self._budget_ledger.get(dataset_id, split)

            # Cap samples to remaining budget if needed
            requested_num_samples = full_split_size
            sample_ids = None
            if budget and budget.remaining_sample_budget is not None:
                requested_num_samples = min(
                    full_split_size, budget.remaining_sample_budget
                )
                sample_ids = self._get_samples_from_split(
                    dataset_id, split, requested_num_samples
                )

            logger.info(
                f"Validating budget for split {split} with {requested_num_samples} samples"
            )

            try:
                self._check_budget(dataset_id, split, requested_num_samples)
            except ExperimentBudgetExceeded as e:
                results[split] = e
                continue

            logger.info(
                f"Evaluating commit {commit} on split {split} with {requested_num_samples} samples"
            )

            try:
                results[split] = await self._evaluate_commit(
                    commit=commit,
                    dataset_id=dataset_id,
                    split=split,
                    sample_ids=sample_ids,
                )
            except Exception as e:
                results[split] = e
                continue
            finally:
                self._update_budget(dataset_id, split, requested_num_samples)

            total_requested_num_samples += requested_num_samples

        if all(isinstance(result, Exception) for result in results.values()):
            raise ValueError(
                f"Failed to evaluate commit {commit} on all splits of dataset {dataset_id}. Errors: {results}"
            )

        message = ""

        for split in results:
            message += f"# Result for split {split}\n"

            if isinstance(results[split], Experiment):
                message += f"Experiment ID {results[split].id} completed with status {results[split].result.status}. \n"
                experiment_summary_json = (
                    results[split].as_pandas_series().to_json(indent=2)
                )
                message += f"```json\n{experiment_summary_json}\n```"
            else:
                message += f"Error: {results[split]}"

        return message
