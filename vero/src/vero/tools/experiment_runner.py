from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn

from vero.core.db.database import Experiment, ExperimentDatabase
from vero.core.evaluation import BaseEvaluationParameters
from vero.evaluator import Evaluator
from vero.exceptions import (
    ExperimentBudgetExceeded,
    ExperimentRunFailedError,
    InvalidSplitError,
)
from vero.tools.utils import is_tool

logger = logging.getLogger(__name__)


def _default_on_fatal(msg: str) -> NoReturn:
    raise RuntimeError(msg)


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
    _budget_map: dict[tuple[str, str], SplitBudget] = field(
        default_factory=dict, repr=False
    )
    _canonical_ledger: Any = field(default=None, repr=False)
    _canonical_backend_id: str | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.split_budgets:
            self._budget_map = {
                (sb.split, sb.dataset_id): sb for sb in self.split_budgets
            }

    def bind(self, session) -> None:
        from copy import deepcopy

        self.evaluator = session.evaluator
        self.split_budgets = deepcopy(session.budget)
        self.db = session.db
        self._session_id = session.session_id
        self._vero_home = session.vero_home
        self.run_constraints = session.evaluation_parameters
        self._task = session.task
        self._budget_map = {(sb.split, sb.dataset_id): sb for sb in self.split_budgets}
        engine = getattr(session, "evaluation_engine", None)
        self._canonical_ledger = (
            engine.budget_ledger if engine is not None else None
        )
        self._canonical_backend_id = getattr(
            session,
            "evaluation_backend_id",
            None,
        )

    def _canonical_budget(self, dataset_id: str, split: str):
        if self._canonical_ledger is None or self._canonical_backend_id is None:
            return None
        from vero.evaluation import EvaluationSet

        evaluation_set = EvaluationSet(name=dataset_id, partition=split)
        return self._canonical_ledger.get(
            self._canonical_backend_id,
            evaluation_set,
        )

    def _get_dataset_info(self, dataset_id: str):
        """Get dataset info from the store."""
        from vero.core.dataset import DatasetInfo
        from vero.core.dataset.store import load_dataset

        sessions_dir = self._vero_home / "sessions" if self._vero_home else None
        dataset_cache = self._vero_home / "datasets" if self._vero_home else None
        dataset = load_dataset(sessions_dir, dataset_cache, self._session_id, dataset_id)
        return DatasetInfo(
            id=dataset_id,
            splits={split: len(dataset[split]) for split in dataset},
            features={split: list(dataset[split].features) for split in dataset},
        )

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
        """Get a list of sample ids from a split. If num_samples is greater than or equal to the size of the split, return None."""
        dataset_info = self._get_dataset_info(dataset_id)
        split_size = dataset_info.splits[split]
        num_samples = min(num_samples, split_size)

        if num_samples >= split_size:
            return None

        sample_ids = list(range(num_samples))
        return sample_ids

    def _validate_and_count_samples(
        self, dataset_id: str, split: str, sample_ids: list[int] | None = None
    ) -> int:
        """Validate and count the number of samples in a split. If sample_ids is None, return the size of the split."""

        dataset_info = self._get_dataset_info(dataset_id)
        split_size = dataset_info.splits[split]

        # If None, the full split is being evaluated
        if sample_ids is None:
            return split_size

        # Validate that the sample ids are within the range of the split
        invalid_sample_ids = []
        for sample_id in sample_ids:
            if sample_id < 0 or sample_id >= split_size:
                invalid_sample_ids.append(sample_id)

        if len(invalid_sample_ids) > 0:
            raise ValueError(
                f"The provided sample ids are outside the range of the split [0, {split_size - 1}]: {invalid_sample_ids}"
            )

        return len(sample_ids)

    def _validate_split_access(self, dataset_id: str, split: str) -> None:
        """Validate that the split and dataset combination is allowed."""

        if (split, dataset_id) not in self._budget_map:
            allowed_keys = list(self._budget_map.keys())
            raise InvalidSplitError(
                f"No split budget found for the combination (dataset_id={dataset_id}, split={split}) either because it does not exist or because it is not allowed. Allowed combinations: {allowed_keys}"
            )

    def _check_budget(
        self, dataset_id: str, split: str, requested_num_samples: int
    ) -> str:
        """Check that the budget allows for the requested number of samples."""

        # Check if this split and dataset combination is allowed
        self._validate_split_access(dataset_id, split)
        canonical = self._canonical_budget(dataset_id, split)
        if self._canonical_ledger is not None:
            if canonical is None:
                raise InvalidSplitError(
                    f"No canonical budget found for dataset_id={dataset_id}, split={split}"
                )
            if canonical.remaining_runs is not None and canonical.remaining_runs <= 0:
                raise ExperimentBudgetExceeded(
                    f"No runs left for the {split} split of the {dataset_id} dataset."
                )
            if (
                canonical.remaining_cases is not None
                and canonical.remaining_cases < requested_num_samples
            ):
                raise ExperimentBudgetExceeded(
                    f"Requested {requested_num_samples} samples for the {split} split "
                    f"of the {dataset_id} dataset, but the remaining sample budget only "
                    f"allows for {canonical.remaining_cases} samples."
                )
            if (
                canonical.max_cases_per_run is not None
                and requested_num_samples > canonical.max_cases_per_run
            ):
                raise ExperimentBudgetExceeded(
                    f"Requested {requested_num_samples} samples for the {split} split "
                    f"of the {dataset_id} dataset, but only "
                    f"{canonical.max_cases_per_run} are allowed per run."
                )
            return
        budget = self._budget_map[(split, dataset_id)]

        # Determine if we have enough runs left
        if not budget.has_run_budget():
            raise ExperimentBudgetExceeded(
                f"No runs left for the {split} split of the {dataset_id} dataset."
            )

        # Check against remaining sample budget
        if not budget.has_sample_budget(requested_num_samples):
            raise ExperimentBudgetExceeded(
                f"Requested {requested_num_samples} samples for the {split} split of the {dataset_id} dataset, but the remaining sample budget only allows for {budget.remaining_sample_budget} samples."
            )

        # Check against max samples per run constraint
        if budget.exceeds_per_run_budget(requested_num_samples):
            raise ExperimentBudgetExceeded(
                f"Requested {requested_num_samples} samples for the {split} split of the {dataset_id} dataset, but only {budget.max_samples_per_run} are allowed per run."
            )

    def _update_budget(self, dataset_id: str, split: str, num_samples: int) -> str:
        """Update the remaining budget for a given dataset and split and return a message about the update."""

        self._validate_split_access(dataset_id, split)
        canonical = self._canonical_budget(dataset_id, split)
        if self._canonical_ledger is not None:
            if canonical is None:
                return ""
            parts = []
            if canonical.total_cases is not None:
                parts.append(
                    f"Used {num_samples} samples from the total "
                    f"{canonical.total_cases} sample budget. Remaining sample budget: "
                    f"{canonical.remaining_cases}."
                )
            if canonical.total_runs is not None:
                parts.append(
                    f"Used 1 run from the total {canonical.total_runs} run budget. "
                    f"Remaining runs: {canonical.remaining_runs}"
                )
            return " ".join(parts)
        budget = self._budget_map[(split, dataset_id)]

        info = ""

        # Update the remaining budget
        budget.decrement_sample_budget(num_samples)
        if budget.total_sample_budget is not None:
            info += f"Used {num_samples} samples from the total {budget.total_sample_budget} sample budget. Remaining sample budget: {budget.remaining_sample_budget}. "

        # Update the remaining runs
        budget.decrement_run_budget()
        if budget.remaining_run_budget is not None:
            info += f"Used 1 run from the total {budget.total_run_budget} run budget. Remaining runs: {budget.remaining_run_budget}"

        return info

    async def _evaluate_commit(
        self,
        commit: str,
        dataset_id: str,
        split: str,
        sample_ids: list[int] | None = None,
        add_to_db: bool = True,
    ) -> Experiment:
        """Evaluate a version of the codebase specified by a Git commit on a subset of a dataset."""

        try:
            kwargs = {}
            if self._canonical_ledger is not None:
                kwargs["meter_budget"] = True
                kwargs["add_to_compatibility_db"] = add_to_db
            return await self.evaluator.evaluate(
                commit=commit,
                dataset_id=dataset_id,
                split=split,
                task=self._task,
                sample_ids=sample_ids,
                db=self.db if add_to_db else None,
                evaluation_parameters=self.run_constraints,
                **kwargs,
            )
        except Exception as e:
            from vero.evaluation import EvaluationBudgetExceeded

            if isinstance(e, EvaluationBudgetExceeded):
                raise ExperimentBudgetExceeded(str(e)) from e
            if isinstance(e, ExperimentRunFailedError) and e.returncode >= 3:
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
        self._validate_split_access(dataset_id, split)
        canonical = self._canonical_budget(dataset_id, split)
        if self._canonical_ledger is not None:
            if canonical is None:
                return "No evaluation budget is configured."
            info = ""
            if canonical.total_cases is not None:
                info += (
                    f"Remaining sample budget: {canonical.remaining_cases} / "
                    f"{canonical.total_cases} samples. "
                )
            if canonical.total_runs is not None:
                info += (
                    f"Remaining run budget: {canonical.remaining_runs} / "
                    f"{canonical.total_runs} runs."
                )
            return info
        budget = self._budget_map[(split, dataset_id)]

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
                add_to_db=True,
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
            split for (split, ds_id) in self._budget_map.keys() if ds_id == dataset_id
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
            budget = self._budget_map.get((split, dataset_id))

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
                    add_to_db=True,
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
