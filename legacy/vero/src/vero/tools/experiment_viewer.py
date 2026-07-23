from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from vero.core.dataset import (
    DefaultSplitNames,
    get_non_viewable_splits,
)
from vero.core.db.database import Experiment, ExperimentDatabase
from vero.core.db.result import SampleResult
from vero.tools.utils import is_tool
from vero.tools.utils.pandas import query_and_order_df
from vero.utils import df_to_format

if TYPE_CHECKING:
    import pandas as pd



@dataclass
class ExperimentViewer:
    """View results and statistics of experiments."""

    exclude_tools: list[str] = field(default_factory=list)

    # Runtime fields — set during bind()
    db: ExperimentDatabase | None = None
    exclude_splits: list[str] = field(default_factory=list)

    def bind(self, session) -> None:
        self.db = session.db
        if session.split_accesses:
            self.exclude_splits = get_non_viewable_splits(session.split_accesses)

        assert isinstance(self.db, ExperimentDatabase), "db must be an ExperimentDatabase"
        assert isinstance(self.exclude_splits, list), "exclude_splits must be a list"

    def experiments(self, splits: list[str] | None = None) -> list[Experiment]:
        """Get experiments by splits. If splits are provided, only experiments in the splits are returned."""

        if splits:
            disallowed = [split for split in splits if split in (self.exclude_splits or [])]
            if disallowed:
                raise ValueError(f"You do not have permission to view these splits: {disallowed}")

        def filter_fn(experiment: Experiment) -> bool:
            split = experiment.run.dataset_subset.split
            if split in self.exclude_splits:
                return False
            if splits is not None:
                return split in splits
            else:
                return True

        return self.db.get_experiments(filter_fn=filter_fn)

    def df(self, splits: list[str] | None = None) -> "pd.DataFrame":
        from vero.core.constants import default_minimum_score

        # TODO: fill_score should come from the task definition (score range
        # is task-specific, not always 0-based). For now, use the global
        # default so errors are penalized and the agent sees their cost.
        return self.db.get_experiments_df(self.experiments(splits), fill_score=default_minimum_score)

    @classmethod
    def load_from_file(cls, path_to_experiments_db_json: Path | str) -> "ExperimentViewer":
        """Load an ExperimentViewer from a file."""
        path_to_experiments_db_json = Path(path_to_experiments_db_json).resolve()
        if not path_to_experiments_db_json.exists():
            raise FileNotFoundError(f"Path {path_to_experiments_db_json} does not exist")
        db = ExperimentDatabase.load_from_file(path_to_experiments_db_json)
        return cls(db=db)

    @is_tool
    def readme(self) -> str:
        """Readme for the experiment viewer tool."""
        non_viewable = self.exclude_splits or []
        non_viewable_str = ", ".join(f'"{s}"' for s in non_viewable) if non_viewable else "none"

        return f"""# ExperimentViewer

## Workflow

1. `view_experiment_table(split="train")` → browse experiments, find `experiment_id`
2. `view_sample_results_table(experiment_id="...")` → browse sample results, find `sample_id` values
3. `view_sample_result_trace(experiment_id="...", sample_id=42)` → debug specific sample execution

## Splits

Typical splits are: `train`, `validation`, `test`.
You CANNOT view details of the following splits: {non_viewable_str}
(Note: You can run experiments on non-viewable splits and see summary stats, but cannot inspect their results)

## Key Concepts

- **experiment_id**: String identifier for an experiment (get from `id` column in experiment table)
- **sample_id**: Integer key for a sample in the dataset (get from `sample_id` column in sample results table)

## Common Mistakes

- Using row index instead of `sample_id` — always get `sample_id` from the table, don't assume it's sequential
- Passing `split` to sample results methods — sample results methods take `experiment_id`, not `split`
- Trying to view non-viewable splits — will raise an error
- Confusing the candidate commit with the experiment id; the candidate commit is a Git commit hash, while the experiment id is a unique identifier for an experiment.

## Concept Hierarchy

```
Experiment
├── id: str (unique experiment identifier)
├── ExperimentRun
│   ├── Candidate (commit, repo_name, parent_commit)
│   └── DatasetSubset (dataset_id, split, sample_ids)
└── ExperimentResult
    ├── status: SUCCESS/FAILED
    └── sample_results: dict[sample_id → SampleResult]
        └── SampleResult (score, feedback, error, execution_trace)
```

## Score Statistics

When viewing experiment tables, these score columns are available:

- **mean_score**: Mean of successful samples, NaNs from errorsfilled with a fill_score (default 0.0).
- **mean_score_optimistic**: NaNs from errors filled with max score (1.0). An optimistic score that gives the benefit of the doubt to errors.
- **mean_score_pessimistic**: NaNs from errors filled with min score (0.0). A pessimistic score that penalizes errors.
- **error_rate**: Fraction of samples that errored/are NaN.
- **error_count**: Number of samples that errored/are NaN.
- **bootstrap_lower_confidence_interval / bootstrap_upper_confidence_interval**: 95% confidence interval bounds for the mean score.
"""

    @is_tool
    def get_experiment_table_metadata(self) -> str:
        """
        Get metadata about the experiment table, i.e. its shape and column names.

        Returns:
            A string containing the metadata about the experiment table
        """
        df = self.df()

        if len(df.columns) == 0 or len(df) == 0:
            return "The experiment table for this split is empty."

        split_info = df["dataset_subset_split"].value_counts().to_dict()
        return f"The experiment table has {len(df)} rows (splits: {split_info}) and {len(df.columns)} columns. The column names are: {list(df.columns)}."

    def _get_experiment(self, experiment_id: str) -> Experiment:
        """Helper to get an experiment by its unique ID.

        Args:
            experiment_id: The unique ID of the experiment

        Returns:
            The Experiment object

        Raises:
            KeyError: If experiment not found or split is excluded
        """
        # Search across all experiments in the database
        all_experiments = self.db.get_experiments()

        for experiment in all_experiments:
            if experiment.id == experiment_id:
                # Check if the split is viewable
                split = experiment.run.dataset_subset.split
                if self.exclude_splits and split in self.exclude_splits:
                    raise KeyError(
                        f"Experiment '{experiment_id}' is in the '{split}' split which is excluded from viewing."
                    )
                return experiment

        available_ids = [e.id for e in self.experiments()]
        raise KeyError(f"Experiment ID '{experiment_id}' not found. Available IDs: {available_ids}")

    @is_tool
    def view_experiment_table(
        self,
        split: str = DefaultSplitNames.train,
        num_rows: int | None = 5,
        row_offset_idx: int = 0,
        columns: list[str] | None = None,
        query: str | None = None,
        sort_values_by: str | None = None,
        ascending: bool = True,
        format: Literal["csv", "json", "yaml", "kv_markdown"] = "kv_markdown",
    ) -> str:
        """
        View the experiments table of experiments, where each row represents an experiment.
        Columns contain statistics and metadata about each experiment, e.g. the number of samples evaluates,
        the average score, the error rate, etc.
        The table is sorted by the experiment index.
        Note that num_rows and row_offset_idx are applied after the query and sort_values_by operations.

        Args:
            split: The split to view the experiment table for
            num_rows: Maximum number of rows to return (optional)
            row_offset_idx: Number of rows to skip (default 0)
            columns: List of columns to include. Leave empty to view all columns.
            query: A query string to filter the dataframe. Example Usage: "dataset_subset_dataset_id == 'math' and dataset_subset_split == 'train'" (optional)
            sort_values_by: Column name to order by (optional)
            ascending: Whether to sort in ascending order (default True)
            format: Output format. Recommended format is "kv_markdown" for readability. (default "kv_markdown")

        Returns:
            Filtered and ordered experiment data in the specified format
        """
        df = self.df(splits=[split])

        if query or sort_values_by:
            df = query_and_order_df(df, query, sort_values_by, ascending)

        before_pagination_len = len(df)

        if row_offset_idx > 0:
            df = df.iloc[row_offset_idx:]

        if num_rows is not None:
            df = df.iloc[:num_rows]

        after_pagination_len = len(df)

        if columns is not None:

            valid_columns = [col for col in columns if col in df.columns]
            if len(valid_columns) == 0:
                raise ValueError(
                    f"Invalid column names: {columns}. Valid column names are: {list(df.columns)}."
                )

            df = df[valid_columns]

        format_kwargs = {}
        if format == "kv_markdown":
            format_kwargs["record_prefix"] = "Experiment"

        df_str = df_to_format(df, format, **format_kwargs)

        if format in ["csv", "json", "yaml"]:
            df_str = f"```{format}\n{df_str}\n```"
        else:
            df_str = f"```{df_str}\n```"

        return f"Found {before_pagination_len} experiment(s) before pagination. Viewing {after_pagination_len} experiment(s) starting at row {row_offset_idx}.\n{df_str}"

    @is_tool
    def get_sample_results_table_metadata(self, experiment_id: str) -> str:
        """
        Get metadata about the sample results table, i.e. its shape and column names.

        Args:
            experiment_id: The unique ID of the experiment (from the 'id' column in experiment table)

        Returns:
            A string containing the metadata about the sample results table
        """
        experiment = self._get_experiment(experiment_id)

        result = experiment.result
        df = result.sample_results_df(exclude=["execution_trace"])

        if len(df.columns) == 0 or len(df) == 0:
            return "The sample results table for this experiment is empty."

        return f"The sample results table has {len(df)} rows and {len(df.columns)} columns. The column names are: {list(df.columns)}."

    @is_tool
    def view_sample_results_table(
        self,
        experiment_id: str,
        num_rows: int | None = 5,
        row_offset_idx: int = 0,
        columns: list[str] | None = None,
        query: str | None = None,
        sort_values_by: str | None = None,
        ascending: bool = True,
        format: Literal["csv", "json", "yaml", "kv_markdown"] = "kv_markdown",
    ) -> str:
        """
        View scores, errors, and score feedback of a particular experiment.
        Each row represents a data sample evaluated in the experiment. Columns contains details about the sample, e.g. the id,
        the score, the error, the feedback, etc.
        Note that num_rows and row_offset_idx are applied after the query and sort_values_by operations.

        Args:
            experiment_id: The unique ID of the experiment (from the 'id' column in experiment table)
            num_rows: Maximum number of rows to return (optional)
            row_offset_idx: Number of rows to skip (default 0)
            columns: List of columns to include. Leave empty to view all columns.
            query: A query string to filter the dataframe. Example Usage: "dataset_subset_dataset_id == 'math' and dataset_subset_split == 'train'" (optional)
            sort_values_by: Column name to order by (optional)
            ascending: Whether to sort in ascending order (default True)
            format: Output format. Recommended format is "kv_markdown" for readability. (default "kv_markdown")

        Returns:
            Filtered and ordered summaries of sample results in the specified format
        """
        experiment = self._get_experiment(experiment_id)

        result = experiment.result
        df = result.sample_results_df(exclude=["execution_trace"])

        if query or sort_values_by:
            try:
                df = query_and_order_df(df, query, sort_values_by, ascending)
            except Exception as e:
                raise ValueError(f"Failed to query and order the dataframe: {e}.")

        before_pagination_len = len(df)

        if row_offset_idx > 0:
            df = df.iloc[row_offset_idx:]
        if num_rows is not None:
            df = df.iloc[:num_rows]

        after_pagination_len = len(df)

        if columns is not None:
            valid_columns = [col for col in columns if col in df.columns]
            if len(valid_columns) == 0:
                raise ValueError(
                    f"Invalid column names: {columns}. Valid column names are: {list(df.columns)}."
                )
            df = df[valid_columns]

        format_kwargs = {}
        if format == "kv_markdown":
            format_kwargs["record_prefix"] = "Sample Result"

        df_str = df_to_format(df, format, **format_kwargs)

        if format in ["csv", "json", "yaml"]:
            df_str = f"```{format}\n{df_str}\n```"
        else:
            df_str = f"```{df_str}\n```"

        return f"Found {before_pagination_len} sample result(s) before pagination. Viewing {after_pagination_len} sample result(s) starting at row {row_offset_idx}. \n{df_str}"

    def _get_sample_result(
        self, experiment_id: str, sample_id: int
    ) -> tuple[Experiment, SampleResult]:
        """Helper to get a sample result by experiment ID and sample_id.

        Args:
            experiment_id: The unique ID of the experiment
            sample_id: The dataset sample_id (index in the original dataset)

        Returns:
            Tuple of (Experiment, SampleResult)

        Raises:
            KeyError: If experiment or sample not found
        """
        experiment = self._get_experiment(experiment_id)

        sample_result = experiment.result.get_sample_result(sample_id)
        if sample_result is None:
            available_ids = experiment.result.sample_ids
            if not available_ids:
                raise KeyError("No sample results found. The experiment has no sample results.")
            raise KeyError(
                f"sample_id={sample_id} not found in experiment. Available sample_ids: {available_ids}"
            )

        return experiment, sample_result

    @is_tool
    def view_sample_result_trace(
        self,
        experiment_id: str,
        sample_id: int,
        num_spans: int = 5,
        start_offset: int = 0,
        format: Literal["json", "yaml"] = "json",
    ) -> str:
        """
        View the execution trace of a particular sample from a particular experiment.
        Execution traces are a list of spans. By default we show the first 5 spans.
        Long traces are truncated to 10_000 characters. Use the `start_offset` to view
        them in a paginated manner.

        Args:
            experiment_id: The unique ID of the experiment (from the 'id' column in experiment table)
            sample_id: The dataset sample_id (index in the original dataset)
            num_spans: The number of spans from the trace to include
            start_offset: The number of spans from the trace to skip
            format: The format to return the sample result in

        Returns:
            A JSON/YAML string containing the details of the sample result

        """
        _, sample_result = self._get_sample_result(experiment_id, sample_id)
        sample_result_dict = sample_result.model_dump()

        def _dump_obj(obj: dict) -> str:
            return (
                json.dumps(obj, indent=2)
                if format == "json"
                else yaml.dump(obj, indent=2, sort_keys=False, allow_unicode=True)
            )

        info = f"Viewing sample_id={sample_id} from experiment '{experiment_id}'. "

        execution_trace = sample_result_dict.get("execution_trace", []) or []
        num_spans_before = len(execution_trace)

        if num_spans_before == 0:
            return f"{info}\n```{format}\n{_dump_obj(sample_result_dict)}\n```"

        char_count = 0
        truncated_trace = []
        truncated = False
        end_offset = min(start_offset + num_spans, len(execution_trace))

        for span in execution_trace[start_offset:end_offset]:
            span_str = _dump_obj(span)
            if char_count + len(span_str) > 10000 and len(truncated_trace) > 0:
                truncated = True
                break
            char_count += len(span_str)
            truncated_trace.append(span)

        num_spans_after = len(truncated_trace)

        info = f"{info}Showing spans {start_offset} to {start_offset + num_spans_after} of {num_spans_before} total spans. "

        if truncated:
            info = f"{info}Requested spans did not fit in the 10,000 character limit. "

        return f"{info}\n```{format}\n{_dump_obj(truncated_trace)}\n```"

    @is_tool
    def get_trace_summary(
        self,
        experiment_id: str,
        sample_id: int,
    ) -> str:
        """
        Get a summary of the execution trace for a sample result. Useful for understanding
        the structure of a trace before drilling into specific spans.

        Args:
            experiment_id: The unique ID of the experiment (from the 'id' column in experiment table)
            sample_id: The dataset sample_id (index in the original dataset split)

        Returns:
            A summary of the trace including span count, types, and keys present
        """
        _, sample_result = self._get_sample_result(experiment_id, sample_id)
        sample_result_dict = sample_result.model_dump()
        execution_trace = sample_result_dict.get("execution_trace", []) or []

        if not execution_trace:
            return f"No execution trace for sample_id={sample_id} in experiment '{experiment_id}'."

        # Analyze span types, keys, and char counts
        type_counts: dict[str, int] = {}
        type_chars: dict[str, int] = {}
        total_chars = 0

        for span in execution_trace:
            span_chars = len(json.dumps(span))
            total_chars += span_chars

            if isinstance(span, dict):
                span_type = "dict"
                keys_str = ",".join(sorted(span.keys()))
            elif isinstance(span, list):
                span_type = "list"
                keys_str = f"len={len(span)}"
            elif isinstance(span, str):
                span_type = "str"
                keys_str = ""
            else:
                span_type = type(span).__name__
                keys_str = ""

            type_key = f"{span_type}({keys_str})" if keys_str else span_type
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            type_chars[type_key] = type_chars.get(type_key, 0) + span_chars

        summary = {
            "num_spans": len(execution_trace),
            "total_chars": total_chars,
            "span_types": {
                k: {"count": type_counts[k], "chars": type_chars[k]} for k in type_counts
            },
        }

        return f"```json\n{json.dumps(summary, indent=2)}\n```"

    @is_tool
    def view_sample_result_span(
        self,
        experiment_id: str,
        sample_id: int,
        span_idx: int,
        char_offset: int = 0,
        char_limit: int = 100_000,
        format: Literal["json", "yaml"] = "json",
    ) -> str:
        """
        View a particular span of the execution trace of a particular sample from a particular experiment.

        Args:
            experiment_id: The unique ID of the experiment (from the 'id' column in experiment table)
            sample_id: The dataset sample_id (index in the original dataset split)
            span_idx: The index of the span to view
            char_offset: The number of characters to skip from the start of the span
            char_limit: The number of characters to limit the span to
            format: The format to return the span in

        Returns:
            A JSON/YAML string containing the details of the span

        """
        _, sample_result = self._get_sample_result(experiment_id, sample_id)
        sample_result_dict = sample_result.model_dump()

        execution_trace = sample_result_dict.get("execution_trace", []) or []
        span = execution_trace[span_idx]

        def _dump_obj(obj: dict) -> str:
            return (
                json.dumps(obj, indent=2)
                if format == "json"
                else yaml.dump(obj, indent=2, sort_keys=False, allow_unicode=True)
            )

        span = _dump_obj(span)

        span_str = span[char_offset : char_offset + char_limit]

        if len(span) > len(span_str):
            span_str = f"{span_str}...<TOOL CALL TRUNCATED TO {char_limit} CHARACTERS. USE CHAR_OFFSET TO VIEW THE REST.>"

        if char_offset > 0:
            span_str = f"<OFFSET {char_offset}>...{span_str}"

        return f"Viewing characters {char_offset} to {char_offset + char_limit} of the span at index {span_idx} from the execution trace of sample_id={sample_id} from experiment '{experiment_id}'. \n\n```{format}\n{span_str}\n```"
