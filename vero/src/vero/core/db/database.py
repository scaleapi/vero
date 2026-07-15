from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

from vero.core.constants import default_minimum_score
from vero.core.dataset import DatasetInfo
from vero.core.db.candidate import Candidate
from vero.core.db.result import ExperimentResult
from vero.core.db.run import ExperimentRun

if TYPE_CHECKING:
    from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class Experiment(BaseModel):
    """Deprecated dataset view of a canonical ``EvaluationRecord``.

    New evaluation and optimization code should use ``vero.EvaluationRecord``.
    This model remains for VeroTask extensions and existing callers.
    """

    run: ExperimentRun
    result: ExperimentResult

    @property
    def id(self) -> str:
        return self.result.id

    def as_pandas_series(
        self, nan_score_fill_value: float | None = default_minimum_score
    ) -> Series:
        """Return the experiment in a pandas representation with basic information about the run and sample results statistics."""
        import pandas as pd

        run_series: Series = self.run.as_pandas_series()
        result_series: Series = self.result.sample_results_statistics(
            nan_score_fill_value=nan_score_fill_value
        )

        series = pd.concat(
            [
                run_series,
                result_series,
            ]
        )
        series["id"] = self.id
        return series

    def summary(self, key_prefix: str | None = None) -> dict:

        if key_prefix is None:
            split = self.run.dataset_subset.split
            key_prefix = split

        lower, upper = self.result.confidence_interval()
        return {
            f"{key_prefix}/score": self.result.score(),
            f"{key_prefix}/error_rate": self.result.error_rate(),
            f"{key_prefix}/candidate_commit": self.run.candidate.commit,
            f"{key_prefix}/num_samples": len(self.result.sample_results),
            f"{key_prefix}/lower_confidence_interval": lower,
            f"{key_prefix}/upper_confidence_interval": upper,
        }


@dataclass
class ExperimentDatabase:
    """Deprecated schema-v1 dataset database.

    Runtime state is held in ``vero.EvaluationDatabase``. This class remains a
    compatibility serialization and projection target.
    """

    id: str
    candidates: dict[tuple[str, ...], Candidate] = field(default_factory=dict)
    runs: dict[str, ExperimentRun] = field(default_factory=dict)
    results: dict[str, ExperimentResult] = field(default_factory=dict)
    datasets: dict[str, DatasetInfo] = field(default_factory=dict)
    listeners: list[Callable[[Experiment], None]] = field(
        default_factory=list, repr=False
    )

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"ExperimentDatabase(id={self.id}, candidates={len(self.candidates)}, runs={len(self.runs)}, results={len(self.results)})"

    def get_candidate(self, candidate: Candidate | tuple[str, str]) -> Candidate | None:
        if isinstance(candidate, Candidate):
            candidate = candidate.id
        return self.candidates.get(candidate)

    def get_run(self, run: ExperimentRun | str) -> ExperimentRun | None:
        if isinstance(run, ExperimentRun):
            run = run.id
        return self.runs.get(run)

    def get_result(self, result: ExperimentResult | str) -> ExperimentResult | None:
        if isinstance(result, ExperimentResult):
            result = result.id
        return self.results.get(result)

    def add_candidate(self, candidate: Candidate):
        if not self.get_candidate(candidate):
            self.candidates[candidate.id] = candidate

    def add_run(self, run: ExperimentRun):
        if not self.get_run(run):
            self.runs[run.id] = run

    def add_result(self, result: ExperimentResult):
        if not self.get_result(result):
            if self.get_run(result.run_id) is not None:
                self.results[result.id] = result
            else:
                raise ValueError(f"ExperimentRun {result.run_id} does not exist!")

    def add_experiment(self, experiment: Experiment):
        self.add_candidate(experiment.run.candidate)
        self.add_run(experiment.run)
        self.add_result(experiment.result)
        for listener in self.listeners:
            listener(experiment)

    def get_experiment(self, result_id: str) -> Experiment:
        result = self.results[result_id]
        run = self.runs[result.run_id]
        return Experiment(run=run, result=result)

    def get_experiments_df(
        self,
        experiments: list[Experiment] | None = None,
        fill_score: float | None = default_minimum_score,
    ) -> "DataFrame":
        import pandas as pd

        if experiments is None:
            experiments = self.get_experiments()
        df = pd.DataFrame(
            [
                experiment.as_pandas_series(nan_score_fill_value=fill_score)
                for experiment in experiments
            ]
        )
        if "id" in df.columns:
            df.set_index("id", inplace=True)

        return df

    def get_experiments(
        self,
        result_ids: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        reverse: bool = False,
        filter_fn: Callable[[Experiment], bool] = lambda _: True,
        sort_key: Callable[[Experiment], Any] | None = None,
    ) -> list[Experiment]:
        """Get experiments by result ids. By default, the list of experiments is sorted in creation order.

        Args:
            result_ids: List of result ids
            limit: Number of experiments to return
            offset: Number of experiments to skip
            reverse: Whether to reverse the list of experiments
            filter_fn: A function that filters the list of experiments
            sort_key: A function that maps an experiment to a value that can be used to order a list

        Returns:
            List of completed (i.e. with results) experiments
        """
        result_ids = result_ids or list(self.results.keys())
        experiments = list(filter(filter_fn, map(self.get_experiment, result_ids)))

        if sort_key is not None:
            experiments.sort(key=sort_key, reverse=reverse)
        elif reverse:
            experiments = experiments[::-1]

        if offset is not None:
            experiments = experiments[offset:]
        if limit is not None:
            experiments = experiments[:limit]
        return experiments

    @classmethod
    def serialize_candidate_id(cls, candidate_id: tuple[str, ...]) -> str:
        return "|".join(candidate_id)

    @classmethod
    def deserialize_candidate_id(cls, candidate_id: str) -> tuple[str, ...]:
        return tuple(candidate_id.split("|"))

    def serialize(self, **model_dump_kwargs: Any) -> dict[str, Any]:
        """Serialize the database to a dictionary."""
        return {
            "id": self.id,
            "candidates": {
                self.serialize_candidate_id(k): v.model_dump(**model_dump_kwargs)
                for k, v in self.candidates.items()
            },
            "runs": {
                k: v.model_dump(**model_dump_kwargs) for k, v in self.runs.items()
            },
            "results": {
                k: v.model_dump(**model_dump_kwargs) for k, v in self.results.items()
            },
            "datasets": {
                k: v.model_dump(**model_dump_kwargs) for k, v in self.datasets.items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ExperimentDatabase:
        """Deserialize a dictionary to an ExperimentDatabase."""
        db = cls(id=data["id"])

        for k, v in data.get("candidates", {}).items():
            db.candidates[cls.deserialize_candidate_id(k)] = Candidate.model_validate(v)

        for k, v in data.get("runs", {}).items():
            db.runs[k] = ExperimentRun.model_validate(v)

        for k, v in data.get("results", {}).items():
            db.results[k] = ExperimentResult.model_validate(v)

        for k, v in data.get("datasets", {}).items():
            db.datasets[k] = DatasetInfo.model_validate(v)

        return db

    def to_json(self) -> str:
        """Convert the database to a JSON string."""
        return json.dumps(self.serialize(), default=str, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentDatabase:
        """Create an ExperimentDatabase from a JSON string."""
        data = json.loads(json_str)
        return cls.deserialize(data)

    def save_to_file(self, file_path: str | Path) -> None:
        """Save the database to a JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> ExperimentDatabase:
        """Load an ExperimentDatabase from a JSON file."""
        file_path = Path(file_path)

        with open(file_path, "r") as f:
            json_str = f.read()

        return cls.from_json(json_str)

    @classmethod
    def from_experiments_dir(
        cls, experiments_dir: Path, db_id: str | None = None
    ) -> ExperimentDatabase:
        """Reconstruct an ExperimentDatabase from the experiments/ directory on disk.

        Each subdirectory should contain:
        - evaluation_parameters.json (Candidate + ExperimentRun)
        - samples/*.json (SampleResult per sample)
        - result_metadata.json (optional: id, run_id, status)

        Args:
            experiments_dir: Path to the experiments/ directory.
            db_id: Optional database ID. Defaults to directory name.

        Returns:
            Reconstructed ExperimentDatabase.
        """
        from vero.core.constants import (
            evaluation_parameters_basename,
            pytest_report_basename,
            result_metadata_basename,
            samples_dir_name,
        )
        from vero.core.db.result import SampleResult
        from vero.core.evaluation import EvaluationParameters

        db = cls(id=db_id or experiments_dir.parent.name)

        if not experiments_dir.exists():
            return db

        for result_dir in sorted(experiments_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            try:
                # Load evaluation parameters (contains Candidate + ExperimentRun)
                params_path = result_dir / evaluation_parameters_basename
                if not params_path.exists():
                    logger.warning(
                        f"Skipping {result_dir.name}: missing {evaluation_parameters_basename}"
                    )
                    continue

                params = EvaluationParameters.model_validate_json(
                    params_path.read_text()
                )
                run = params.run
                result_id = result_dir.name

                # Load sample results
                samples_dir = result_dir / samples_dir_name
                sample_results: dict[int, SampleResult] = {}
                if samples_dir.exists():
                    for sample_path in samples_dir.glob("*.json"):
                        try:
                            sample_id = int(sample_path.stem)
                            sample_results[sample_id] = (
                                SampleResult.model_validate_json(
                                    sample_path.read_text()
                                )
                            )
                        except (ValueError, Exception) as e:
                            logger.warning(
                                f"Skipping corrupt sample {sample_path}: {e}"
                            )

                # Load result metadata (status) if available
                metadata_path = result_dir / result_metadata_basename
                status = None
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    from vero.core.db.result import ExperimentResultStatus

                    status = ExperimentResultStatus(metadata.get("status", "unknown"))

                # Load pytest report if available
                pytest_report = None
                pytest_path = result_dir / pytest_report_basename
                if pytest_path.exists():
                    from vero.core.db.pytest import PyTestReport

                    pytest_report = PyTestReport.model_validate_json(
                        pytest_path.read_text()
                    )

                # Create ExperimentResult
                if status is not None:
                    result = ExperimentResult(
                        id=result_id,
                        run_id=run.id,
                        status=status,
                        sample_results=sample_results,
                        pytest_report=pytest_report,
                    )
                else:
                    # Compute status from error rate
                    result = ExperimentResult.create_with_status(
                        id=result_id,
                        error_rate=params.error_rate_threshold,
                        run_id=run.id,
                        sample_results=sample_results,
                        pytest_report=pytest_report,
                    )

                experiment = Experiment(run=run, result=result)
                db.add_experiment(experiment)

            except Exception as e:
                logger.warning(f"Skipping corrupt experiment {result_dir.name}: {e}")

        return db
