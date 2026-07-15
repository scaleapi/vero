from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from vero.core.db.candidate import Candidate
from vero.core.db.dataset import DatasetSubset

if TYPE_CHECKING:
    from pandas import Series


class ExperimentRun(BaseModel):
    """Deprecated VeroTask protocol request for a dataset subset.

    General evaluation code should use ``EvaluationRequest`` and
    ``EvaluationSet``. This type remains part of the Python task subprocess
    protocol.

    Attributes:
        candidate: The candidate system to evaluate.
        dataset_subset: The dataset subset to evaluate the candidate on.
    """

    candidate: Candidate
    dataset_subset: DatasetSubset

    def id_parts(self) -> tuple[str, str, str, str, str]:
        return (*self.candidate.id, *self.dataset_subset.id)

    @property
    def id(self) -> str:
        return "_".join(self.id_parts())

    def as_pandas_series(self) -> Series:
        import pandas as pd

        return pd.json_normalize(self.model_dump(), sep="_").iloc[0]
