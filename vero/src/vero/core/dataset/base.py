"""Core dataset types: splits, access control, and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, model_validator

from vero.core.utils import is_valid_id


class DefaultSplitNames(StrEnum):
    """Common dataset split names."""

    train = "train"
    validation = "validation"
    test = "test"


class SplitAccessLevel(StrEnum):
    """Access levels for dataset splits.

    Three tiers of increasing restriction:
      - viewable:     rows materialized + full per-sample results visible.
      - non_viewable: no rows, but the split can be evaluated and summary stats seen.
      - no_access:    no rows, no summary, and not agent-evaluable (admin/verifier only).
    """

    viewable = "viewable"
    non_viewable = "non_viewable"
    no_access = "no_access"


@dataclass
class SplitAccess:
    """Defines access level for a dataset split."""

    split: str
    access: SplitAccessLevel

    @classmethod
    def viewable(cls, split: str) -> SplitAccess:
        return cls(split=split, access=SplitAccessLevel.viewable)

    @classmethod
    def non_viewable(cls, split: str) -> SplitAccess:
        return cls(split=split, access=SplitAccessLevel.non_viewable)

    @classmethod
    def no_access(cls, split: str) -> SplitAccess:
        return cls(split=split, access=SplitAccessLevel.no_access)


default_split_accesses = (
    SplitAccess.no_access(DefaultSplitNames.test),
    SplitAccess.non_viewable(DefaultSplitNames.validation),
)


def get_non_viewable_splits(split_accesses: list[SplitAccess]) -> list[str]:
    """Splits whose rows/details are not viewable (non_viewable and no_access).

    no_access is strictly more restrictive than non_viewable, so it is excluded
    everywhere non_viewable is. The non_viewable/no_access distinction (summary +
    agent-evaluable vs. not) is enforced in the evaluation engine, not here.
    """
    return [
        sa.split
        for sa in split_accesses
        if sa.access in (SplitAccessLevel.non_viewable, SplitAccessLevel.no_access)
    ]


class DatasetInfo(BaseModel):
    """An identifier and summary of a dataset.

    Attributes:
        id: Unique id of the dataset
        splits: The number of samples in each split
        description: A description of the dataset
        features: The features of the dataset
    """

    id: str
    splits: dict[str, int]
    description: str | None = None
    features: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_id(self) -> DatasetInfo:
        """Validate that the id is a valid id."""
        assert is_valid_id(self.id), "Dataset id must be a valid id."
        return self
