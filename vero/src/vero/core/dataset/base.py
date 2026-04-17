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
    """Access levels for dataset splits."""

    viewable = "viewable"
    non_viewable = "non_viewable"


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


default_split_accesses = (
    SplitAccess.non_viewable(DefaultSplitNames.test),
    SplitAccess.non_viewable(DefaultSplitNames.validation),
)


def get_non_viewable_splits(split_accesses: list[SplitAccess]) -> list[str]:
    """Extract non-viewable splits from a list of SplitAccess."""
    return [
        sa.split for sa in split_accesses if sa.access == SplitAccessLevel.non_viewable
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
