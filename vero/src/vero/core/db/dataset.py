import hashlib

from pydantic import BaseModel, Field, field_validator


class DatasetSample(BaseModel):
    """Deprecated VeroTask identity for one dataset case."""

    dataset_id: str
    split: str
    sample_id: int


class DatasetSubset(BaseModel):
    """Deprecated VeroTask selection; use ``EvaluationSet``."""

    dataset_id: str
    split: str
    sample_ids: list[int] | None = Field(default=None, repr=False)

    @field_validator("sample_ids")
    def validate_sample_ids(cls, v: list[int] | None) -> list[int] | None:
        """Validate that the sample ids are sorted, so that the id is unique for a subset."""
        return sorted(v) if v else v

    @property
    def is_full_set(self) -> bool:
        return self.sample_ids is None

    @property
    def id(self) -> tuple[str, str, str]:
        hash_str = (
            hashlib.sha256(str(self.sample_ids).encode()).hexdigest()[:8]
            if not self.is_full_set
            else "full"
        )
        return (self.dataset_id, self.split, hash_str)
