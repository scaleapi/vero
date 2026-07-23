from datetime import datetime

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """A candidate system."""

    commit: str
    repo_name: str
    parent_commit: str | None = Field(default=None, repr=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(), repr=False)
    message: str | None = Field(default=None, repr=False)

    @property
    def id(self) -> tuple[str, str]:
        """Returns the id of the candidate."""
        return (self.repo_name, self.commit)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return False
        return self.id == other.id
