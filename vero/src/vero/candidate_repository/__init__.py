"""Durable repositories for program candidates."""

from vero.candidate_repository.base import CandidateRepository
from vero.candidate_repository.git import (
    CandidateRepositoryError,
    GitCandidateRepository,
)

__all__ = [
    "CandidateRepository",
    "CandidateRepositoryError",
    "GitCandidateRepository",
]
