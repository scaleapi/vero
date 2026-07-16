"""Durable repositories for program candidates."""

from vero.candidate_repository.base import CandidateRepository, CandidateRepositoryError
from vero.candidate_repository.git import GitCandidateRepository

__all__ = [
    "CandidateRepository",
    "CandidateRepositoryError",
    "GitCandidateRepository",
]
