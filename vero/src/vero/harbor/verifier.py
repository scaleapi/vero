"""Verifier: admin-side commit selection + hidden-split scoring -> reward.

Runs at trial end. In the shared-verifier deployment the eval sidecar is still
up, so the verifier (root, in the `main` container) reaches this logic through
the sidecar's token-gated ``finalize`` endpoint, sharing the engine's state
(repo, dataset, scoring, ledger, submission record). It selects the candidate
commit (submit: the agent's nominated commit | auto_best: the best commit on the
selection split, excluding the baseline) and scores it on a configured battery
of targets, emitting a multi-key reward dict that the wiring writes to Harbor's
reward.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vero.core.constants import default_minimum_score
from vero.evaluation.engine import EvaluationEngine

logger = logging.getLogger(__name__)


class NoCandidateError(RuntimeError):
    """Raised when no commit can be selected (no submission / no experiments)."""


@dataclass
class VerificationTarget:
    """One scoring target -> one named reward in reward.json."""

    task: str | None  # None in Mode B (the nested harbor strategy ignores the vero task)
    dataset_id: str
    split: str
    reward_key: str
    sample_ids: list[int] | None = None  # None = full split


class Verifier:
    def __init__(
        self,
        *,
        engine: EvaluationEngine,
        admin_volume: Path,
        reward_mode: Literal["submit", "auto_best"],
        targets: list[VerificationTarget],
        selection_split: str = "validation",
        base_commit: str | None = None,
    ):
        self.engine = engine
        self.admin_volume = Path(admin_volume)
        self.reward_mode = reward_mode
        self.targets = targets
        self.selection_split = selection_split
        self.base_commit = base_commit

    async def finalize(self) -> dict[str, float]:
        """Select the commit and score it on every target -> {reward_key: score}."""
        sha = self._select_commit()
        logger.info(f"Verifier selected commit {sha} (mode={self.reward_mode})")
        rewards: dict[str, float] = {}
        for target in self.targets:
            exp = await self.engine.evaluate_admin(
                task=target.task,
                dataset_id=target.dataset_id,
                split=target.split,
                commit=sha,
                sample_ids=target.sample_ids,
            )
            score = exp.result.score()
            rewards[target.reward_key] = (
                float(score) if score is not None else default_minimum_score
            )
        return rewards

    def _select_commit(self) -> str:
        if self.reward_mode == "submit":
            return self._submitted_commit()
        return self._best_from_db()

    def _submitted_commit(self) -> str:
        path = self.admin_volume / "submission.json"
        if not path.exists():
            raise NoCandidateError(
                "submit mode but no submission.json — the agent never submitted a commit."
            )
        commit = json.loads(path.read_text()).get("commit")
        if not commit:
            raise NoCandidateError("submission.json has no commit.")
        return commit

    def _best_from_db(self) -> str:
        """Best candidate by recorded score on the selection split (excludes baseline)."""
        if self.engine.db is None:
            raise NoCandidateError("auto_best mode but no experiment database.")
        df = self.engine.db.get_experiments_df(fill_score=default_minimum_score)
        if df.empty or "dataset_subset_split" not in df.columns:
            raise NoCandidateError("auto_best mode but no experiments recorded.")

        split_df = df[df["dataset_subset_split"] == self.selection_split]
        if self.base_commit is not None:
            split_df = split_df[split_df["candidate_commit"] != self.base_commit]
        if len(split_df) == 0:
            raise NoCandidateError(
                f"auto_best mode but no candidate experiments on split "
                f"'{self.selection_split}'."
            )
        best = split_df.sort_values(
            by=["mean_score", "candidate_created_at"], ascending=[False, False]
        ).iloc[0]
        return best["candidate_commit"]
