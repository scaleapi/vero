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
        selection_task: str | None = None,
        selection_dataset_id: str | None = None,
        rescore_top_k: int = 3,
    ):
        self.engine = engine
        self.admin_volume = Path(admin_volume)
        self.reward_mode = reward_mode
        self.targets = targets
        self.selection_split = selection_split
        self.base_commit = base_commit
        # auto_best re-scores the top-K shortlist admin-side; selection_task is the
        # task to score the selection split with (the trusted, sidecar-baked scorer),
        # and selection_dataset_id constrains ranking to the intended dataset.
        self.selection_task = selection_task
        self.selection_dataset_id = selection_dataset_id
        self.rescore_top_k = rescore_top_k

    async def finalize(self) -> dict[str, float]:
        """Select the commit and score it on every target -> {reward_key: score}.

        A run in which the optimizer produced no scorable candidate (never
        submitted in ``submit`` mode; no non-baseline experiments on the
        selection split in ``auto_best`` mode) is a legitimate *outcome* of an
        optimization run, not an infrastructure failure: every target is
        floored at ``default_minimum_score`` so the outer harness records a
        reward of 0.0 instead of a missing-reward exception. Infrastructure
        problems (e.g. a missing experiment database) still raise.
        """
        try:
            sha = await self._select_commit()
        except NoCandidateError as exc:
            logger.warning(
                "No candidate commit to finalize (%s); flooring all %d target(s) "
                "at %s.",
                exc,
                len(self.targets),
                default_minimum_score,
            )
            return {t.reward_key: float(default_minimum_score) for t in self.targets}
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

    async def _select_commit(self) -> str:
        if self.reward_mode == "submit":
            return self._submitted_commit()
        return await self._best_from_db()

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

    async def _best_from_db(self) -> str:
        """Best candidate on the selection split, chosen by an admin re-score.

        The recorded ``mean_score`` is agent-influenced (the agent ran those
        selection-split evals), so it is used only to shortlist the top-K
        candidates. The winner is then decided by re-running ``evaluate_admin``
        on the selection split with the trusted (sidecar-baked) scorer and
        ranking by the admin score. This fails closed: an agent that inflated
        its recorded score cannot win unless the admin scorer agrees.
        """
        if self.engine.db is None:
            # Misconfiguration, not an agent outcome: surface as a hard error so
            # a broken sidecar doesn't silently zero every trial.
            raise RuntimeError("auto_best mode but no experiment database.")
        df = self.engine.db.get_experiments_df(fill_score=default_minimum_score)
        if df.empty or "dataset_subset_split" not in df.columns:
            raise NoCandidateError("auto_best mode but no experiments recorded.")

        split_df = df[df["dataset_subset_split"] == self.selection_split]
        if self.selection_dataset_id is not None and "dataset_subset_dataset_id" in split_df.columns:
            # Only rank candidates scored on the intended selection dataset: a
            # shared experiment DB may hold same-named splits across datasets, and
            # a high score from a foreign dataset must not select the winner.
            split_df = split_df[
                split_df["dataset_subset_dataset_id"] == self.selection_dataset_id
            ]
        if self.base_commit is not None:
            split_df = split_df[split_df["candidate_commit"] != self.base_commit]
        if len(split_df) == 0:
            raise NoCandidateError(
                f"auto_best mode but no candidate experiments on split "
                f"'{self.selection_split}'."
            )
        # Shortlist by recorded score (cheap, agent-influenced -> not trusted as
        # final), one row per candidate (highest recorded score wins the slot).
        ranked = split_df.sort_values(
            by=["mean_score", "candidate_created_at"], ascending=[False, False]
        )
        ranked = ranked.drop_duplicates(subset=["candidate_commit"], keep="first")
        shortlist = ranked.head(max(1, self.rescore_top_k))

        rescored: list[tuple[float, int, str]] = []
        for idx, (_, row) in enumerate(shortlist.iterrows()):
            commit = row["candidate_commit"]
            dataset_id = row.get("dataset_subset_dataset_id")
            exp = await self.engine.evaluate_admin(
                task=self.selection_task,
                dataset_id=dataset_id,
                split=self.selection_split,
                commit=commit,
            )
            score = exp.result.score()
            admin_score = float(score) if score is not None else default_minimum_score
            # Tie-break by shortlist position (already ordered by recorded score
            # then recency), so ties resolve deterministically without depending on
            # the type of candidate_created_at (a datetime in the real DB).
            rescored.append((admin_score, idx, commit))
            logger.info(
                "auto_best re-score: commit=%s admin_score=%s (recorded=%s)",
                commit,
                admin_score,
                row["mean_score"],
            )
        # Highest admin score wins; ties break to the earliest shortlist position.
        rescored.sort(key=lambda t: (-t[0], t[1]))
        return rescored[0][2]
