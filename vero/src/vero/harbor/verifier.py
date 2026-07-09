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

import asyncio
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
        score_baseline: bool = False,
        baseline_score_attempts: int = 2,
        auto_best_baseline_floor: bool = True,
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
        self.score_baseline = score_baseline
        # auto_best selection floor: never ship a candidate that fails to beat the
        # untouched baseline on the selection split. Without it, auto_best (which
        # excludes base_commit from the candidate pool) selects the least-bad
        # candidate even when every candidate regressed, shipping a regression
        # (observed live: a weak inner model, every candidate below baseline).
        self.auto_best_baseline_floor = auto_best_baseline_floor
        # Every reward-critical finalize eval (targets, shortlist re-scores,
        # floor, baseline) is retried this many times total: the nested eval can
        # fail transiently (a nested harbor run crashing right after a large
        # eval), and a single blip must not abort finalize; a trial that ships
        # no reward.json loses its result entirely.
        self._baseline_score_attempts = max(1, baseline_score_attempts)
        # Finalize is idempotent: Harbor may retry the verifier (or an operator
        # may re-POST /finalize), and a replayed finalize must return the FIRST
        # completed result verbatim. Re-running it would re-rank against a DB
        # that now also contains the first finalize's own admin evals, so a
        # retry could select a DIFFERENT champion than the one already reported.
        self._finalize_lock = asyncio.Lock()
        self._finalize_result: dict | None = None

    async def finalize(self) -> dict:
        """Idempotent entry point: the first completed finalize is cached and
        replayed verbatim on any retry (see __init__ for why re-running would
        be unsound). The lock serializes concurrent calls so exactly one
        finalize ever computes."""
        async with self._finalize_lock:
            if self._finalize_result is not None:
                logger.info("finalize: replaying cached result (idempotent)")
                return self._finalize_result
            result = await self._finalize()
            self._finalize_result = result
            return result

    async def _finalize(self) -> dict:
        """Select the commit, score it on every target, and score the baseline.

        Returns a wrapper ``{"rewards": {reward_key: score}, "baseline": {...}}``.
        ``rewards`` is the reward.json payload the outer harness consumes (the CLI
        writes only that to reward.json); ``baseline`` is the outcome of baseline
        scoring, surfaced here because it is otherwise invisible: the admin volume
        it used to be written to does not survive teardown, and the finalize
        response echoed to the trial's stdout is the only host-durable channel.

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
            rewards = {t.reward_key: float(default_minimum_score) for t in self.targets}
            return {"rewards": rewards, "baseline": {"skipped": "no candidate commit"}}
        logger.info(f"Verifier selected commit {sha} (mode={self.reward_mode})")
        rewards: dict[str, float] = {}
        target_errors: dict[str, str] = {}
        for target in self.targets:
            score, cause = await self._admin_eval_score(
                task=target.task,
                dataset_id=target.dataset_id,
                split=target.split,
                commit=sha,
                sample_ids=target.sample_ids,
                what=f"target '{target.reward_key}'",
            )
            if score is None:
                # Persistent failure: floor the target so reward.json still
                # ships, and record the failure WITH ITS CAUSE in the wrapper
                # (echoed to the trial's durable stdout). A floored-by-outage
                # reward must never masquerade as a measured 0.0, and the cause
                # separates the two floored cases that demand opposite actions:
                # a champion that deterministically crashes on this target's
                # executor (a real, reportable portability failure) vs an infra
                # outage (invalidate and re-run).
                rewards[target.reward_key] = float(default_minimum_score)
                target_errors[target.reward_key] = (
                    f"eval failed after {self._baseline_score_attempts} attempt(s); "
                    f"reward floored, not measured"
                    + (f"; cause: {cause}" if cause else "")
                )
            else:
                rewards[target.reward_key] = score
        baseline = await self._maybe_score_baseline(rewards)
        result = {"rewards": rewards, "baseline": baseline}
        if target_errors:
            result["target_errors"] = target_errors
        return result

    async def _admin_eval_score(
        self,
        *,
        task: str | None,
        dataset_id: str,
        split: str,
        commit: str,
        sample_ids: list[int] | None = None,
        what: str,
    ) -> tuple[float | None, str | None]:
        """One reward-critical admin eval with bounded retry.

        Returns ``(score, failure_cause)``. The score counts errored samples
        as 0.0 (min-fill: an errored sample is a failed measurement of the
        candidate, and excluding it would reward candidates whose failures
        error out rather than score). An eval in which NO sample scored is
        indistinguishable from an infrastructure outage and must never quietly
        become 0.0, so it is retried like an exception; ``(None, cause)``
        after the last attempt means "could not measure", and the caller
        decides the fail-safe. ``cause`` summarizes the dominant per-sample
        errors so the durable record can distinguish a deterministically
        crashing candidate from an outage.
        """
        last_error: Exception | str | None = None
        for attempt in range(1, self._baseline_score_attempts + 1):
            try:
                exp = await self.engine.evaluate_admin(
                    task=task,
                    dataset_id=dataset_id,
                    split=split,
                    commit=commit,
                    sample_ids=sample_ids,
                )
                if exp.result.score(fill_score=None) is None:
                    last_error = (
                        "eval scored no samples (all errored or empty); "
                        + self._dominant_sample_errors(exp)
                    )
                    logger.warning(
                        "%s attempt %d/%d: %s",
                        what, attempt, self._baseline_score_attempts, last_error,
                    )
                    continue
                score = exp.result.score()
                if score is None:
                    # Should be unreachable (the strict check above already
                    # passed), but a None here must consume a retry like any
                    # other unmeasurable outcome, never bypass the loop.
                    last_error = "eval returned no aggregate score"
                    continue
                return float(score), None
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced as None
                last_error = exc
                logger.warning(
                    "%s attempt %d/%d failed: %s",
                    what, attempt, self._baseline_score_attempts, exc,
                )
        logger.error(
            "%s failed after %d attempt(s): %s",
            what, self._baseline_score_attempts, last_error,
        )
        return None, str(last_error) if last_error is not None else None

    @staticmethod
    def _dominant_sample_errors(exp) -> str:
        """Frequency summary of the per-sample error strings of an experiment
        (e.g. "12x: No verifier rewards for task ... (attempts died:
        UnsupportedParamsError x6)"). One identical cause across every sample
        is the signature of a deterministic candidate crash; a mixed bag points
        at infra. Top few only: the value is the shape, not the full list.
        Diagnostics must never fail finalize, so any surprise shape degrades
        to a fixed string instead of raising."""
        try:
            counts: dict[str, int] = {}
            for r in exp.result.sample_results.values():
                if r.error:
                    counts[r.error] = counts.get(r.error, 0) + 1
            if not counts:
                return "no per-sample errors recorded"
            top = sorted(counts.items(), key=lambda i: -i[1])[:3]
            return "; ".join(f"{n}x: {err}" for err, n in top)
        except Exception:  # noqa: BLE001 - diagnostics only
            return "no per-sample errors recorded"

    async def _maybe_score_baseline(self, rewards: dict[str, float]) -> dict:
        """Admin-score the unmodified baseline on every target and report it.

        An optimized candidate can score WORSE than the untouched baseline
        (observed live: a weak inner model went 0.3 -> 0.2 after optimization);
        without this, the regression is invisible because auto_best excludes the
        baseline from selection and nothing else ever scores it.

        Returns a structured outcome (``{"scores": ...}`` / ``{"error": ...}`` /
        ``{"skipped": ...}``) that ``finalize`` surfaces in its response, so a
        skip or failure is durably recorded rather than lost. A live trial once
        skipped this silently: the nested baseline eval failed transiently and
        the only record (a log line) died with the container at teardown. So the
        eval is retried once, and any failure is returned instead of swallowed.
        Baseline scoring still never fails the trial (reward.json is unaffected).
        A best-effort copy is also written to <admin_volume>/baseline.json for
        in-cluster debugging while the sidecar is alive.
        """
        if not self.score_baseline:
            return {"skipped": "score_baseline is disabled"}
        if not self.base_commit:
            # Misconfiguration must not be a silent no-op: the operator asked
            # for baseline scoring and would otherwise never learn it is off.
            logger.warning(
                "score_baseline=True but base_commit is not set; skipping "
                "baseline scoring."
            )
            return {"skipped": "base_commit is not set"}

        last_error: Exception | None = None
        for attempt in range(1, self._baseline_score_attempts + 1):
            try:
                baselines: dict[str, float] = {}
                for target in self.targets:
                    exp = await self.engine.evaluate_admin(
                        task=target.task,
                        dataset_id=target.dataset_id,
                        split=target.split,
                        commit=self.base_commit,
                        sample_ids=target.sample_ids,
                    )
                    if exp.result.score(fill_score=None) is None:
                        # All-error/empty is an outage, not a 0.0 baseline: a
                        # zero here would fake a huge candidate improvement.
                        # Raise into the retry loop instead.
                        raise RuntimeError(
                            f"baseline eval on '{target.reward_key}' scored no "
                            f"samples (all errored or empty)"
                        )
                    score = exp.result.score()
                    baselines[target.reward_key] = (
                        float(score) if score is not None else default_minimum_score
                    )
                # Best-effort local copy (admin volume does not survive teardown;
                # the return value is the durable record).
                try:
                    self.admin_volume.mkdir(parents=True, exist_ok=True)
                    (self.admin_volume / "baseline.json").write_text(
                        json.dumps(baselines, indent=2)
                    )
                except OSError:
                    logger.warning("could not write baseline.json to the admin volume")
                for key, value in rewards.items():
                    base = baselines.get(key)
                    tag = (
                        " (REGRESSION vs baseline)"
                        if base is not None and value < base
                        else ""
                    )
                    logger.info("finalize: %s=%s baseline=%s%s", key, value, base, tag)
                return {"scores": baselines, "attempts": attempt}
            except Exception as exc:  # noqa: BLE001 - never fail the trial on baseline scoring
                last_error = exc
                logger.warning(
                    "baseline scoring attempt %d/%d failed: %s",
                    attempt,
                    self._baseline_score_attempts,
                    exc,
                )
        logger.exception(
            "baseline scoring failed after %d attempt(s); reward.json is unaffected",
            self._baseline_score_attempts,
            exc_info=last_error,
        )
        return {
            "error": str(last_error),
            "error_type": type(last_error).__name__ if last_error else None,
            "attempts": self._baseline_score_attempts,
        }

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
        # Rank on FULL-split evals only, WHEN ANY EXIST. A subset eval
        # (num_samples / sample_ids, taught by the multi-fidelity lever) records
        # a mean over a handful of samples, so a lucky small subset can inflate a
        # candidate's recorded score and push it into the top-K shortlist over a
        # genuinely better full-split candidate. A full-split eval is recorded
        # with dataset_subset_sample_ids = None (DatasetSubset.is_full_set); any
        # non-null value is a subset. If at least one candidate has a full-split
        # eval, subset evals are dropped for ranking so they cannot displace it.
        # If EVERY eval is a subset, there is no full-split candidate to protect,
        # so the subset evals are the only ranking signal and are kept (the
        # winner is still decided by an admin re-score on the full split, so this
        # only controls which commits enter the shortlist).
        if "dataset_subset_sample_ids" in split_df.columns:
            full_split_df = split_df[split_df["dataset_subset_sample_ids"].isna()]
            if len(full_split_df) > 0:
                split_df = full_split_df
        # Shortlist by recorded score (cheap, agent-influenced -> not trusted as
        # final). Recorded evals are POOLED before shortlisting, in two steps:
        # every eval of the same commit averages into that commit's score, then
        # commits with the same git TREE (identical content) collapse into one
        # candidate group scored by the group mean. Max-over-rows selection made
        # every re-measurement an independent lottery draw, and one live
        # optimizer farmed exactly that ("distinct empty commits = clean
        # independent lottery tickets") while another refused to re-measure its
        # champion to protect a lucky draw. Pooling makes re-measurement
        # variance-REDUCING (as statistics wants) instead of max-inflating, and
        # tree-dedup stops identical content from stuffing the top-K shortlist
        # or collecting several admin re-score draws.
        agg: dict[str, tuple[str, str]] = {
            "mean_score": ("mean_score", "mean"),
            "candidate_created_at": ("candidate_created_at", "max"),
        }
        if "dataset_subset_dataset_id" in split_df.columns:
            agg["dataset_subset_dataset_id"] = ("dataset_subset_dataset_id", "first")
        per_commit = (
            split_df.groupby("candidate_commit").agg(**agg).reset_index()
        )
        trees: dict[str, str] = {}
        for commit in per_commit["candidate_commit"]:
            # Unresolvable tree (non-git workspace, unknown sha) falls back to
            # the commit itself: no pooling across commits, never a crash.
            trees[commit] = (await self._tree_of(commit)) or commit
        per_commit["_tree"] = per_commit["candidate_commit"].map(trees)
        # Newest commit represents its tree group (any member is equivalent
        # content-wise; newest keeps logs intuitive).
        per_commit = per_commit.sort_values(
            by=["candidate_created_at"], ascending=[False]
        )
        group_agg: dict[str, tuple[str, str]] = {
            "mean_score": ("mean_score", "mean"),
            "candidate_commit": ("candidate_commit", "first"),
            "candidate_created_at": ("candidate_created_at", "first"),
        }
        if "dataset_subset_dataset_id" in per_commit.columns:
            group_agg["dataset_subset_dataset_id"] = (
                "dataset_subset_dataset_id", "first",
            )
        pooled = per_commit.groupby("_tree", sort=False).agg(**group_agg)
        ranked = pooled.sort_values(
            by=["mean_score", "candidate_created_at"], ascending=[False, False]
        )
        shortlist = ranked.head(max(1, self.rescore_top_k))

        rescored: list[tuple[float, int, str]] = []
        for idx, (_, row) in enumerate(shortlist.iterrows()):
            commit = row["candidate_commit"]
            dataset_id = row.get("dataset_subset_dataset_id")
            score, _cause = await self._admin_eval_score(
                task=self.selection_task,
                dataset_id=dataset_id,
                split=self.selection_split,
                commit=commit,
                what=f"auto_best re-score of {commit}",
            )
            # A candidate whose re-score persistently fails is unscorable, not
            # zero-scored: it keeps the floor value and cannot win on a fluke,
            # but its failure does not abort the other candidates' re-scores.
            admin_score = score if score is not None else float(default_minimum_score)
            # Tie-break by shortlist position (already ordered by recorded score
            # then recency), so ties resolve deterministically without depending on
            # the type of candidate_created_at (a datetime in the real DB).
            rescored.append((admin_score, idx, commit))
            logger.info(
                "auto_best re-score: commit=%s admin_score=%s (pooled recorded=%s)",
                commit,
                admin_score,
                row["mean_score"],
            )
        # Highest admin score wins; ties break to the earliest shortlist position.
        rescored.sort(key=lambda t: (-t[0], t[1]))
        best_score, _, best_commit = rescored[0]

        # Selection floor: never ship a candidate that fails to beat the untouched
        # baseline on the selection split. auto_best excludes base_commit from the
        # candidate pool, so without this it selects the least-bad candidate even
        # when every candidate regressed. Revert to the seed instead. Strict '>' so
        # a statistical tie also reverts: if the optimizer cannot show an
        # improvement, shipping the seed is the safe outcome. Needs a base_commit to
        # compare against; costs one extra admin eval on the selection split.
        if self.auto_best_baseline_floor and self.base_commit is not None:
            base_dataset_id = self.selection_dataset_id
            if base_dataset_id is None:
                base_dataset_id = shortlist.iloc[0].get("dataset_subset_dataset_id")
            base_score_opt, _cause = await self._admin_eval_score(
                task=self.selection_task,
                dataset_id=base_dataset_id,
                split=self.selection_split,
                commit=self.base_commit,
                what="auto_best floor (baseline)",
            )
            if base_score_opt is None:
                # The floor exists to stop shipped regressions; shipping a
                # candidate with the floor check unmeasured re-opens exactly
                # that hole. Fail safe: revert to the seed.
                logger.error(
                    "auto_best floor: baseline could not be measured; failing "
                    "safe to base_commit %s instead of shipping unverified "
                    "candidate %s.",
                    self.base_commit, best_commit,
                )
                return self.base_commit
            base_score = base_score_opt
            if best_score <= base_score:
                logger.info(
                    "auto_best floor: best candidate %s (admin_score=%s) does not beat "
                    "baseline %s (admin_score=%s); reverting to base_commit.",
                    best_commit, best_score, self.base_commit, base_score,
                )
                return self.base_commit
            logger.info(
                "auto_best floor: best candidate %s (%s) beats baseline (%s); keeping it.",
                best_commit, best_score, base_score,
            )
        return best_commit

    async def _tree_of(self, commit: str) -> str | None:
        """Git tree hash for a commit via the engine's workspace, or None when
        it cannot be resolved (non-git workspace, unknown sha). None makes the
        caller treat the commit as its own pooling group: degraded, never wrong."""
        workspace = getattr(self.engine.evaluator, "workspace", None)
        tree_hash = getattr(workspace, "tree_hash", None)
        if tree_hash is None:
            return None
        try:
            return await tree_hash(commit)
        except Exception:  # noqa: BLE001 - pooling is an optimization, not a gate
            logger.warning("could not resolve tree hash for commit %s", commit)
            return None
