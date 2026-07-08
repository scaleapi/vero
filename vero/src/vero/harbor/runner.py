"""HarborRunner — the Mode-B evaluation strategy.

Implements ``EvalStrategy``: for a checked-out candidate, runs a nested ``harbor run``
(in the candidate's own uv env) over the Harbor tasks selected by the split/sample_ids,
then collates the jobs dir into vero ``SampleResult``s. One Harbor task = one sample.

Shells out to the ``harbor`` CLI (no harbor import here) and reads trial ``result.json``
as plain dicts, so ``vero`` itself needs no ``harbor`` dependency at runtime.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from vero.core.db.dataset import DatasetSample
from vero.core.db.result import SampleResult
from vero.core.sessions import (
    get_vero_home_dir,
    load_sample_result,
    save_sample_result,
)
from vero.harbor.config import HarborConfig
from vero.utils import SubprocessTimeoutError, run_subprocess_with_tee

if TYPE_CHECKING:
    from vero.core.evaluation import EvaluationParameters
    from vero.workspace import Workspace

logger = logging.getLogger(__name__)


class HarborRunner:
    """Mode-B EvalStrategy: nested `harbor run` + collate -> SampleResults."""

    def __init__(
        self,
        config: HarborConfig,
        *,
        feedback_transcripts: bool = False,
        feedback_max_bytes: int = 3000,
        expose_attempt_detail: bool = False,
    ):
        self.config = config
        # Lever 1: attach the transcript tail of a FAILED sample's trial to its
        # SampleResult.feedback. Whether the agent ever sees it is decided by
        # the sidecar's tier routing (per-sample files are viewable-only), not
        # here; this only controls whether the field is filled at collation.
        self.feedback_transcripts = feedback_transcripts
        self.feedback_max_bytes = feedback_max_bytes
        # Lever 3: attach a per-attempt {reward, exception} list to each
        # sample's output. Same tier gate as feedback: filled at collation,
        # exposed only via the viewable-split per-sample files.
        self.expose_attempt_detail = expose_attempt_detail

    async def produce_sample_results(
        self,
        *,
        workspace: Workspace,
        params: EvaluationParameters,
        result_dir: Path,
    ) -> None:
        pairs = self._task_names_for(params)  # [(sample_id, task_name), ...]
        if not pairs:
            return
        jobs_dir = Path(result_dir) / "jobs"

        # Resume: only run tasks not already completed successfully. A persisted
        # *error* sample (transient harbor/verifier failure) is NOT done, so it is
        # re-run rather than permanently skipped.
        pending = [(sid, t) for sid, t in pairs if not self._is_done(params, sid)]
        if pending:
            await self._run_harbor(
                str(workspace.project_path), params, [t for _, t in pending], jobs_dir
            )
        self._collate(jobs_dir, pairs, params, ran=[t for _, t in pending])

    # ------------------------------------------------------------------
    # Task selection (host-side; just task names)
    # ------------------------------------------------------------------

    def _task_names_for(self, params: EvaluationParameters) -> list[tuple[int, str]]:
        from vero.core.dataset.store import load_dataset

        vero_home = get_vero_home_dir()
        dataset = load_dataset(
            vero_home / "sessions",
            vero_home / "datasets",
            params.session_id,
            params.run.dataset_subset.dataset_id,
        )
        split = dataset[params.run.dataset_subset.split]
        ids = params.run.dataset_subset.sample_ids
        if ids is None:
            ids = list(range(len(split)))
        return [(i, split[i]["task_name"]) for i in ids]

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def _build_command(
        self,
        project_path: str,
        params: EvaluationParameters,
        task_names: list[str],
        jobs_dir: Path,
    ) -> list[str]:
        c = self.config
        cmd = ["uv", "run", "--project", project_path]
        # Resolve the nested harbor from the trusted spec rather than the
        # candidate's lockfile (see HarborConfig.harbor_requirement). This
        # raises the bar from "edit one pyproject line" to "tamper with the
        # orchestrator in-process at runtime"; it is not a full boundary —
        # the agent's code still imports into the nested harbor process via
        # --agent-import-path, so a hostile candidate can in principle still
        # forge results from inside. Full isolation needs an out-of-process
        # verifier and is tracked separately.
        if c.harbor_requirement:
            cmd += ["--with", c.harbor_requirement]
        cmd += [
            "harbor", "run",
            *c.source_args(),
            "--agent-import-path", c.agent_import_path,
            "-e", c.environment,
            "-n", str(params.max_concurrency),
            "--n-attempts", str(c.n_attempts),
            "--max-retries", str(c.max_retries),
        ]
        if c.model:
            cmd += ["-m", c.model]
        for task_name in task_names:
            cmd += ["-i", task_name]
        cmd += ["--jobs-dir", str(jobs_dir), *c.extra_args]
        return cmd

    async def _run_harbor(
        self,
        project_path: str,
        params: EvaluationParameters,
        task_names: list[str],
        jobs_dir: Path,
    ) -> None:
        cmd = self._build_command(project_path, params, task_names, jobs_dir)
        logger.info(f"Mode B: {' '.join(cmd)}")
        try:
            result = await run_subprocess_with_tee(
                cmd, timeout=params.timeout, cwd=project_path
            )
        except SubprocessTimeoutError as e:
            # A timed-out nested run is not fatal either: the eval's budget is
            # already spent and completed trials are on disk, so salvage them.
            # Propagating instead would return the agent a bare 500 with zero
            # information for a fully-debited eval.
            logger.warning(
                f"`harbor run` timed out after {params.timeout}s; collating "
                f"the trials that completed before the cutoff. stderr tail: "
                f"{(e.result.stderr or '')[-500:]}"
            )
            return
        # Non-zero is not fatal: partial trials may still exist; collation fills gaps.
        if result.returncode != 0:
            logger.warning(
                f"`harbor run` exited {result.returncode}: "
                f"{(result.stderr or '')[:500]}"
            )

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    def _collate(
        self,
        jobs_dir: Path,
        pairs: list[tuple[int, str]],
        params: EvaluationParameters,
        ran: list[str] | None = None,
    ) -> None:
        trials = self._load_trials(jobs_dir)  # {task_name: result_dict}
        # Guard against silently scoring everything 0. If we just ran tasks and
        # got either no trial results at all, or trial results whose task_names
        # match NONE of the requested ones (a task-name keying mismatch: the
        # dataset must store harbor's canonical '<org>/<name>' form), this is an
        # infrastructure failure, not an agent failure. Recording it as all-zero
        # samples would be indistinguishable from "the agent failed every task".
        if ran:
            if not trials:
                raise RuntimeError(
                    f"Nested `harbor run` produced no trial results for "
                    f"{len(ran)} task(s) (see harbor output above); refusing to "
                    f"score all samples 0."
                )
            if not any(t in trials for t in ran):
                raise RuntimeError(
                    f"Nested `harbor run` produced {len(trials)} trial "
                    f"result(s), but none match the requested task names "
                    f"(requested e.g. {ran[0]!r}; recorded e.g. "
                    f"{next(iter(trials))!r}). Task names must use harbor's "
                    f"canonical '<org>/<name>' form; refusing to score all "
                    f"samples 0."
                )
        need_attempts = (
            self.config.aggregate_attempts == "mean"
            or self.feedback_transcripts
            or self.expose_attempt_detail
        )
        groups = self._trial_groups(jobs_dir) if need_attempts else {}
        for sample_id, task_name in pairs:
            if self._is_done(params, sample_id):
                continue  # already collated successfully (resume); errors are redone
            sample_result = self._sample_result(
                trials.get(task_name),
                sample_id,
                task_name,
                params,
                attempts=groups.get(task_name),
            )
            save_sample_result(
                get_vero_home_dir() / "sessions",
                params.session_id,
                params.result_id,
                sample_id=sample_id,
                result=sample_result,
            )

    def _load_trials(self, jobs_dir: Path) -> dict[str, dict]:
        trials: dict[str, dict] = {}
        if not jobs_dir.exists():
            return trials
        # Trial result.json files live at <jobs>/<timestamp>/<trial>/result.json; the
        # job-level <jobs>/<timestamp>/result.json carries no task_name, so recurse and
        # key on task_name (skipping the job summary). A task may have several trials
        # (retries / multiple attempts); rglob order is undefined, so keep the BEST
        # trial per task deterministically rather than last-write-wins (a failing
        # retry must never clobber a passing trial).
        best_rank: dict[str, tuple] = {}
        for result_json in jobs_dir.rglob("result.json"):
            try:
                data = json.loads(result_json.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            task_name = data.get("task_name")
            if not task_name:
                continue
            rank = self._trial_rank(data, result_json)
            if task_name not in best_rank or rank > best_rank[task_name]:
                best_rank[task_name] = rank
                trials[task_name] = data
        return trials

    def _trial_groups(self, jobs_dir: Path) -> dict[str, list[dict]]:
        """ALL trial results per task (for mean aggregation across attempts)."""
        groups: dict[str, list[dict]] = {}
        if not jobs_dir.exists():
            return groups
        for result_json in jobs_dir.rglob("result.json"):
            try:
                data = json.loads(result_json.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            task_name = data.get("task_name")
            if not task_name:
                continue
            # Transcripts (agent/terminus_2.pane etc.) live next to result.json;
            # keep the dir so feedback can find them after the path is dropped.
            data["_trial_dir"] = str(result_json.parent)
            groups.setdefault(task_name, []).append(data)
        # rglob order is undefined; sort each group so "first attempt" is a
        # stable notion (feedback uses the first failed attempt's transcript).
        # An attempt with no finished_at must sort LAST, not first: an empty
        # string would sort ahead of every real ISO timestamp and mislabel a
        # timestamp-less attempt as the "first". The leading bool puts present
        # timestamps first (False < True), then orders by the timestamp, and
        # finally tie-breaks on the stable trial_name.
        for attempts in groups.values():
            attempts.sort(
                key=lambda d: (
                    d.get("finished_at") is None,
                    d.get("finished_at") or "",
                    d.get("trial_name") or "",
                )
            )
        return groups

    def _trial_rank(self, data: dict, result_json: Path) -> tuple:
        """Sort key for picking the best of several trials of one task. Higher wins:
        a clean scored trial first, then any scored trial, then the HIGHER REWARD,
        then recency (finished_at, falling back to file mtime).

        The reward must be part of the key: with concurrent attempts, finish order
        is nondeterministic, so ranking on recency alone made 'best' mean "the last
        clean attempt to finish", and a later clean 0.0 could silently replace an
        earlier clean 1.0. 'best' has to be monotone in the attempt scores
        (max-of-score, pass@k-like) or a passing trial can be clobbered."""
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        reward = self._extract_reward(rewards) if rewards else None
        has_rewards = reward is not None
        clean = has_rewards and not data.get("exception_info")
        finished_at = data.get("finished_at") or ""
        try:
            mtime = result_json.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (clean, has_rewards, reward if has_rewards else -1.0, finished_at, mtime)

    def _sample_result(
        self,
        trial: dict | None,
        sample_id: int,
        task_name: str,
        params: EvaluationParameters,
        attempts: list[dict] | None = None,
    ) -> SampleResult:
        common = {
            "dataset_sample": DatasetSample(
                sample_id=sample_id,
                split=params.run.dataset_subset.split,
                dataset_id=params.run.dataset_subset.dataset_id,
            ),
            "commit": params.run.candidate.commit,
            "result_id": params.result_id,
        }
        if trial is None:
            return SampleResult(
                error=f"No Harbor trial result for task '{task_name}'.", **common
            )
        attempt_detail = self._attempt_detail(attempts)

        def _out(output: dict) -> dict:
            if attempt_detail is not None:
                output["attempts"] = attempt_detail
            return output
        # Mean aggregation across attempts: average the reward over every attempt
        # that RAN. Harbor can record an exception (agent timeout, non-zero agent
        # exit) and still run the verifier, so such an attempt carries a real
        # measured 0.0. An attempt that died BEFORE the verifier scored it
        # (crash, rate limit) is also a real, failed attempt and counts as 0.0:
        # dropping it would estimate P(pass | attempt survived to scoring), which
        # a candidate can game by dying early on hard tasks. Measured live: a
        # no-retry candidate outscored its retry-hardened successors purely
        # through dropped rate-limited attempts, and won selection on the
        # artifact. n_dead in the metrics records how many zeros came from
        # unscored attempts so infra noise stays visible. A sample where NO
        # attempt scored falls through to the single-trial path (which errors):
        # an all-dead sample is an outage to investigate, never a silent 0.0.
        # `attempts` may also be present under 'best' aggregation (collation
        # loads them for the feedback levers), so the mean path is gated on the
        # config, not their presence.
        if attempts and self.config.aggregate_attempts == "mean":
            measured: list[float] = []
            n_scored = 0
            n_dead = 0
            n_clean = 0
            # Dead attempts are not interchangeable: a rate-limited attempt is
            # infra noise that retunes with capacity, while a crash points at
            # the candidate (or a task bug), and dead attempts cluster hard by
            # cause in practice. n_dead alone hides that, so record the
            # exception type behind every zero-filled attempt.
            dead_types: dict[str, int] = {}
            for t in attempts:
                rewards = (t.get("verifier_result") or {}).get("rewards") or {}
                reward = self._extract_reward(rewards) if rewards else None
                if reward is not None:
                    measured.append(reward)
                    n_scored += 1
                    if not t.get("exception_info"):
                        n_clean += 1
                else:
                    measured.append(0.0)
                    n_dead += 1
                    exc = (t.get("exception_info") or {}).get("exception_type")
                    # An attempt can die without a recorded exception (the
                    # verifier simply produced no rewards); keep it countable.
                    key = exc or "no_rewards_recorded"
                    dead_types[key] = dead_types.get(key, 0) + 1
            if n_scored:
                if len(measured) < self.config.n_attempts or n_dead:
                    # Fewer or dirtier measurements than the config promises:
                    # the mean is noisier (or partly zero-filled). Never let
                    # that happen silently; the metrics carry the actual counts.
                    logger.warning(
                        f"Task '{task_name}': mean over {len(measured)} "
                        f"attempt(s) of {self.config.n_attempts} configured "
                        f"({n_scored} scored, {n_dead} dead counted 0.0)."
                    )
                mean = sum(measured) / len(measured)
                mean_output = {
                    "task_name": task_name,
                    "attempt_scores": measured,
                    "aggregate": "mean",
                }
                if dead_types:
                    # dict, not metrics: metrics are float-valued by contract.
                    mean_output["dead_exception_types"] = dead_types
                return SampleResult(
                    score=mean,
                    feedback=self._failure_feedback(mean, attempts),
                    metrics={
                        "reward_mean": mean,
                        "n_attempts": float(len(attempts)),
                        "n_scored": float(n_scored),
                        "n_dead": float(n_dead),
                        "n_clean": float(n_clean),
                    },
                    output=_out(mean_output),
                    **common,
                )
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        if not rewards:
            return SampleResult(
                error=f"No verifier rewards for task '{task_name}'.",
                # The agent died before scoring. A candidate edit that CRASHES
                # the agent lands here, and "no verifier rewards" alone gives
                # the optimizer no way to see its own crash; the transcript
                # does. Passed as score 0.0: an unscored attempt counts as a
                # failure everywhere else too.
                feedback=self._failure_feedback(0.0, attempts),
                output=_out(
                    {"task_name": task_name, "trial_name": trial.get("trial_name")}
                ),
                **common,
            )
        score = self._extract_reward(rewards)
        if score is None:
            # The verifier scored, but not on the configured metric (or on
            # several unrecognized ones). Scoring a substitute metric, or an
            # average, would silently change what the number means: error loud.
            return SampleResult(
                error=(
                    f"Rewards for task '{task_name}' carry no usable metric "
                    f"(reward_key={self.config.reward_key!r}, "
                    f"keys={sorted(rewards)})."
                ),
                output=_out(
                    {"task_name": task_name, "trial_name": trial.get("trial_name")}
                ),
                **common,
            )
        return SampleResult(
            score=score,
            feedback=self._failure_feedback(score, attempts),
            metrics={k: float(v) for k, v in rewards.items()},
            output=_out({
                "task_name": task_name,
                "trial_name": trial.get("trial_name"),
                "rewards": rewards,
            }),
            **common,
        )

    def _extract_reward(self, rewards: dict) -> float | None:
        """Reward for one trial's rewards dict, or None when no unambiguous
        metric is present.

        A configured reward_key is a contract: a rewards dict missing it is an
        unscorable measurement (None), never a silent fallback to another key.
        Falling back would let attempts within one mean be scored on different
        metrics, and averaging arbitrary keys would let a candidate inflate its
        score by emitting easy auxiliary metrics (lint, partial credit) beside
        the real one. Without a configured key, 'pass' then 'reward' are
        accepted, then a sole remaining key (unambiguous); several unrecognized
        keys are refused (None), not averaged.
        """
        if self.config.reward_key:
            value = rewards.get(self.config.reward_key)
            return None if value is None else float(value)
        for key in ("pass", "reward"):
            if key in rewards:
                return float(rewards[key])
        if len(rewards) == 1:
            return float(next(iter(rewards.values())))
        return None

    def _attempt_detail(self, attempts: list[dict] | None) -> list[dict] | None:
        """Lever 3: one {reward, exception} entry per attempt, in attempt order
        (sorted at load). reward is None when the attempt died before the
        verifier scored it; exception is the recorded exception class name
        (None for clean attempts). Off (or no attempts loaded) returns None,
        which leaves the output dict without an 'attempts' key at all."""
        if not self.expose_attempt_detail or not attempts:
            return None
        detail = []
        for attempt in attempts:
            rewards = (attempt.get("verifier_result") or {}).get("rewards")
            detail.append(
                {
                    "reward": self._extract_reward(rewards) if rewards else None,
                    "exception": (attempt.get("exception_info") or {}).get(
                        "exception_type"
                    ),
                }
            )
        return detail

    def _failure_feedback(
        self, score: float, attempts: list[dict] | None
    ) -> str | None:
        """Lever 1: transcript tail for a failed sample (score 0.0).

        Walks the failed attempts in load order (attempts are sorted at load)
        and returns the FIRST one with a readable transcript tail: the earliest
        failure is the cheapest reproducible one, and one tail per sample bounds
        the payload. A failed attempt with no recorded trial dir, or whose trial
        recorded no transcript, does not end the search: the next failed attempt
        is tried before giving up. Passed samples, and everything with the lever
        off, return None (the field serializes as null either way, so responses
        are byte-identical to before when disabled).
        """
        if not self.feedback_transcripts or score != 0.0 or not attempts:
            return None
        # feedback_max_bytes <= 0 means "no feedback", never "unbounded": a bare
        # data[-0:] slice would return the WHOLE transcript, so the cap must be
        # positive to emit anything at all.
        if self.feedback_max_bytes <= 0:
            return None
        for attempt in attempts:
            rewards = (attempt.get("verifier_result") or {}).get("rewards")
            reward = self._extract_reward(rewards) if rewards else None
            if reward is not None and reward != 0.0:
                continue
            # reward is None = the attempt died before the verifier scored it.
            # It counts as 0.0 in mean aggregation, so its transcript (which
            # shows the crash) is fair feedback material like any failure.
            trial_dir = attempt.get("_trial_dir")
            if not trial_dir:
                continue
            tail = self._read_transcript_tail(Path(trial_dir))
            if tail is not None:
                return tail
        return None

    def _read_transcript_tail(self, trial_dir: Path) -> str | None:
        """Last ``feedback_max_bytes`` of the trial's transcript: the terminal
        pane when present, else the trajectory; None (field omitted) when the
        trial recorded neither.

        A non-positive cap emits nothing (matches _failure_feedback's guard).
        The transcript path is confined to the trial dir: a symlinked transcript
        file, or a resolved path that escapes the trial dir, is skipped silently
        so a hostile trial layout cannot exfiltrate files outside its own dir.
        """
        if self.feedback_max_bytes <= 0:
            return None
        trial_root = trial_dir.resolve()
        for rel in ("agent/terminus_2.pane", "agent/trajectory.json"):
            path = trial_dir / rel
            # Reject symlinks outright, and any path that resolves outside the
            # trial dir, before touching the bytes.
            if path.is_symlink():
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(trial_root)
            except (OSError, ValueError):
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            # An empty transcript carries nothing: keep looking (an empty pane
            # must fall through to the trajectory), and if every candidate is
            # empty return None so the caller tries the next failed attempt
            # instead of emitting "" as feedback.
            if not data:
                continue
            # errors="replace": a multibyte char straddling the cap boundary is
            # rendered as U+FFFD rather than crashing the collation.
            return data[-self.feedback_max_bytes :].decode("utf-8", errors="replace")
        return None

    def _existing(self, params: EvaluationParameters, sample_id: int) -> SampleResult | None:
        return load_sample_result(
            get_vero_home_dir() / "sessions",
            params.session_id,
            params.result_id,
            sample_id,
        )

    def _is_done(self, params: EvaluationParameters, sample_id: int) -> bool:
        """A sample is done only if a persisted result exists AND is not an error;
        a transiently-failed sample must be re-run on resume, not skipped."""
        existing = self._existing(params, sample_id)
        return existing is not None and not existing.is_error()
