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
from vero.utils import run_subprocess_with_tee

if TYPE_CHECKING:
    from vero.core.evaluation import EvaluationParameters
    from vero.workspace import Workspace

logger = logging.getLogger(__name__)


class HarborRunner:
    """Mode-B EvalStrategy: nested `harbor run` + collate -> SampleResults."""

    def __init__(self, config: HarborConfig):
        self.config = config

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
        cmd = [
            "uv", "run", "--project", project_path,
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
        result = await run_subprocess_with_tee(
            cmd, timeout=params.timeout, cwd=project_path
        )
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
        groups = (
            self._trial_groups(jobs_dir)
            if self.config.aggregate_attempts == "mean"
            else {}
        )
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
            groups.setdefault(task_name, []).append(data)
        return groups

    @staticmethod
    def _trial_rank(data: dict, result_json: Path) -> tuple:
        """Sort key for picking the best of several trials of one task. Higher wins:
        prefer a clean trial with rewards, then any trial with rewards, then the most
        recent attempt (finished_at, falling back to file mtime)."""
        has_rewards = bool((data.get("verifier_result") or {}).get("rewards"))
        clean = has_rewards and not data.get("exception_info")
        finished_at = data.get("finished_at") or ""
        try:
            mtime = result_json.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (clean, has_rewards, finished_at, mtime)

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
        # Mean aggregation across attempts: average the reward over every clean
        # scored attempt (a verified 0.0 is a valid measurement; an exception is
        # not). Falls through to the single best trial when nothing scored clean.
        if attempts:
            scored = [
                self._extract_reward((t.get("verifier_result") or {}).get("rewards"))
                for t in attempts
                if (t.get("verifier_result") or {}).get("rewards")
                and not t.get("exception_info")
            ]
            if scored:
                return SampleResult(
                    score=sum(scored) / len(scored),
                    metrics={
                        "reward_mean": sum(scored) / len(scored),
                        "n_attempts": float(len(attempts)),
                        "n_scored": float(len(scored)),
                    },
                    output={
                        "task_name": task_name,
                        "attempt_scores": scored,
                        "aggregate": "mean",
                    },
                    **common,
                )
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        if not rewards:
            return SampleResult(
                error=f"No verifier rewards for task '{task_name}'.",
                output={"task_name": task_name, "trial_name": trial.get("trial_name")},
                **common,
            )
        return SampleResult(
            score=self._extract_reward(rewards),
            metrics={k: float(v) for k, v in rewards.items()},
            output={
                "task_name": task_name,
                "trial_name": trial.get("trial_name"),
                "rewards": rewards,
            },
            **common,
        )

    def _extract_reward(self, rewards: dict) -> float:
        for key in (self.config.reward_key, "pass", "reward"):
            if key and key in rewards:
                return float(rewards[key])
        values = [float(v) for v in rewards.values()]
        return sum(values) / len(values) if values else 0.0

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
