"""EvaluationSidecar: the privileged, transport-agnostic frontend over the
EvaluationEngine, plus the trust-boundary mechanics that only exist in the Harbor
sidecar — commit transfer from the mounted agent repo and tier-gated
write-routing of results across the two volumes.

The HTTP binding (`serve()`) is a thin shell added when the `vero harbor serve`
CLI lands; these handlers are framework-agnostic and unit-testable on their own.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

from vero.core.dataset.base import SplitAccess, SplitAccessLevel
from vero.core.db.database import Experiment
from vero.evaluation.engine import EvalRequest, EvaluationEngine
from vero.exceptions import InvalidSplitError
from vero.harbor.protocol import (
    EvalSummary,
    StatusSummary,
    build_status,
    summarize_experiment,
    tier_for_split,
)

logger = logging.getLogger(__name__)


class CommitTransferError(RuntimeError):
    """Raised when a commit cannot be fetched from the agent's mounted repo."""


class SubmitDisabledError(RuntimeError):
    """Raised when submit() is called but the task does not use submit selection."""


class EvaluationSidecar:
    """Agent-facing handlers over the EvaluationEngine.

    Wraps the engine with: commit transfer (mounted agent repo -> sidecar repo),
    result write-routing by split tier, and aggregate-safe responses. The engine
    meters agent calls (admin calls bypass).
    """

    def __init__(
        self,
        *,
        engine: EvaluationEngine,
        split_accesses: list[SplitAccess],
        agent_repo_path: Path,
        agent_volume: Path,
        admin_volume: Path,
        submit_enabled: bool = False,
    ):
        self.engine = engine
        self.split_accesses = split_accesses
        self.agent_repo_path = Path(agent_repo_path)
        self.agent_volume = Path(agent_volume)
        self.admin_volume = Path(admin_volume)
        self.submit_enabled = submit_enabled

    # ------------------------------------------------------------------
    # Handlers (the HTTP layer resolves `admin` from auth and calls these)
    # ------------------------------------------------------------------

    async def evaluate(self, req: EvalRequest, *, admin: bool = False) -> EvalSummary:
        sha = await self._transfer_commit(req.commit)
        exp = await self.engine.evaluate(replace(req, commit=sha), admin=admin)
        result_path = self._route_results(exp, admin=admin)
        budget_remaining = None
        if not admin:
            try:
                budget_remaining = self.engine.budget.get(req.dataset_id, req.split)
            except InvalidSplitError:
                pass
        return summarize_experiment(
            exp, result_path=result_path, budget_remaining=budget_remaining
        )

    async def submit(self, commit: str | None = None) -> dict:
        """Record the agent's nominated commit; terminal. No score returned."""
        if not self.submit_enabled:
            raise SubmitDisabledError(
                "This task does not use submit-based selection; submit is disabled."
            )
        sha = await self._transfer_commit(commit)
        self.admin_volume.mkdir(parents=True, exist_ok=True)
        (self.admin_volume / "submission.json").write_text(
            json.dumps({"commit": sha}, indent=2)
        )
        return {"submitted_commit": sha}

    def status(self) -> StatusSummary:
        return build_status(
            submit_enabled=self.submit_enabled,
            budget=self.engine.budget.status(),
            split_accesses=self.split_accesses,
        )

    def list_experiments(self) -> list[dict]:
        """All recorded experiments, unredacted (admin observability).

        One row per experiment: commit, dataset/split, recorded score, and
        creation time. Lets an operator (or the outer harness) watch an
        optimization run mid-flight without exec-ing into the container or
        waiting for finalize. Admin-only: recorded scores on non_viewable
        splits must not reach the agent.
        """
        if self.engine.db is None:
            return []
        df = self.engine.db.get_experiments_df()
        if df.empty:
            return []
        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "commit": r.get("candidate_commit"),
                    "dataset_id": r.get("dataset_subset_dataset_id"),
                    "split": r.get("dataset_subset_split"),
                    "mean_score": r.get("mean_score"),
                    "created_at": str(r.get("candidate_created_at")),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Trust-boundary mechanics
    # ------------------------------------------------------------------

    async def _transfer_commit(self, ref: str | None) -> str:
        """Fetch ``ref`` (default agent HEAD) from the mounted agent repo into the
        sidecar's own repo and return its resolved sha.

        The agent repo is untrusted: hooks are disabled and ``file://`` forces an
        object copy (no hardlink/alternates) so the fetched commit is fully owned
        by the sidecar repo and tamper-evident.
        """
        workspace = self.engine.evaluator.workspace
        root = workspace.root
        target = ref or "HEAD"
        fetch = await workspace.sandbox.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "protocol.file.allow=always",
                "-C",
                root,
                "fetch",
                "--no-tags",
                "--no-recurse-submodules",
                f"file://{self.agent_repo_path}",
                target,
            ],
            timeout=120,
        )
        if fetch.returncode != 0:
            raise CommitTransferError(
                f"git fetch of {target!r} from agent repo failed: {fetch.stderr}"
            )
        rev = await workspace.sandbox.run(
            ["git", "-C", root, "rev-parse", "FETCH_HEAD"], timeout=30
        )
        if rev.returncode != 0:
            raise CommitTransferError(f"rev-parse FETCH_HEAD failed: {rev.stderr}")
        return rev.stdout.strip()

    def _route_results(self, experiment: Experiment, *, admin: bool) -> str | None:
        """Write the agent-visible projection of an experiment by split tier.

        Full per-sample results always live admin-side (the session store). Here we
        write only what the agent may see:
          - visible:      aggregate summary + full per-sample results
          - non_viewable: aggregate summary only (no per-sample / no labels)
          - no_access:    nothing
        Admin/verifier evals never write to the agent volume.
        Returns the agent-volume path written, or None.
        """
        if admin:
            return None
        split = experiment.run.dataset_subset.split
        tier = tier_for_split(split, self.split_accesses)
        if tier == SplitAccessLevel.no_access:
            return None

        commit = experiment.run.candidate.commit
        dest = self.agent_volume / "results" / f"{split}__{commit[:12]}"
        # Recreate the dir so it reflects exactly this metered run. The dir is keyed
        # only on (split, commit[:12]); a prior eval of the same commit on a larger
        # sample set would otherwise leave stale per-sample files behind that this
        # run did not produce, and result_path would surface them as if they were.
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        # Aggregate summary is label-safe for both visible and partial tiers.
        (dest / "summary.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "commit": commit,
                    "n_samples": len(experiment.result.sample_results),
                    "mean_score": experiment.result.score(),
                    "status": str(experiment.result.status),
                },
                indent=2,
            )
        )
        if tier == SplitAccessLevel.viewable:
            for sample_id, sample_result in experiment.result.sample_results.items():
                (dest / f"{sample_id}.json").write_text(
                    sample_result.model_dump_json(indent=2)
                )
        return str(dest)
