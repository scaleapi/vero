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


class KAnonymityError(RuntimeError):
    """Raised when an agent eval on a non_viewable split selects fewer samples
    than the k-anonymity floor allows."""


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
        base_commit: str | None = None,
        k_anonymity_floor: int = 5,
    ):
        self.engine = engine
        self.split_accesses = split_accesses
        self.agent_repo_path = Path(agent_repo_path)
        self.agent_volume = Path(agent_volume)
        self.admin_volume = Path(admin_volume)
        self.submit_enabled = submit_enabled
        self.base_commit = base_commit
        # Minimum sample count for an agent-chosen SUBSET eval of a non_viewable
        # split. The aggregate response carries mean_score, so a singleton subset
        # returns the sample's label-derived score verbatim, and n singleton
        # evals reconstruct the split's per-sample labels wholesale. The floor
        # applies only to proper subsets (sample_ids/num_samples): a full-split
        # eval reveals exactly the intended aggregate, so it always passes and a
        # split smaller than the floor degrades to full-split-only rather than
        # becoming unevaluable. This is a cost multiplier, not a proof: means of
        # k-sized overlapping subsets still admit reconstruction by elimination,
        # but at >= k times the sample budget per label. <= 1 disables the floor.
        self.k_anonymity_floor = k_anonymity_floor
        self._free_baseline_used = False
        self._eval_seq = 0  # per-eval ordinal for result-dir versioning

    # ------------------------------------------------------------------
    # Handlers (the HTTP layer resolves `admin` from auth and calls these)
    # ------------------------------------------------------------------

    async def evaluate(self, req: EvalRequest, *, admin: bool = False) -> EvalSummary:
        # k-anonymity floor, checked before any work (no commit transfer, no
        # budget debit, no eval) so a rejected request costs the agent nothing.
        # sample_ids is None exactly when the request covers the full split
        # (resolve_samples collapses a covering num_samples to None too), and a
        # full-split aggregate is the intended surface, so only proper subsets
        # are floored.
        if not admin and self.k_anonymity_floor > 1:
            tier = tier_for_split(req.split, self.split_accesses)
            if tier == SplitAccessLevel.non_viewable:
                sample_ids, n = self.engine.resolve_samples(req)
                if sample_ids is not None and n < self.k_anonymity_floor:
                    raise KAnonymityError(
                        f"Evals on non_viewable split '{req.split}' must cover "
                        f"at least {self.k_anonymity_floor} samples (or the "
                        f"whole split); got {n}. Aggregate scores over smaller "
                        f"subsets would reveal per-sample results."
                    )
        sha = await self._transfer_commit(req.commit)
        # The agent's FIRST eval of the seeded baseline is budget-free. The
        # baseline is the reference every candidate is implicitly compared to,
        # yet it can never win selection (auto_best excludes base_commit), so
        # metering it forces a choice between optimizing blind and paying a
        # budgeted eval for a commit that cannot be selected (observed live:
        # an optimizer that skipped the reference could not tell a no-op edit
        # from an improvement and quit with budget unspent). Capped at one:
        # later baseline evals debit normally, so free compute is bounded.
        # `free` waives only the budget debit; the eval still runs as the agent
        # (tier gates apply), so the free baseline cannot touch no_access
        # splits — riding the admin flag here did exactly that.
        free_baseline = (
            not admin
            and self.base_commit is not None
            and sha == self.base_commit
            and not self._free_baseline_used
        )
        exp = await self.engine.evaluate(
            replace(req, commit=sha), admin=admin, free=free_baseline
        )
        # Consume the freebie only after the eval succeeded: an eval that
        # raised (invalid split, infra failure) has given the agent nothing,
        # so it must not burn the one free reference measurement.
        if free_baseline:
            self._free_baseline_used = True
        # Route with the agent's real tier even when the eval was unmetered.
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
            base_commit=self.base_commit,
            free_baseline_available=(
                self.base_commit is not None and not self._free_baseline_used
            ),
            k_anonymity_floor=self.k_anonymity_floor,
        )

    def list_experiments(self) -> list[dict]:
        """All recorded experiments, unredacted (admin observability).

        One row per experiment: commit, dataset/split, recorded score, and
        creation time. Lets an operator (or the outer harness) watch an
        optimization run mid-flight without exec-ing into the container or
        waiting for finalize. Admin-only: recorded scores on non_viewable
        splits must not reach the agent.
        """
        import pandas as pd

        if self.engine.db is None:
            return []
        # fill_score=None: errored samples must surface as null, not as the
        # minimum-score floor: a synthetic 0.0 is indistinguishable from a real
        # measured failure when watching the trajectory. error_rate carries the
        # failure signal explicitly.
        df = self.engine.db.get_experiments_df(fill_score=None)
        if df.empty:
            return []

        def _clean(v):
            return None if v is None or pd.isna(v) else float(v)

        rows = []
        for _, r in df.iterrows():
            created = r.get("candidate_created_at")
            rows.append(
                {
                    "commit": r.get("candidate_commit"),
                    "dataset_id": r.get("dataset_subset_dataset_id"),
                    "split": r.get("dataset_subset_split"),
                    "mean_score": _clean(r.get("mean_score")),
                    "error_rate": _clean(r.get("error_rate")),
                    "created_at": (
                        created.isoformat()
                        if created is not None
                        and str(created) not in ("NaT", "None")
                        and hasattr(created, "isoformat")
                        else None
                    ),
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
        # Every metered eval gets its own versioned dir. Keying on
        # (split, commit) alone forced a wipe-and-rewrite, so a re-measurement
        # (a multifidelity confirm, a noise re-eval of the champion) erased the
        # agent's earlier evidence for the same commit; repeat measurements are
        # exactly the ones worth comparing. result_path in the response names
        # the dir for THIS eval.
        self._eval_seq += 1
        dest = (
            self.agent_volume
            / "results"
            / f"{split}__{commit[:12]}__e{self._eval_seq}"
        )
        if dest.exists():  # ordinal collision only on volume reuse; never merge
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        # Aggregate summary is label-safe for both visible and partial tiers.
        # n_scored / n_errored / score_se qualify the mean: a mean over 3
        # scored samples of 18, or one dominated by errored zero-fills, is a
        # different measurement than a clean full-split mean, and the agent
        # (and any auditor) should see that without per-sample access.
        sample_results = experiment.result.sample_results
        filled = [
            r.score if r.score is not None else 0.0
            for r in sample_results.values()
        ]
        score_se = None
        if len(filled) > 1:
            m = sum(filled) / len(filled)
            var = sum((x - m) ** 2 for x in filled) / (len(filled) - 1)
            score_se = (var / len(filled)) ** 0.5
        (dest / "summary.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "commit": commit,
                    "n_samples": len(sample_results),
                    "n_scored": sum(
                        1 for r in sample_results.values() if r.score is not None
                    ),
                    "n_errored": sum(
                        1 for r in sample_results.values() if r.is_error()
                    ),
                    "mean_score": experiment.result.score(),
                    "score_se": score_se,
                    "status": experiment.result.status.value,
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
