"""`vero harbor serve` — the eval-sidecar entrypoint.

Assembles the EvaluationEngine + EvaluationSidecar + Verifier from a ServeConfig
(written by the compiler, baked into the sidecar image), generates the per-trial admin
token, and serves the FastAPI app under uvicorn. ServeConfig is the compiler↔serve
contract.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from vero.core.budget import BudgetLedger, SplitBudget
from vero.core.dataset.base import SplitAccess, SplitAccessLevel
from vero.core.db.database import ExperimentDatabase
from vero.core.evaluation import BaseEvaluationParameters
from vero.core.sessions import get_vero_home_dir
from vero.evaluation.engine import EvaluationEngine
from vero.evaluation.evaluator import Evaluator
from vero.harbor.app import create_app
from vero.harbor.auth import generate_token, write_admin_token
from vero.harbor.server import EvaluationSidecar
from vero.harbor.verifier import VerificationTarget, Verifier
from vero.workspace.git import GitWorkspace

logger = logging.getLogger(__name__)


class _SplitAccessCfg(BaseModel):
    split: str
    access: str  # "viewable" | "non_viewable" | "no_access"


class _TargetCfg(BaseModel):
    task: str | None = None
    dataset_id: str
    split: str
    reward_key: str = "reward"
    sample_ids: list[int] | None = None
    # Executor-model override for this target (transfer probe; Mode B only).
    model: str | None = None


class _ServeConfigBase(BaseModel):
    """Fields shared by both serve modes. The compiled twin of BuildConfig's
    shared base. Not instantiated directly."""

    # extra="forbid" so a wrong-mode key (e.g. Mode-B feedback levers in a Mode-A
    # serve.json) is a load-time error rather than a silently-ignored no-op; the
    # Mode-A / Mode-B split makes those keys unknown to the resolved variant.
    model_config = ConfigDict(extra="forbid")

    repo_path: str            # sidecar's own repo (baseline target) = the engine workspace
    agent_repo_path: str      # mounted agent workspace (commit-transfer source)
    session_id: str
    dataset_id: str           # already registered in the sidecar's VERO_HOME
    split_accesses: list[_SplitAccessCfg]
    budgets: list[dict]       # SplitBudget kwargs

    # selection / reward
    reward_mode: Literal["submit", "auto_best"] = "auto_best"
    selection_split: str = "validation"
    targets: list[_TargetCfg] = Field(default_factory=list)
    base_commit: str | None = None
    submit_enabled: bool = False
    # Also admin-score the unmodified baseline on every target at finalize and
    # write it to <admin_volume>/baseline.json: makes regressions visible
    # (an optimized candidate can score WORSE than the untouched baseline).
    score_baseline: bool = False
    # Total attempts for the finalize baseline eval (>=1): a transient nested-run
    # failure once silently dropped the regression check.
    baseline_score_attempts: int = 2
    # auto_best never ships a candidate that fails to beat the untouched baseline
    # on the selection split; it reverts to base_commit instead (needs base_commit).
    auto_best_baseline_floor: bool = True

    # Minimum sample count for agent-chosen subset evals of non_viewable
    # splits (full-split evals always pass; <=1 disables). See
    # EvaluationSidecar.k_anonymity_floor for the leak this closes.
    k_anonymity_floor: int = 5

    # Consumed at COMPILE time (the instruction's exhaust-budget bullet);
    # recorded here so serve.json mirrors build.yaml, like instruct_multifidelity.
    instruct_exhaust_budget: bool = True

    # volumes / token
    agent_volume: str
    admin_volume: str
    admin_token_path: str

    # eval params
    timeout: int = 600
    max_concurrency: int = 20
    use_copy: bool = True  # isolate each eval in a temp copy (clean tree, concurrency-safe)

    host: str = "0.0.0.0"
    port: int = 8000


class ServeConfigA(_ServeConfigBase):
    """Mode A: vero runs inference + scoring."""

    mode: Literal["A"] = "A"

    task: str | None = None
    task_project: str | None = None
    task_module: str | None = None
    # Per-sample vero-scoring cap. Mode-A only (Mode B uses each nested task's
    # own harbor-configured timeouts, capped only by `timeout`).
    sample_timeout: int = 180


class ServeConfigB(_ServeConfigBase):
    """Mode B: scoring runs in a nested `harbor run`."""

    mode: Literal["B"]

    harbor: dict | None = None  # HarborConfig kwargs
    # Lever 1: failed samples carry their trial-transcript tail in the per-sample
    # `feedback` field. Exposure stays gated by the sidecar's tier routing
    # (per-sample files are written only for viewable splits).
    feedback_transcripts: bool = False
    feedback_max_bytes: int = 3000
    # Lever 3: sample output carries a per-attempt {reward, exception} list. Same
    # viewable-only exposure path as feedback_transcripts.
    expose_attempt_detail: bool = False
    # Lever 2: consumed at COMPILE time (the instruction's multi-fidelity
    # section); recorded here so serve.json mirrors build.yaml. The sidecar's
    # subset-eval support itself is unconditional (EvalRequest.num_samples /
    # sample_ids), so there is nothing to toggle at serve time.
    instruct_multifidelity: bool = False


# Discriminated union on `mode`, the compiled twin of BuildConfig.
ServeConfig = Annotated[ServeConfigA | ServeConfigB, Field(discriminator="mode")]

_ServeConfigAdapter: TypeAdapter[ServeConfigA | ServeConfigB] = TypeAdapter(ServeConfig)


def load_serve_config(path: Path | str) -> ServeConfigA | ServeConfigB:
    """Load and validate a serve.json into the correct Mode-A / Mode-B variant."""
    data = json.loads(Path(path).read_text())
    # `mode` defaults to "A" when absent (older serve.json predates the tag). The
    # discriminated union needs the tag explicitly, so inject it before validating.
    if isinstance(data, dict):
        data.setdefault("mode", "A")
    return _ServeConfigAdapter.validate_python(data)


def _load_or_build_ledger(
    budget_cfgs: list[dict], persist_path: Path
) -> BudgetLedger:
    """Build the durable ledger, reloading spent budget from ``persist_path`` if present.

    The ledger flushes every mutation to ``persist_path``; without reloading it,
    a sidecar restart would reset all spent budget to full, letting the agent
    regain its full evaluation budget by triggering a restart. On startup we
    reconstruct each SplitBudget and restore its persisted ``remaining_*`` values.

    A MISSING file is a fresh boot: configured budgets. A file that exists but
    cannot be parsed fails CLOSED: metered budgets restore with zero remaining.
    The old fallback (configured budgets) refunded the agent everything already
    spent, so any crash that corrupted the flush minted budget; spend that
    cannot be read must be treated as fully spent, never as never-happened.
    """
    if persist_path.exists():
        try:
            persisted = json.loads(persist_path.read_text())
            budgets: list[SplitBudget] = []
            for entry in persisted:
                b = SplitBudget(
                    split=entry["split"],
                    dataset_id=entry.get("dataset_id", ""),
                    total_sample_budget=entry.get("total_sample_budget"),
                    total_run_budget=entry.get("total_run_budget"),
                    max_samples_per_run=entry.get("max_samples_per_run"),
                )
                # __post_init__ reset remaining_* to total_*; restore spent state.
                b.remaining_sample_budget = entry.get("remaining_sample_budget")
                b.remaining_run_budget = entry.get("remaining_run_budget")
                budgets.append(b)
            logger.info(
                "Reloaded persisted budget ledger from %s (%d splits).",
                persist_path,
                len(budgets),
            )
            return BudgetLedger(budgets, persist_path=persist_path)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error(
                "Persisted ledger %s exists but is unreadable (%s); failing "
                "CLOSED: metered agent budgets restore as exhausted. Admin and "
                "finalize are unaffected. The unreadable file is preserved at "
                "%s; delete ledger.json deliberately to boot fresh.",
                persist_path,
                e,
                persist_path.with_suffix(".corrupt"),
            )
            try:  # keep the evidence: the next flush overwrites persist_path
                shutil.copyfile(persist_path, persist_path.with_suffix(".corrupt"))
            except OSError:
                pass
            budgets = []
            for cfg in budget_cfgs:
                b = SplitBudget(**cfg)
                if b.total_sample_budget is not None:
                    b.remaining_sample_budget = 0
                if b.total_run_budget is not None:
                    b.remaining_run_budget = 0
                budgets.append(b)
            return BudgetLedger(budgets, persist_path=persist_path)
    return BudgetLedger(
        [SplitBudget(**b) for b in budget_cfgs], persist_path=persist_path
    )


# PR #20's _warn_mode_b_sample_timeout and _warn_mode_a_ignores_feedback_levers
# are superseded by the Mode-A / Mode-B type split: sample_timeout no longer
# exists on Mode B, and the feedback levers no longer exist on Mode A, so both
# warned-about conditions are structurally impossible (a ValidationError at load).


async def build_components(
    config: ServeConfigA | ServeConfigB,
) -> tuple[EvaluationSidecar, Verifier, str]:
    """Assemble the sidecar + verifier (sharing one engine) and the admin token."""
    vero_home = get_vero_home_dir()

    # Integrity guard: in Mode A the vero scorer (@task.evaluation()) is resolved
    # and executed by the Evaluator. With task_project unset, that scorer would be
    # discovered from the AGENT's committed repo, so a committed scorer returning
    # 1.0 would win the hidden-split/admin reward. Require a sidecar-baked task
    # project so the scorer is trusted (agent code is layered as --with-editable,
    # never the scorer's source). Mode B uses an eval_strategy that ignores the
    # vero scorer and is exempt.
    if isinstance(config, ServeConfigA) and not config.task_project:
        raise ValueError(
            "Mode A requires `task_project` so the scorer is loaded from the "
            "sidecar-baked task project, not the agent's committed repo. Refusing "
            "to start: with task_project unset the agent controls its own scoring."
        )

    workspace = await GitWorkspace.create(config.repo_path)

    persist_path = Path(config.admin_volume) / "ledger.json"
    budget = _load_or_build_ledger(config.budgets, persist_path)

    eval_strategy = None
    if isinstance(config, ServeConfigB) and config.harbor is not None:
        from vero.harbor.runner import HarborRunner
        from vero.harbor.config import HarborConfig

        eval_strategy = HarborRunner(
            HarborConfig(**config.harbor),
            feedback_transcripts=config.feedback_transcripts,
            feedback_max_bytes=config.feedback_max_bytes,
            expose_attempt_detail=config.expose_attempt_detail,
        )

    is_mode_a = isinstance(config, ServeConfigA)
    evaluator = Evaluator(
        workspace,
        config.session_id,
        vero_home=vero_home,
        use_copy=config.use_copy,
        task_project=Path(config.task_project) if is_mode_a and config.task_project else None,
        task_module=config.task_module if is_mode_a else None,
        eval_strategy=eval_strategy,
    )

    split_accesses = [
        SplitAccess(split=s.split, access=SplitAccessLevel(s.access))
        for s in config.split_accesses
    ]
    db = ExperimentDatabase(id=config.session_id)  # shared by engine (writes) + verifier (reads)
    engine = EvaluationEngine(
        evaluator=evaluator,
        budget=budget,
        default_task=config.task if is_mode_a else None,
        db=db,
        run_constraints=BaseEvaluationParameters(
            timeout=config.timeout,
            sample_timeout=config.sample_timeout if is_mode_a else config.timeout,
            max_concurrency=config.max_concurrency,
        ),
        session_id=config.session_id,
        vero_home=vero_home,
        # The engine-side no_access gate is only armed when split_accesses is
        # set. Without it the ledger was the sole gate (no_access splits are
        # unbudgeted, so reserve() raised) — and every unmetered path (admin,
        # the free baseline eval) walked straight past it.
        split_accesses=split_accesses,
    )
    sidecar = EvaluationSidecar(
        engine=engine,
        split_accesses=split_accesses,
        agent_repo_path=Path(config.agent_repo_path),
        agent_volume=Path(config.agent_volume),
        admin_volume=Path(config.admin_volume),
        submit_enabled=config.submit_enabled,
        base_commit=config.base_commit,
        k_anonymity_floor=config.k_anonymity_floor,
    )
    verifier = Verifier(
        engine=engine,
        admin_volume=Path(config.admin_volume),
        reward_mode=config.reward_mode,
        targets=[VerificationTarget(**t.model_dump()) for t in config.targets],
        selection_split=config.selection_split,
        base_commit=config.base_commit,
        selection_task=config.task if is_mode_a else None,
        selection_dataset_id=config.dataset_id,
        score_baseline=config.score_baseline,
        baseline_score_attempts=config.baseline_score_attempts,
        auto_best_baseline_floor=config.auto_best_baseline_floor,
    )

    token = generate_token()
    write_admin_token(config.admin_token_path, token)
    return sidecar, verifier, token


async def build_app(config: ServeConfigA | ServeConfigB):
    sidecar, verifier, token = await build_components(config)
    return create_app(sidecar=sidecar, verifier=verifier, admin_token=token)


def serve(config_path: Path | str) -> None:
    """Sidecar entrypoint: build the app and run it under uvicorn."""
    import asyncio

    import uvicorn

    config = load_serve_config(config_path)
    app = asyncio.run(build_app(config))
    logger.info(f"Serving eval sidecar on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)
