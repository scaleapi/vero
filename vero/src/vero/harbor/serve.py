"""`vero harbor serve` — the eval-sidecar entrypoint.

Assembles the EvaluationEngine + EvaluationSidecar + Verifier from a ServeConfig
(written by the compiler, baked into the sidecar image), generates the per-trial admin
token, and serves the FastAPI app under uvicorn. ServeConfig is the compiler↔serve
contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

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


class ServeConfig(BaseModel):
    """Everything the sidecar needs to assemble itself. Baked by the compiler."""

    repo_path: str            # sidecar's own repo (baseline target) = the engine workspace
    agent_repo_path: str      # mounted agent workspace (commit-transfer source)
    session_id: str
    dataset_id: str           # already registered in the sidecar's VERO_HOME
    split_accesses: list[_SplitAccessCfg]
    budgets: list[dict]       # SplitBudget kwargs

    # Mode A
    task: str | None = None
    task_project: str | None = None
    task_module: str | None = None
    # Mode B
    harbor: dict | None = None  # HarborConfig kwargs

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

    # volumes / token
    agent_volume: str
    admin_volume: str
    admin_token_path: str

    # eval params
    timeout: int = 600
    sample_timeout: int = 180
    max_concurrency: int = 20
    use_copy: bool = True  # isolate each eval in a temp copy (clean tree, concurrency-safe)

    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_file(cls, path: Path | str) -> ServeConfig:
        return cls.model_validate_json(Path(path).read_text())


def _load_or_build_ledger(
    budget_cfgs: list[dict], persist_path: Path
) -> BudgetLedger:
    """Build the durable ledger, reloading spent budget from ``persist_path`` if present.

    The ledger flushes every mutation to ``persist_path``; without reloading it,
    a sidecar restart would reset all spent budget to full, letting the agent
    regain its full evaluation budget by triggering a restart. On startup we
    reconstruct each SplitBudget and restore its persisted ``remaining_*`` values.
    Falls back to the configured budgets if the file is missing or unreadable
    (fail-safe to the configured budget, never to unlimited).
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
            logger.warning(
                "Could not reload persisted ledger %s (%s); using configured budgets.",
                persist_path,
                e,
            )
    return BudgetLedger(
        [SplitBudget(**b) for b in budget_cfgs], persist_path=persist_path
    )


def _warn_mode_b_sample_timeout(config: ServeConfig) -> None:
    """sample_timeout only governs Mode A (per-sample vero scoring). In Mode B
    the nested `harbor run` applies each task's OWN harbor-configured timeouts;
    the only vero-side cap is `timeout` on the whole nested run. An author who
    set sample_timeout expecting a per-task cap would silently get none: say so.
    """
    if config.harbor is not None and "sample_timeout" in config.model_fields_set:
        logger.warning(
            "sample_timeout=%s is not enforced in Mode B: nested `harbor run` "
            "tasks use their harbor-configured timeouts (tune via "
            "harbor.extra_args, e.g. --agent-timeout-multiplier); only "
            "`timeout` (%ss) caps the whole nested run.",
            config.sample_timeout,
            config.timeout,
        )


async def build_components(config: ServeConfig) -> tuple[EvaluationSidecar, Verifier, str]:
    """Assemble the sidecar + verifier (sharing one engine) and the admin token."""
    vero_home = get_vero_home_dir()

    # Integrity guard: in Mode A the vero scorer (@task.evaluation()) is resolved
    # and executed by the Evaluator. With task_project unset, that scorer would be
    # discovered from the AGENT's committed repo, so a committed scorer returning
    # 1.0 would win the hidden-split/admin reward. Require a sidecar-baked task
    # project so the scorer is trusted (agent code is layered as --with-editable,
    # never the scorer's source). Mode B (config.harbor set) uses an eval_strategy
    # that ignores the vero scorer and is exempt.
    if config.harbor is None and not config.task_project:
        raise ValueError(
            "Mode A requires `task_project` so the scorer is loaded from the "
            "sidecar-baked task project, not the agent's committed repo. Refusing "
            "to start: with task_project unset the agent controls its own scoring."
        )

    _warn_mode_b_sample_timeout(config)

    workspace = await GitWorkspace.create(config.repo_path)

    persist_path = Path(config.admin_volume) / "ledger.json"
    budget = _load_or_build_ledger(config.budgets, persist_path)

    eval_strategy = None
    if config.harbor is not None:
        from vero.harbor.runner import HarborRunner
        from vero.harbor.config import HarborConfig

        eval_strategy = HarborRunner(HarborConfig(**config.harbor))

    evaluator = Evaluator(
        workspace,
        config.session_id,
        vero_home=vero_home,
        use_copy=config.use_copy,
        task_project=Path(config.task_project) if config.task_project else None,
        task_module=config.task_module,
        eval_strategy=eval_strategy,
    )

    db = ExperimentDatabase(id=config.session_id)  # shared by engine (writes) + verifier (reads)
    engine = EvaluationEngine(
        evaluator=evaluator,
        budget=budget,
        default_task=config.task,
        db=db,
        run_constraints=BaseEvaluationParameters(
            timeout=config.timeout,
            sample_timeout=config.sample_timeout,
            max_concurrency=config.max_concurrency,
        ),
        session_id=config.session_id,
        vero_home=vero_home,
    )

    split_accesses = [
        SplitAccess(split=s.split, access=SplitAccessLevel(s.access))
        for s in config.split_accesses
    ]
    sidecar = EvaluationSidecar(
        engine=engine,
        split_accesses=split_accesses,
        agent_repo_path=Path(config.agent_repo_path),
        agent_volume=Path(config.agent_volume),
        admin_volume=Path(config.admin_volume),
        submit_enabled=config.submit_enabled,
        base_commit=config.base_commit,
    )
    verifier = Verifier(
        engine=engine,
        admin_volume=Path(config.admin_volume),
        reward_mode=config.reward_mode,
        targets=[VerificationTarget(**t.model_dump()) for t in config.targets],
        selection_split=config.selection_split,
        base_commit=config.base_commit,
        selection_task=config.task,
        selection_dataset_id=config.dataset_id,
        score_baseline=config.score_baseline,
    )

    token = generate_token()
    write_admin_token(config.admin_token_path, token)
    return sidecar, verifier, token


async def build_app(config: ServeConfig):
    sidecar, verifier, token = await build_components(config)
    return create_app(sidecar=sidecar, verifier=verifier, admin_token=token)


def serve(config_path: Path | str) -> None:
    """Sidecar entrypoint: build the app and run it under uvicorn."""
    import asyncio

    import uvicorn

    config = ServeConfig.from_file(config_path)
    app = asyncio.run(build_app(config))
    logger.info(f"Serving eval sidecar on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)
