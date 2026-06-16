"""`vero harbor serve` — the eval-sidecar entrypoint.

Assembles the EvaluationEngine + EvaluationSidecar + Verifier from a ServeConfig
(written by the compiler, baked into the sidecar image), generates the per-trial admin
token, and serves the FastAPI app under uvicorn. ServeConfig is the compiler↔serve
contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    reward_mode: str = "auto_best"
    selection_split: str = "validation"
    targets: list[_TargetCfg] = Field(default_factory=list)
    base_commit: str | None = None
    submit_enabled: bool = False

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


async def build_components(config: ServeConfig) -> tuple[EvaluationSidecar, Verifier, str]:
    """Assemble the sidecar + verifier (sharing one engine) and the admin token."""
    vero_home = get_vero_home_dir()
    workspace = await GitWorkspace.create(config.repo_path)

    budget = BudgetLedger(
        [SplitBudget(**b) for b in config.budgets],
        persist_path=Path(config.admin_volume) / "ledger.json",
    )

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
    )
    verifier = Verifier(
        engine=engine,
        admin_volume=Path(config.admin_volume),
        reward_mode=config.reward_mode,  # type: ignore[arg-type]
        targets=[VerificationTarget(**t.model_dump()) for t in config.targets],
        selection_split=config.selection_split,
        base_commit=config.base_commit,
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
