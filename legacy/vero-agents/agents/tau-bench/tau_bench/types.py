# Copyright Sierra

from typing import Any

from pydantic import BaseModel

from tau_bench.constants import DEFAULT_AGENT_STRATEGY

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


class Action(BaseModel):
    name: str
    kwargs: dict[str, Any]


class Task(BaseModel):
    user_id: str
    actions: list[Action]
    instruction: str
    outputs: list[str]


class RewardOutputInfo(BaseModel):
    r_outputs: float
    outputs: dict[str, bool]


class RewardActionInfo(BaseModel):
    r_actions: float
    gt_data_hash: str


class RewardResult(BaseModel):
    reward: float
    info: RewardOutputInfo | RewardActionInfo
    actions: list[Action]


class SolveResult(BaseModel):
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    total_cost: float | None = None


class EnvInfo(BaseModel):
    task: Task
    source: str | None = None
    user_cost: float | None = None
    reward_info: RewardResult | None = None


class EnvResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    observation: str
    info: EnvInfo


class EnvRunResult(BaseModel):
    task_id: int
    reward: float
    info: dict[str, Any]
    traj: list[dict[str, Any]]
    trial: int


class RunConfig(BaseModel):
    model_provider: str
    user_model_provider: str
    model: str
    user_model: str = "gpt-4.1-mini"
    num_trials: int = 1
    env: str = "retail"
    agent_strategy: str = DEFAULT_AGENT_STRATEGY
    temperature: float = 0.0
    task_split: str = "test"
    start_index: int = 0
    end_index: int = -1
    task_ids: list[int] | None = None
    log_dir: str = "results"
    max_concurrency: int = 20
    task_timeout: float | None = 600  # timeout in seconds per task, None = no timeout
    verbose: bool = False
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    few_shot_displays_path: str | None = None
