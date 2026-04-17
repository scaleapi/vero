# Copyright Sierra

import asyncio
import json
import multiprocessing
import os
import random
import traceback
from copy import deepcopy
from datetime import datetime
from math import comb

from litellm import provider_list
from tqdm import tqdm

from tau_bench.agents import agent_factory
from tau_bench.envs import get_env
from tau_bench.envs.user import UserStrategy
from tau_bench.types import EnvRunResult, RunConfig


async def run(config: RunConfig) -> list[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_{time_str}.json"
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=deepcopy(
            config
        ),  # this ensures that attributes unrelated to the agent strategy are not modified by the agent factory
    )
    end_index = len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    results: list[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})")
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int, trial: int = i) -> EnvRunResult:
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            if config.verbose:
                print(f"Running task {idx}")
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=trial,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=trial,
                )
            if config.verbose:
                print(
                    "✅" if result.reward == 1 else "❌",
                    f"task_id={idx}",
                    result.info,
                )
                print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path) as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        current_results: dict[int, EnvRunResult] = {}

        semaphore = asyncio.Semaphore(config.max_concurrency)

        async def run_with_timeout(
            idx: int, trial: int = i, sem: asyncio.Semaphore = semaphore
        ) -> tuple[int, EnvRunResult]:
            async with sem:
                loop = asyncio.get_running_loop()
                coro = loop.run_in_executor(None, _run, idx)
                try:
                    if config.task_timeout is not None:
                        result = await asyncio.wait_for(coro, timeout=config.task_timeout)
                    else:
                        result = await coro
                except TimeoutError:
                    result = EnvRunResult(
                        task_id=idx,
                        reward=0.0,
                        info={"error": "timeout", "timeout_seconds": config.task_timeout},
                        traj=[],
                        trial=trial,
                    )
                    if config.verbose:
                        print(f"⏱️ task_id={idx} timed out after {config.task_timeout}s")
                return idx, result

        tasks = [run_with_timeout(idx) for idx in idxs]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(idxs), desc=f"Trial {i+1}/{config.num_trials}"):
            idx, result = await coro
            current_results[idx] = result

        results.extend([current_results[idx] for idx in idxs])

    if config.verbose:
        display_metrics(results)

    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


def display_metrics(results: list[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len({r.trial for r in results})
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
