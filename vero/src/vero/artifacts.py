from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vero.core.db.database import Experiment
    from vero.policy import Policy
    from vero.sandbox import Sandbox

logger = logging.getLogger(__name__)


@dataclass
class FileSystemArtifact(ABC):
    """An artifact that gets materialized into _vero/ in the agent's workspace."""

    @abstractmethod
    async def on_init(self, policy: Policy, dest: str, sandbox: Sandbox) -> None:
        """Called during Policy.init(). dest is the _vero/ path inside the sandbox."""
        pass

    @abstractmethod
    async def on_experiment(self, policy: Policy, experiment: Experiment, dest: str, sandbox: Sandbox) -> None:
        """Called after each evaluate_commit(). dest is the _vero/ path inside the sandbox."""
        pass


@dataclass
class DatasetArtifact(FileSystemArtifact):
    """Materializes viewable dataset splits as per-sample JSON files."""

    async def on_init(self, policy: Policy, dest: str, sandbox: Sandbox) -> None:
        from vero.core.dataset import get_non_viewable_splits
        from vero.core.dataset.store import list_datasets
        from vero.core.dataset.store import load_dataset as store_load_dataset

        non_viewable = get_non_viewable_splits(policy.split_accesses)
        datasets_dir = f"{dest}/datasets"

        for ds_id in list_datasets(policy.sessions_dir, policy.session_id):
            dataset = store_load_dataset(policy.sessions_dir, policy.dataset_cache, policy.session_id, ds_id)
            for split_name, split_data in dataset.items():
                if split_name in non_viewable:
                    continue
                split_dir = f"{datasets_dir}/{ds_id}/{split_name}"
                await sandbox.mkdir(split_dir)
                for i, sample in enumerate(split_data):
                    await sandbox.write_file(
                        f"{split_dir}/{i}.json",
                        json.dumps(dict(sample), indent=2, default=str),
                    )
                logger.info(f"Materialized {len(split_data)} samples to {split_dir}")

    async def on_experiment(self, policy: Policy, experiment: Experiment, dest: str, sandbox: Sandbox) -> None:
        pass


@dataclass
class RawDatasetArtifact(FileSystemArtifact):
    """Copies raw HF dataset dirs (for code that calls load_from_disk)."""

    async def on_init(self, policy: Policy, dest: str, sandbox: Sandbox) -> None:
        from vero.core.dataset.store import _read_mapping

        mapping = _read_mapping(policy.sessions_dir, policy.session_id)
        if mapping:
            datasets_dst = f"{dest}/datasets"
            await sandbox.mkdir(datasets_dst)
            for ds_id, fp in mapping.items():
                cache_path = str(policy.dataset_cache / fp)
                dst = f"{datasets_dst}/{ds_id}"
                if not await sandbox.exists(dst):
                    await sandbox.upload(cache_path, dst)
                    logger.info(f"Copied raw dataset '{ds_id}' to {dst}")

    async def on_experiment(self, policy: Policy, experiment: Experiment, dest: str, sandbox: Sandbox) -> None:
        pass


@dataclass
class SkillsArtifact(FileSystemArtifact):
    """Copies skills directories by namespace."""

    async def on_init(self, policy: Policy, dest: str, sandbox: Sandbox) -> None:
        if not policy.session or not policy.session.skills:
            return

        skills_dir = f"{dest}/skills"
        for namespace, path in policy.session.skills.items():
            dst = f"{skills_dir}/{namespace}"
            await sandbox.mkdir(dst)

            if isinstance(path, dict):
                # Inline skills: {name: content}
                for name, content in path.items():
                    await sandbox.write_file(f"{dst}/{name}.md", str(content))
            else:
                # Path-based skills: upload from host
                path_str = str(path)
                await sandbox.upload(path_str, dst)

            logger.info(f"Materialized skills '{namespace}' to {dst}")

    async def on_experiment(self, policy: Policy, experiment: Experiment, dest: str, sandbox: Sandbox) -> None:
        pass


@dataclass
class TracesArtifact(FileSystemArtifact):
    """Materializes experiment traces as JSON files after each eval."""

    async def on_init(self, policy: Policy, dest: str, sandbox: Sandbox) -> None:
        """Materialize traces from existing experiments in the DB."""
        from vero.core.db.database import Experiment

        if not policy.session or not policy.session.db:
            return
        db = policy.session.db
        for result_id, result in db.results.items():
            run = db.runs.get(result.run_id)
            if run:
                await self.on_experiment(policy, Experiment(run=run, result=result), dest, sandbox)

    async def on_experiment(self, policy: Policy, experiment: Experiment, dest: str, sandbox: Sandbox) -> None:
        from vero.core.dataset import get_non_viewable_splits

        non_viewable = get_non_viewable_splits(policy.split_accesses)
        split = experiment.run.dataset_subset.split

        if split in non_viewable:
            return

        commit = experiment.run.candidate.commit[:8]
        trace_dir = f"{dest}/traces/{split}__{commit}"
        await sandbox.mkdir(trace_dir)

        summary = {
            "experiment_id": experiment.id,
            "commit": experiment.run.candidate.commit,
            "split": split,
            "status": experiment.result.status.value,
            "score": experiment.result.score(),
            "error_rate": experiment.result.error_rate(),
            "num_samples": len(experiment.result.sample_results),
        }
        await sandbox.write_file(
            f"{trace_dir}/summary.json",
            json.dumps(summary, indent=2, default=str),
        )

        for sample_id, sample_result in experiment.result.sample_results.items():
            await sandbox.write_file(
                f"{trace_dir}/{sample_id}.json",
                sample_result.model_dump_json(indent=2),
            )

        logger.info(
            f"Materialized traces for {split}__{commit} ({len(experiment.result.sample_results)} samples)"
        )
