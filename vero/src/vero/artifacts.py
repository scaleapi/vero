from __future__ import annotations

import hashlib
import json
import logging
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vero.core.db.database import Experiment
    from vero.evaluation import EvaluationRecord
    from vero.policy import Policy
    from vero.sandbox import Sandbox

logger = logging.getLogger(__name__)


def _safe_trace_label(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if safe == value and safe not in {".", ".."}:
        return safe
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'evaluation'}-{digest}"


@dataclass
class FileSystemArtifact(ABC):
    """An artifact that gets materialized into _vero/ in the agent's workspace."""

    @abstractmethod
    async def on_init(self, policy: Policy, dest: str, sandbox: Sandbox) -> None:
        """Called during Policy.init(). dest is the _vero/ path inside the sandbox."""
        pass

    async def on_evaluation(
        self,
        policy: Policy,
        evaluation: EvaluationRecord,
        dest: str,
        sandbox: Sandbox,
    ) -> None:
        """Materialize a canonical evaluation.

        Subclasses written against the legacy ``on_experiment`` hook continue to
        work through an on-demand compatibility projection. New implementations
        should override this method.
        """
        if type(self).on_experiment is FileSystemArtifact.on_experiment:
            return
        from vero.evaluation import evaluation_record_to_experiment

        warnings.warn(
            "FileSystemArtifact.on_experiment() is deprecated; implement "
            "on_evaluation() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        await self.on_experiment(
            policy,
            evaluation_record_to_experiment(evaluation),
            dest,
            sandbox,
        )

    async def on_experiment(
        self,
        policy: Policy,
        experiment: Experiment,
        dest: str,
        sandbox: Sandbox,
    ) -> None:
        """Deprecated compatibility hook for schema-v1 artifact extensions."""


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

@dataclass
class TracesArtifact(FileSystemArtifact):
    """Materialize canonical evaluation traces as JSON files."""

    async def on_init(self, policy: Policy, dest: str, sandbox: Sandbox) -> None:
        """Materialize traces from existing canonical evaluations."""
        if policy.evaluation_db is None:
            return
        for evaluation in policy.evaluation_db.get_evaluations():
            await self.on_evaluation(policy, evaluation, dest, sandbox)

    async def on_evaluation(
        self,
        policy: Policy,
        evaluation: EvaluationRecord,
        dest: str,
        sandbox: Sandbox,
    ) -> None:
        from vero.core.dataset import get_non_viewable_splits

        non_viewable = get_non_viewable_splits(policy.split_accesses)
        evaluation_set = evaluation.request.evaluation_set
        split = evaluation_set.partition

        if split is not None and split in non_viewable:
            return

        label = _safe_trace_label(split or evaluation_set.name)
        commit = evaluation.request.candidate.commit[:8]
        trace_dir = f"{dest}/traces/{label}__{commit}"
        await sandbox.mkdir(trace_dir)

        summary = {
            "evaluation_id": evaluation.id,
            "commit": evaluation.request.candidate.commit,
            "evaluation_set": evaluation_set.model_dump(mode="json"),
            "status": evaluation.report.status.value,
            "metrics": evaluation.report.metrics,
            "objective": evaluation.objective.model_dump(mode="json")
            if evaluation.objective is not None
            else None,
            "num_cases": len(evaluation.report.cases),
        }
        await sandbox.write_file(
            f"{trace_dir}/summary.json",
            json.dumps(summary, indent=2, default=str),
        )

        for case in evaluation.report.cases:
            case_filename = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()
            await sandbox.write_file(
                f"{trace_dir}/{case_filename}.json",
                case.model_dump_json(indent=2),
            )

        logger.info(
            "Materialized traces for %s__%s (%s cases)",
            label,
            commit,
            len(evaluation.report.cases),
        )

    async def on_experiment(
        self,
        policy: Policy,
        experiment: Experiment,
        dest: str,
        sandbox: Sandbox,
    ) -> None:
        """Preserve the schema-v1 filesystem shape for explicit legacy calls."""
        from vero.core.dataset import get_non_viewable_splits

        split = experiment.run.dataset_subset.split
        if split in get_non_viewable_splits(policy.split_accesses):
            return
        commit = experiment.run.candidate.commit[:8]
        trace_dir = f"{dest}/traces/{_safe_trace_label(split)}__{commit}"
        await sandbox.mkdir(trace_dir)
        await sandbox.write_file(
            f"{trace_dir}/summary.json",
            json.dumps(
                {
                    "experiment_id": experiment.id,
                    "commit": experiment.run.candidate.commit,
                    "split": split,
                    "status": experiment.result.status.value,
                    "score": experiment.result.score(),
                    "error_rate": experiment.result.error_rate(),
                    "num_samples": len(experiment.result.sample_results),
                },
                indent=2,
                default=str,
            ),
        )
        for sample_id, sample_result in experiment.result.sample_results.items():
            await sandbox.write_file(
                f"{trace_dir}/{sample_id}.json",
                sample_result.model_dump_json(indent=2),
            )
