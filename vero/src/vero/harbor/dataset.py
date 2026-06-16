"""Build the vero dataset (task-name references + split partition) for Mode B.

A Mode-B vero dataset has no labels — each "sample" is a Harbor task name. A local
task's name is its subdirectory name (the dir containing ``task.toml``), matching what
``harbor run -i/--include-task-name`` filters on; registry task names come from the
registry's task configs.

The split partition is a ``dict[str, list[str]]`` (e.g. ``{"train": [...], "test": [...]}``)
supplied by the benchmark author; this module compiles + validates it into a DatasetDict.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict


def build_harbor_dataset(partition: dict[str, list[str]]) -> DatasetDict:
    """Compile a ``{split: [task_names]}`` partition into a vero DatasetDict.

    Each split is a single-column (`task_name`) Dataset — the label-free sample
    references Mode B evaluates.
    """
    from datasets import Dataset, DatasetDict

    if not partition:
        raise ValueError("Harbor dataset partition is empty.")
    return DatasetDict(
        {split: Dataset.from_dict({"task_name": list(names)}) for split, names in partition.items()}
    )


def enumerate_local_task_names(task_source: str | Path) -> list[str]:
    """Task names available in a local Harbor task source.

    If the path is itself a task dir (contains ``task.toml``), returns ``[dir_name]``;
    otherwise returns the names of immediate subdirectories that contain ``task.toml``.
    """
    path = Path(task_source).expanduser()
    if (path / "task.toml").exists():
        return [path.name]
    if not path.is_dir():
        raise ValueError(f"Local task source is not a directory: {path}")
    return sorted(
        d.name for d in path.iterdir() if d.is_dir() and (d / "task.toml").exists()
    )


async def enumerate_registry_task_names(
    ref: str, *, registry_url: str | None = None
) -> list[str]:
    """Task names in a registry dataset (``org/name[@version]``).

    Lazy-imports the ``harbor`` SDK (the ``harbor`` extra) — registry resolution is a
    build-time concern, not a sidecar-runtime one. Integration-verified.
    """
    from harbor.models.job.config import RegistryDatasetConfig
    from harbor.models.registry import RemoteRegistryInfo

    name, _, version = ref.partition("@")
    config = RegistryDatasetConfig(
        registry=RemoteRegistryInfo(url=registry_url) if registry_url else None,
        name=name,
        version=version or None,
    )
    return sorted(tc.path.name for tc in await config.get_task_configs())


def validate_partition(partition: dict[str, list[str]], available: list[str]) -> None:
    """Raise if the partition references task names not in ``available``."""
    avail = set(available)
    referenced = {name for names in partition.values() for name in names}
    unknown = referenced - avail
    if unknown:
        raise ValueError(
            f"Partition references task names not found in the source: {sorted(unknown)}"
        )
