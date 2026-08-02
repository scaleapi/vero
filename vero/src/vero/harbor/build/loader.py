"""Reading a build.yaml: parameter substitution, then path resolution.

Two things happen before HarborBuildConfig ever sees the document. ``${NAME}``
placeholders are resolved from --param and the environment, which is how one
checked-in benchmark config serves several run-time configurations. Then the
handful of fields that name host paths are resolved relative to the YAML file,
so a config can be written with paths relative to itself.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from vero.harbor.build.config import HarborBuildConfig

_BUILD_PARAM = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::(-|\?)([^}]*))?\}")


def _substitute_build_param(text: str, context: dict[str, str]) -> str:
    """Resolve ``${NAME}`` / ``${NAME:-default}`` / ``${NAME:?message}`` in one scalar."""

    def replace(match: re.Match[str]) -> str:
        name, operator, argument = match.group(1), match.group(2), match.group(3)
        resolved = context.get(name)
        if resolved:
            return resolved
        if operator == "-":
            return argument or ""
        if operator == "?":
            raise ValueError(
                f"required build parameter {name!r} is unset: "
                f"{argument or 'no message provided'}"
            )
        raise ValueError(
            f"build parameter {name!r} is unset; pass --param {name}=VALUE "
            "or set the environment variable"
        )

    return _BUILD_PARAM.sub(replace, text)


def _resolve_build_params(value: object, context: dict[str, str]) -> object:
    """Recursively resolve ``${...}`` placeholders in string scalars of a YAML value."""
    if isinstance(value, str):
        return _substitute_build_param(value, context)
    if isinstance(value, dict):
        return {
            key: _resolve_build_params(item, context) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_build_params(item, context) for item in value]
    return value


def _read_partition_files(
    partition_files: object,
    base: Path,
) -> dict[str, list[str]]:
    """Expand the ``partition_files`` shorthand into inline partitions.

    A partition can hold hundreds of task names, so benchmarks keep them in JSON
    files beside the config rather than inline in the YAML.
    """
    if not isinstance(partition_files, dict) or not partition_files:
        raise ValueError("partition_files must be a non-empty YAML object")
    partitions: dict[str, list[str]] = {}
    for partition, filename in partition_files.items():
        if not isinstance(partition, str) or not isinstance(filename, str):
            raise ValueError("partition_files must map names to JSON files")
        partition_path = Path(filename).expanduser()
        if not partition_path.is_absolute():
            partition_path = base / partition_path
        try:
            tasks = json.loads(partition_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"partition file {partition_path} must contain valid JSON"
            ) from error
        if not isinstance(tasks, list) or any(
            not isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                f"partition file {partition_path} must be a JSON array of task names"
            )
        partitions[partition] = tasks
    return partitions


def _resolve_local_paths(value: dict, base: Path) -> None:
    """Rewrite the path-valued fields in place, relative to the config's directory."""
    agent_repo = value.get("agent_repo")
    if isinstance(agent_repo, str) and not Path(agent_repo).is_absolute():
        value["agent_repo"] = str((base / agent_repo).resolve())
    task_source = value.get("task_source")
    if isinstance(task_source, str):
        local_source = base / task_source
        # Left alone when no such directory exists: it is a registry reference,
        # not a path.
        if not Path(task_source).is_absolute() and local_source.exists():
            value["task_source"] = str(local_source.resolve())
    task_manifest = value.get("task_manifest")
    if isinstance(task_manifest, str) and not Path(task_manifest).is_absolute():
        value["task_manifest"] = str((base / task_manifest).resolve())
    instruction_template = value.get("instruction_template")
    if isinstance(instruction_template, str) and not Path(
        instruction_template
    ).is_absolute():
        value["instruction_template"] = str((base / instruction_template).resolve())
    command_backend = value.get("command_backend")
    if isinstance(command_backend, dict):
        harness_source = command_backend.get("harness_source")
        if isinstance(harness_source, str) and not Path(harness_source).is_absolute():
            command_backend["harness_source"] = str((base / harness_source).resolve())
    overlays = value.get("workspace_overlays")
    if isinstance(overlays, list):
        for entry in overlays:
            source = entry.get("source") if isinstance(entry, dict) else None
            if isinstance(source, str) and not Path(source).is_absolute():
                entry["source"] = str((base / source).resolve())


def load_harbor_build_config(
    path: Path | str,
    *,
    params: dict[str, str] | None = None,
) -> HarborBuildConfig:
    """Load YAML and resolve local paths relative to the configuration file.

    ``${NAME}`` placeholders in the YAML are substituted at load time from
    ``params`` (explicit, e.g. ``--param NAME=VALUE``) layered over the process
    environment, so run-time knobs (optimizer model, inner sandbox provider,
    concurrency, ...) can be varied without rebuilding the task. Use
    ``${NAME:-default}`` for a fallback and ``${NAME:?message}`` to require a
    value. Fields left un-templated stay fixed (the reproducible measurement
    substrate).
    """
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "install scale-vero[harbor] to load Harbor builds"
        ) from error

    config_path = Path(path).expanduser().resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Harbor build config must be a YAML object")
    context = {**os.environ, **(params or {})}
    value = _resolve_build_params(value, context)
    base = config_path.parent
    partition_files = value.pop("partition_files", None)
    if partition_files is not None:
        if "partitions" in value:
            raise ValueError("use either partitions or partition_files, not both")
        value["partitions"] = _read_partition_files(partition_files, base)
    _resolve_local_paths(value, base)
    return HarborBuildConfig.model_validate(value)
