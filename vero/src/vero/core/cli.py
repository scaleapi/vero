import json
import subprocess
from pathlib import Path

import click
import toml

from .constants import PACKAGE_DIR, SCAFFOLDS_DIR


def _get_package_name() -> str | None:
    """Check we're in a uv project and return the package name.

    Returns:
        Package name on success, None on failure (error is printed).
    """
    if not Path("pyproject.toml").exists():
        click.echo(
            "Error: Not in a uv project directory. Please run this command from a directory containing a pyproject.toml"
        )
        return None

    with open("pyproject.toml", "r") as f:
        toml_data = toml.load(f)

    package_name = toml_data.get("project", {}).get("name")

    if not package_name:
        click.echo("Error: Could not find package name in pyproject.toml")
        return None

    return package_name


def _add_vero_dependency(use_pypi: bool) -> int:
    """Add scale-vero as a dev dependency.

    Args:
        use_pypi: If True, install from PyPI. Otherwise, try editable install from source.

    Returns:
        0 on success, 1 on failure.
    """
    if use_pypi:
        subprocess.run(
            ["uv", "add", "--dev", "scale-vero"], check=True, capture_output=True
        )
        click.echo("✅ Added scale-vero from PyPI")
        return 0

    if Path.cwd() == PACKAGE_DIR:
        click.echo(
            "✅ scale-vero is already available (running from source in the project directory)"
        )
        return 0

    try:
        subprocess.run(
            ["uv", "add", "--dev", "--editable", str(PACKAGE_DIR)],
            check=True,
            capture_output=True,
        )
        click.echo("✅ Added scale-vero from source (editable)")
        return 0
    except subprocess.CalledProcessError as e:
        click.echo(f"⚠️  Failed to add scale-vero as editable: {e}")
        if click.confirm("Do you want to add scale-vero from PyPI instead?"):
            subprocess.run(
                ["uv", "add", "--dev", "scale-vero"], check=True, capture_output=True
            )
            click.echo("✅ Added scale-vero from PyPI")
            return 0
        else:
            click.echo("⚠️  Failed to add scale-vero")
            return 1


@click.group()
def main():
    """A CLI tool for running vero end-to-end including test suite setup."""
    from vero.logging import setup_logging

    setup_logging()


@main.group()
def init():
    """Initialize evaluation scaffolds for your uv project."""
    pass


# =============================================================================
# Session commands
# =============================================================================


@main.group()
def session():
    """Manage and inspect optimization sessions."""
    pass


@session.command(name="list")
def session_list():
    """List all sessions."""
    from vero.core.sessions import get_vero_home_dir

    sessions_dir = get_vero_home_dir() / "sessions"

    if not sessions_dir.exists():
        click.echo("No sessions directory found.")
        return

    sessions = sorted(d.name for d in sessions_dir.iterdir() if d.is_dir())
    if not sessions:
        click.echo("No sessions found.")
        return

    click.echo(f"Sessions ({len(sessions)}):")
    for s in sessions:
        session_dir = sessions_dir / s
        config_path = session_dir / "config.json"
        suffix = ""
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                task = config.get("task") or "?"
                base = config.get("base_commit", "?")[:8]
                suffix = f"  task={task}  base={base}"
            except Exception:
                pass
        click.echo(f"  {s}{suffix}")

    click.echo(f"\nLocation: {sessions_dir}")


@session.command(name="inspect")
@click.argument("session_id")
def session_inspect(session_id: str):
    """Inspect a session: config, evaluations, and objective values."""
    from vero.core.sessions import get_vero_home_dir

    session_dir = get_vero_home_dir() / "sessions" / session_id
    if not session_dir.exists():
        click.echo(f"Session not found: {session_id}")
        return

    # Config
    config_path = session_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        click.echo("Config:")
        click.echo(f"  session_id:     {config.get('session_id', '?')}")
        click.echo(
            f"  base_commit:    {config.get('base_version', config.get('base_commit', '?'))}"
        )
        click.echo(f"  current_commit: {config.get('current_commit', '?')}")
        click.echo(f"  base_branch:    {config.get('base_branch', '?')}")
        click.echo(f"  task:           {config.get('task') or '(not set)'}")
        if config.get("model"):
            click.echo(f"  model:          {config['model']}")
        elif config.get("claude_agent_options", {}).get("model"):
            click.echo(f"  model:          {config['claude_agent_options']['model']}")
        if config.get("metadata"):
            for k, v in config["metadata"].items():
                click.echo(f"  metadata.{k}: {v}")
    else:
        click.echo("Config: (not found)")

    # Schema-v2 evaluations and schema-v1 compatibility records share the
    # stable historical directory name.
    experiments_dir = session_dir / "experiments"
    if experiments_dir.exists():
        experiment_ids = sorted(d.name for d in experiments_dir.iterdir() if d.is_dir())
        click.echo(f"\nEvaluations ({len(experiment_ids)}):")

        for exp_id in experiment_ids:
            exp_dir = experiments_dir / exp_id
            canonical_path = exp_dir / "evaluation.json"
            meta_path = exp_dir / "result_metadata.json"
            params_path = exp_dir / "evaluation_parameters.json"

            info_parts = [f"  {exp_id[:12]}"]

            if canonical_path.exists():
                try:
                    manifest = json.loads(canonical_path.read_text())
                    request = manifest.get("request", {})
                    candidate = request.get("candidate", {})
                    evaluation_set = request.get("evaluation_set", {})
                    report = manifest.get("report", {})
                    objective = manifest.get("objective") or {}
                    info_parts.append(
                        f"commit={str(candidate.get('commit', '?'))[:8]}"
                    )
                    info_parts.append(
                        f"set={evaluation_set.get('name', '?')}"
                    )
                    if evaluation_set.get("partition") is not None:
                        info_parts.append(
                            f"partition={evaluation_set['partition']}"
                        )
                    info_parts.append(f"status={report.get('status', '?')}")
                    info_parts.append(
                        f"cases={len(manifest.get('case_files', []))}"
                    )
                    if objective.get("value") is not None:
                        info_parts.append(f"objective={objective['value']:.3f}")
                    if objective.get("feasible") is not None:
                        info_parts.append(f"feasible={objective['feasible']}")
                except Exception:
                    info_parts.append("invalid-canonical-manifest")
            elif params_path.exists():
                try:
                    params = json.loads(params_path.read_text())
                    run = params.get("run", {})
                    commit = run.get("candidate", {}).get("commit", "?")[:8]
                    split = run.get("dataset_subset", {}).get("split", "?")
                    info_parts.append(f"commit={commit}")
                    info_parts.append(f"split={split}")
                except Exception:
                    pass

            if not canonical_path.exists() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    status = meta.get("status", "?")
                    info_parts.append(f"status={status}")
                except Exception:
                    pass

            # Count samples
            samples_dir = exp_dir / "samples"
            if not canonical_path.exists() and samples_dir.exists():
                n_samples = sum(1 for f in samples_dir.iterdir() if f.suffix == ".json")
                info_parts.append(f"samples={n_samples}")

                # Compute score from samples
                scores = []
                errors = 0
                for sample_file in sorted(samples_dir.iterdir()):
                    if sample_file.suffix != ".json":
                        continue
                    try:
                        sample = json.loads(sample_file.read_text())
                        score = sample.get("score")
                        if score is not None:
                            scores.append(float(score))
                        if sample.get("error"):
                            errors += 1
                    except Exception:
                        pass

                if scores:
                    mean = sum(scores) / len(scores)
                    info_parts.append(f"score={mean:.3f}")
                if errors:
                    info_parts.append(f"errors={errors}")

            click.echo("  ".join(info_parts))
    else:
        click.echo("\nExperiments: (none)")

    # Files
    click.echo("\nFiles:")
    for f in sorted(session_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            click.echo(f"  {f.name} ({size:,} bytes)")
        elif f.is_dir():
            n_items = sum(1 for _ in f.rglob("*") if _.is_file())
            click.echo(f"  {f.name}/ ({n_items} files)")


@session.command(name="clear")
@click.argument("session_ids", nargs=-1)
@click.option("--all", "clear_all", is_flag=True, help="Clear all sessions")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def session_clear(session_ids: tuple[str, ...], clear_all: bool, yes: bool):
    """Clear sessions by ID, or all sessions with --all."""
    import shutil

    from vero.core.sessions import get_vero_home_dir

    sessions_dir = get_vero_home_dir() / "sessions"

    if not sessions_dir.exists():
        click.echo("No sessions directory found.")
        return

    if clear_all:
        if not yes:
            if not click.confirm(f"Clear all sessions at {sessions_dir}?"):
                click.echo("Aborted.")
                return
        shutil.rmtree(sessions_dir)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        click.echo("Cleared all sessions.")
    elif session_ids:
        for sid in session_ids:
            session_dir = sessions_dir / sid
            if session_dir.exists():
                shutil.rmtree(session_dir)
                click.echo(f"Cleared session {sid}")
            else:
                click.echo(f"Session not found: {sid}")
    else:
        click.echo("Specify session IDs or use --all.")


# =============================================================================
# Dataset commands
# =============================================================================


@main.group()
def dataset():
    """Manage the dataset cache."""
    pass


@dataset.command(name="list")
def dataset_list():
    """List cached datasets across all sessions."""
    from vero.core.sessions import get_vero_home_dir

    vero_home = get_vero_home_dir()
    sessions_dir = vero_home / "sessions"
    dataset_cache = vero_home / "datasets"

    # Collect dataset mappings from all sessions
    datasets: dict[str, dict] = {}  # dataset_id -> {fingerprint, sessions, size}

    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            mapping_path = session_dir / "datasets.json"
            if mapping_path.exists():
                try:
                    mapping = json.loads(mapping_path.read_text())
                    for dataset_id, fingerprint in mapping.items():
                        if dataset_id not in datasets:
                            datasets[dataset_id] = {"fingerprint": fingerprint, "sessions": []}
                        datasets[dataset_id]["sessions"].append(session_dir.name[:12])
                except Exception:
                    pass

    # Also list cache entries
    cache_entries = set()
    if dataset_cache.exists():
        cache_entries = {d.name for d in dataset_cache.iterdir() if d.is_dir()}

    if not datasets and not cache_entries:
        click.echo("No cached datasets found.")
        return

    if datasets:
        click.echo(f"Datasets ({len(datasets)}):")
        for dataset_id, info in sorted(datasets.items()):
            fp = info["fingerprint"][:12]
            n_sessions = len(info["sessions"])
            cache_path = dataset_cache / info["fingerprint"]
            size = ""
            if cache_path.exists():
                total_bytes = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                size = f"  {total_bytes / 1024 / 1024:.1f}MB"
            click.echo(f"  {dataset_id}  fp={fp}  sessions={n_sessions}{size}")

    # Orphaned cache entries (not referenced by any session)
    referenced_fps = {info["fingerprint"] for info in datasets.values()}
    orphaned = cache_entries - referenced_fps
    if orphaned:
        click.echo(f"\nOrphaned cache entries ({len(orphaned)}):")
        for fp in sorted(orphaned):
            cache_path = dataset_cache / fp
            total_bytes = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            click.echo(f"  {fp[:12]}  {total_bytes / 1024 / 1024:.1f}MB")

    click.echo(f"\nCache location: {dataset_cache}")


@dataset.command(name="inspect")
@click.argument("dataset_id")
@click.option("--session", "session_id", default=None, help="Session ID to look up dataset in")
def dataset_inspect(dataset_id: str, session_id: str | None):
    """Inspect a cached dataset: splits, columns, row counts."""
    from vero.core.sessions import get_vero_home_dir

    vero_home = get_vero_home_dir()
    sessions_dir = vero_home / "sessions"
    dataset_cache = vero_home / "datasets"

    # Find the fingerprint
    fingerprint = None

    if session_id:
        mapping_path = sessions_dir / session_id / "datasets.json"
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text())
            fingerprint = mapping.get(dataset_id)
    else:
        # Search all sessions
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                mapping_path = session_dir / "datasets.json"
                if mapping_path.exists():
                    try:
                        mapping = json.loads(mapping_path.read_text())
                        if dataset_id in mapping:
                            fingerprint = mapping[dataset_id]
                            break
                    except Exception:
                        pass

    if fingerprint is None:
        click.echo(f"Dataset '{dataset_id}' not found in any session.")
        return

    cache_path = dataset_cache / fingerprint
    if not cache_path.exists():
        click.echo(f"Cache entry missing: {fingerprint}")
        return

    try:
        from datasets import DatasetDict

        ds = DatasetDict.load_from_disk(str(cache_path))
    except Exception as e:
        click.echo(f"Failed to load dataset: {e}")
        return

    click.echo(f"Dataset: {dataset_id}")
    click.echo(f"Fingerprint: {fingerprint}")
    click.echo(f"Cache path: {cache_path}")
    click.echo(f"\nSplits ({len(ds)}):")
    for split_name, split_ds in ds.items():
        click.echo(f"  {split_name}: {len(split_ds)} rows")
        click.echo(f"    Columns: {', '.join(split_ds.column_names)}")
        # Show first row preview
        if len(split_ds) > 0:
            row = split_ds[0]
            click.echo("    First row:")
            for col, val in row.items():
                val_str = str(val)
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                click.echo(f"      {col}: {val_str}")


@dataset.command(name="clear")
@click.argument("dataset_ids", nargs=-1)
@click.option("--all", "clear_all", is_flag=True, help="Clear entire dataset cache")
@click.option("--orphaned", is_flag=True, help="Clear only orphaned cache entries")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def dataset_clear(dataset_ids: tuple[str, ...], clear_all: bool, orphaned: bool, yes: bool):
    """Clear cached datasets by ID, orphaned entries, or everything."""
    import shutil

    from vero.core.sessions import get_vero_home_dir

    vero_home = get_vero_home_dir()
    sessions_dir = vero_home / "sessions"
    dataset_cache = vero_home / "datasets"

    if not dataset_cache.exists():
        click.echo("No dataset cache found.")
        return

    if clear_all:
        if not yes:
            if not click.confirm(f"Clear entire dataset cache at {dataset_cache}?"):
                click.echo("Aborted.")
                return
        shutil.rmtree(dataset_cache)
        dataset_cache.mkdir(parents=True, exist_ok=True)
        click.echo("Cleared dataset cache.")

    elif orphaned:
        # Find referenced fingerprints
        referenced = set()
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                mapping_path = session_dir / "datasets.json"
                if mapping_path.exists():
                    try:
                        mapping = json.loads(mapping_path.read_text())
                        referenced.update(mapping.values())
                    except Exception:
                        pass

        removed = 0
        for entry in dataset_cache.iterdir():
            if entry.is_dir() and entry.name not in referenced:
                shutil.rmtree(entry)
                removed += 1
                click.echo(f"Cleared orphaned entry: {entry.name[:12]}")

        click.echo(f"Cleared {removed} orphaned entries.")

    elif dataset_ids:
        # Find fingerprints for the given dataset IDs
        fp_map: dict[str, str] = {}
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                mapping_path = session_dir / "datasets.json"
                if mapping_path.exists():
                    try:
                        mapping = json.loads(mapping_path.read_text())
                        for did, fp in mapping.items():
                            if did in dataset_ids:
                                fp_map[did] = fp
                    except Exception:
                        pass

        for did in dataset_ids:
            fp = fp_map.get(did)
            if fp:
                cache_path = dataset_cache / fp
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    click.echo(f"Cleared dataset '{did}' (fp={fp[:12]})")
                else:
                    click.echo(f"Cache entry not found for '{did}'")
            else:
                click.echo(f"Dataset '{did}' not found in any session mapping.")
    else:
        click.echo("Specify dataset IDs, --orphaned, or --all.")


# =============================================================================
# Check command
# =============================================================================


@main.command()
@click.option(
    "--project-path",
    type=click.Path(exists=True),
    default=".",
    help="Path to agent project (default: current directory)",
)
@click.option(
    "--task", type=str, default=None, help="Task name to validate (default: check all discovered tasks)"
)
@click.option(
    "--dataset",
    "--dataset-path",
    "dataset_path",
    type=str,
    default=None,
    help="Path to dataset (optional — validates splits if provided)",
)
@click.option(
    "--task-project",
    type=click.Path(exists=True),
    default=None,
    help="Separate uv project for task/eval code",
)
@click.option(
    "--task-module",
    type=str,
    default=None,
    help="Explicit Python module for task registration",
)
def check(
    project_path: str,
    task: str | None,
    dataset_path: str | None,
    task_project: str | None,
    task_module: str | None,
):
    """Validate project setup without running inference.

    Checks: uv project, git repo, task discovery, required env vars, dataset.
    Fast, no LLM calls, no evaluation.

    \b
    Examples:
      vero check
      vero check --project-path ./my-agent --task main
      vero check --project-path ./my-agent --task main --dataset ./data
    """
    import asyncio
    import os

    errors = []
    warnings = []

    # 1. Project — uv package?
    pyproject = Path(project_path) / "pyproject.toml"
    if not pyproject.exists():
        errors.append(f"No pyproject.toml found in {project_path}")
        click.echo("  [FAIL] Not a uv project (no pyproject.toml)")
    else:
        click.echo("  [OK]   uv project found")

    # 2. Git repo?
    import subprocess as _sp

    result = _sp.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project_path, capture_output=True, text=True,
    )
    if result.returncode != 0:
        errors.append(f"Not a git repository: {project_path}")
        click.echo("  [FAIL] Not a git repository")
    else:
        click.echo(f"  [OK]   Git repo: {result.stdout.strip()}")

    # 3. Task discovery
    if errors:
        click.echo("\n  Skipping task discovery (project issues above)")
    else:
        from vero.evaluator import Evaluator
        from vero.workspace.git import GitWorkspace

        async def _discover():
            workspace = await GitWorkspace.create(str(project_path))
            evaluator = Evaluator(
                workspace=workspace,
                session_id="check",
                task_project=Path(task_project) if task_project else None,
                task_module=task_module,
            )
            return await evaluator._discover_tasks(workspace.project_path)

        try:
            discovery = asyncio.run(_discover())
            tasks = discovery.get("tasks", {})
            package = discovery.get("package", "?")
            click.echo(f"  [OK]   Task discovery: {package} ({len(tasks)} task(s))")

            for name, info in tasks.items():
                has_inf = info.get("has_inference", False)
                has_eval = info.get("has_evaluation", False)
                status = "OK" if has_inf and has_eval else "WARN"
                missing = []
                if not has_inf:
                    missing.append("inference")
                if not has_eval:
                    missing.append("evaluation")
                suffix = f" (missing: {', '.join(missing)})" if missing else ""
                click.echo(f"         - {name}: {status}{suffix}")

                if status == "WARN":
                    warnings.append(f"Task '{name}' missing {', '.join(missing)}")

            # Validate requested task exists
            if task and task not in tasks:
                errors.append(f"Task '{task}' not found. Available: {list(tasks.keys())}")
                click.echo(f"  [FAIL] Task '{task}' not found")

            # 4. Required env vars
            check_tasks = [task] if task else list(tasks.keys())
            for t in check_tasks:
                required = tasks.get(t, {}).get("required_env_vars", [])
                if required:
                    missing_env = [v for v in required if not os.environ.get(v)]
                    if missing_env:
                        errors.append(f"Task '{t}' requires: {', '.join(missing_env)}")
                        click.echo(f"  [FAIL] Missing env vars for '{t}': {', '.join(missing_env)}")
                    else:
                        click.echo(f"  [OK]   Env vars for '{t}': all set")

        except Exception as e:
            errors.append(f"Task discovery failed: {e}")
            click.echo(f"  [FAIL] Task discovery failed: {e}")

    # 5. Dataset
    if dataset_path:
        try:
            path = Path(dataset_path)
            if path.exists():
                from datasets import DatasetDict

                ds = DatasetDict.load_from_disk(str(path))
                splits = list(ds.keys())
                sizes = {s: len(ds[s]) for s in splits}
                click.echo(f"  [OK]   Dataset: {splits} {sizes}")
            else:
                warnings.append(f"Dataset path does not exist: {dataset_path}")
                click.echo(f"  [WARN] Dataset path not found: {dataset_path}")
        except Exception as e:
            errors.append(f"Dataset load failed: {e}")
            click.echo(f"  [FAIL] Dataset: {e}")

    # Summary
    click.echo("")
    if errors:
        click.echo(f"RESULT: {len(errors)} error(s), {len(warnings)} warning(s)")
        raise SystemExit(1)
    elif warnings:
        click.echo(f"RESULT: OK with {len(warnings)} warning(s)")
    else:
        click.echo("RESULT: All checks passed")


# =============================================================================
# Evaluate command
# =============================================================================


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Generic program configuration (defaults to ./vero.toml when present)",
)
@click.option(
    "--project-path",
    type=click.Path(exists=True),
    required=False,
    help="Path to agent project",
)
@click.option(
    "--task", type=str, required=False, help="Task name from vero_tasks module"
)
@click.option(
    "--dataset",
    "--dataset-path",
    "dataset_path",
    type=str,
    required=False,
    help="Path to dataset (or dataset ID)",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "validation"]),
    required=False,
    help="Dataset split",
)
@click.option("--commit", type=str, default=None, help="Git commit to evaluate")
@click.option(
    "--sample-ids",
    type=str,
    default=None,
    callback=lambda ctx, param, v: (
        [int(x.strip()) for x in v.split(",") if x.strip()] if v else None
    ),
    help="Comma-separated sample IDs",
)
@click.option(
    "--num-samples", type=int, default=None, help="Number of samples to evaluate"
)
@click.option(
    "--task-params",
    type=str,
    default=None,
    callback=lambda ctx, param, v: __import__("json").loads(v) if v else None,
    help="JSON string of task-specific parameters",
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--timeout", type=int, default=3600, help="Timeout in seconds")
@click.option(
    "--per-sample-timeout", type=int, default=180, help="Timeout per sample in seconds"
)
@click.option(
    "--create-temporary-worktree", is_flag=True, help="Create a temporary worktree"
)
@click.option(
    "--isolate",
    is_flag=True,
    help="Copy the project into a fresh git repo before evaluating (useful for monorepos or dirty working trees)",
)
@click.option(
    "--max-concurrency", type=int, default=None, help="Maximum concurrent tasks"
)
@click.option(
    "--task-project",
    type=click.Path(exists=True),
    default=None,
    help="Separate uv project for task/eval code",
)
@click.option(
    "--task-module",
    type=str,
    default=None,
    help="Explicit Python module for task registration (e.g. my_eval_tasks.vero_tasks)",
)
def evaluate(
    config_path: Path | None,
    project_path: Path | None,
    task: str | None,
    dataset_path: Path | None,
    split: str | None,
    commit: str | None = None,
    sample_ids: list[int] | None = None,
    num_samples: int | None = None,
    task_params: dict | None = None,
    seed: int = 42,
    timeout: int = 3600,
    per_sample_timeout: int = 180,
    create_temporary_worktree: bool = False,
    isolate: bool = False,
    max_concurrency: int | None = None,
    task_project: str | None = None,
    task_module: str | None = None,
):
    """Run an evaluation on an agent codebase."""
    import asyncio

    if config_path is None and project_path is None and Path("vero.toml").exists():
        config_path = Path("vero.toml")
    if config_path is not None:
        from vero.config import build_program_runtime, load_config

        async def _evaluate_program():
            runtime = await build_program_runtime(load_config(config_path))
            record = await runtime.policy.evaluate_candidate(runtime.policy.base_version)
            click.echo(f"Session ID: {runtime.session_id}")
            click.echo(f"Evaluation ID: {record.id}")
            click.echo(f"Commit: {record.request.candidate.commit}")
            click.echo(f"Status: {record.report.status.value}")
            value = record.objective.value if record.objective is not None else None
            click.echo(f"Objective: {value}")
            return record

        return asyncio.run(_evaluate_program())

    if project_path is None or task is None or dataset_path is None or split is None:
        raise click.UsageError(
            "provide --config for generic programs, or --project-path, --dataset, "
            "--task, and --split for VeroTask evaluation"
        )

    from vero.evaluator import run_evaluation

    asyncio.run(
        run_evaluation(
            project_path=project_path,
            dataset=str(dataset_path),
            split=split,
            task=task,
            commit=commit,
            sample_ids=sample_ids,
            num_samples=num_samples,
            task_params=task_params,
            seed=seed,
            timeout=timeout,
            per_sample_timeout=per_sample_timeout,
            create_temporary_worktree=create_temporary_worktree,
            isolate=isolate,
            max_concurrency=max_concurrency,
            task_project=task_project,
            task_module=task_module,
        )
    )


# =============================================================================
# Run command
# =============================================================================


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Generic program configuration (defaults to ./vero.toml when present)",
)
@click.option(
    "--project-path",
    type=click.Path(exists=True),
    required=False,
    help="Path to agent project",
)
@click.option(
    "--task", type=str, required=False, help="Task name from vero_tasks module"
)
@click.option(
    "--dataset",
    "--dataset-path",
    "dataset_path",
    type=str,
    required=False,
    help="Path to dataset (or dataset ID)",
)
@click.option(
    "--agent",
    type=click.Choice(["claude-code", "vero"]),
    default="claude-code",
    help="Agent backend (default: claude-code)",
)
@click.option(
    "--model",
    type=str,
    default="claude-sonnet-4-5-20250929",
    help="Model name (default: claude-sonnet-4-5-20250929)",
)
@click.option("--max-turns", type=int, default=200, help="Max optimization turns (default: 200)")
@click.option("--train-budget", type=int, default=10, help="Evaluation budget on train split (default: 10)")
@click.option("--validation-budget", type=int, default=0, help="Evaluation budget on validation split (default: 0)")
@click.option("--git-ref", type=str, default="main", help="Git ref to start from (default: main)")
@click.option("--isolate", is_flag=True, help="Copy project into a fresh git repo")
@click.option("--enable-wandb", is_flag=True, help="Enable wandb logging")
@click.option("--wandb-project", type=str, default=None, help="Wandb project name")
@click.option("--skip-initial-eval", is_flag=True, help="Skip baseline evaluation")
@click.option("--eval-split", type=str, default="test", help="Split for initial/final eval (default: test)")
@click.option(
    "--task-project",
    type=click.Path(exists=True),
    default=None,
    help="Separate uv project for task/eval code",
)
@click.option(
    "--task-module",
    type=str,
    default=None,
    help="Explicit Python module for task registration",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to .env file for the optimizer process (LLM API keys, etc.)",
)
@click.option(
    "--subprocess-env-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to .env file for evaluation subprocesses",
)
def run(
    config_path: Path | None,
    project_path: str | None,
    task: str | None,
    dataset_path: str | None,
    agent: str,
    model: str,
    max_turns: int,
    train_budget: int,
    validation_budget: int,
    git_ref: str,
    isolate: bool,
    enable_wandb: bool,
    wandb_project: str | None,
    skip_initial_eval: bool,
    eval_split: str,
    task_project: str | None,
    task_module: str | None,
    env_file: str | None,
    subprocess_env_file: str | None,
):
    """Run a full optimization loop.

    Creates a Policy with the specified agent and runs the optimization loop:
    initial eval, agent optimization steps, final eval.

    \b
    Examples:
      vero run --project-path ./my-agent --dataset-path ./data --task main
      vero run --project-path ./my-agent --dataset-path ./data --task main --agent vero --model anthropic/claude-sonnet-4-5-20250929
      vero run --project-path ./my-agent --dataset-path ./data --task main --isolate --enable-wandb
    """
    import asyncio

    if config_path is None and project_path is None and Path("vero.toml").exists():
        config_path = Path("vero.toml")
    if config_path is not None:
        from vero.config import build_program_runtime, load_config

        async def _run_program():
            runtime = await build_program_runtime(
                load_config(config_path),
                require_optimizer=True,
            )
            result = await runtime.policy.run()
            click.echo(f"Session ID: {runtime.session_id}")
            click.echo(f"Baseline commit: {result.baseline.request.candidate.commit}")
            click.echo(
                f"Baseline objective: "
                f"{result.baseline.objective.value if result.baseline.objective else None}"
            )
            click.echo(
                f"Best commit: "
                f"{result.best.request.candidate.commit if result.best else None}"
            )
            click.echo(
                f"Best objective: "
                f"{result.best.objective.value if result.best and result.best.objective else None}"
            )
            return result

        return asyncio.run(_run_program())

    if project_path is None or task is None or dataset_path is None:
        raise click.UsageError(
            "provide --config for generic programs, or --project-path, --dataset, "
            "and --task for VeroTask optimization"
        )

    from vero.policy import Policy

    if agent == "claude-code":
        from claude_agent_sdk import ClaudeAgentOptions

        from vero.agents.claude_code import ClaudeCodeAgent, default_tool_sets

        agent_instance = ClaudeCodeAgent(
            options=ClaudeAgentOptions(model=model, permission_mode="bypassPermissions"),
            tool_sets=default_tool_sets(),
        )
    else:
        from agents import Agent as OAIAgent

        from vero.agents.vero import VeroAgent
        from vero.agents.vero import default_tool_sets as vero_default_tool_sets

        agent_instance = VeroAgent(
            oai_agent=OAIAgent(name="VeroAgent", model=model),
            tool_sets=vero_default_tool_sets(),
        )

    policy = Policy(
        project_path=project_path,
        dataset=dataset_path,
        agent=agent_instance,
        task=task,
        ref=git_ref,
        isolate=isolate,
        max_turns=max_turns,
        train_budget=train_budget,
        validation_budget=validation_budget,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project or "vero",
        task_project=task_project,
        task_module=task_module,
        optimizer_env_file=env_file,
        subprocess_env_vars=subprocess_env_file,
    )

    async def _run():
        best = await policy.run(
            skip_initial_eval=skip_initial_eval,
            eval_split=eval_split,
        )
        click.echo(f"\nSession ID: {policy.session_id}")
        click.echo(f"Best commit: {best.commit}")
        click.echo(f"Best score:  {best.score}")
        return best

    asyncio.run(_run())


# =============================================================================
# Init commands
# =============================================================================


@init.command(name="accesses")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing .veroaccess file without prompting",
)
@click.option(
    "--auto",
    "mode",
    flag_value="auto",
    help="Scan project structure and generate tailored rules",
)
@click.option(
    "--interactive",
    "mode",
    flag_value="interactive",
    help="Walk through directories and choose access levels interactively",
)
@click.option(
    "--default",
    "mode",
    flag_value="default",
    default=True,
    help="Use the bundled default rules (default)",
)
def init_accesses(force: bool, mode: str):
    """Initialize a .veroaccess file for agent filesystem permissions.

    Creates a .veroaccess file in the current directory that controls what files
    the Vero agent can read, write, or must avoid. This is similar to .gitignore
    but for agent access control.

    Three modes are available:

    \b
      --default      Copy the bundled default rules (the default)
      --auto         Scan the project and generate rules based on what exists
      --interactive  Walk through each directory and choose access levels
    """
    from vero.core.constants import VEROACCESS_FILENAME

    veroaccess_path = Path(VEROACCESS_FILENAME)

    if veroaccess_path.exists() and not force:
        click.echo(f"⚠️  {VEROACCESS_FILENAME} already exists.")
        if not click.confirm("Do you want to overwrite it?"):
            click.echo("Aborted.")
            return 0

    if mode == "auto":
        content = _init_accesses_auto()
    elif mode == "interactive":
        content = _init_accesses_interactive()
    else:
        content = _init_accesses_default()

    veroaccess_path.write_text(content)

    click.echo(f"\n✅ Created {VEROACCESS_FILENAME} (mode: {mode})\n")
    click.echo("   This file controls what the Vero agent can access:")
    click.echo("   - [exclude] sections: Agent cannot access these paths")
    click.echo("   - [read] sections: Agent can read but not modify")
    click.echo(
        "   - [write] sections: Agent has full access (default for unlisted paths)"
    )
    click.echo(
        f"\n📝 Edit {VEROACCESS_FILENAME} to customize agent permissions for your project."
    )

    return 0


def _init_accesses_default() -> str:
    """Return the bundled default .veroaccess content."""
    from vero.core.constants import _DEFAULT_VEROACCESS_PATH

    return _DEFAULT_VEROACCESS_PATH.read_text()


def _init_accesses_auto() -> str:
    """Scan project structure and generate tailored .veroaccess content."""
    from vero.core.veroaccess import generate_veroaccess_auto

    project_root = Path.cwd()
    content = generate_veroaccess_auto(project_root)
    click.echo("   Scanned project structure:")
    # Show a summary of what was detected
    for line in content.splitlines():
        if line.startswith("#") or line.startswith("[") or not line.strip():
            continue
        click.echo(f"     {line}")
    return content


def _init_accesses_interactive() -> str:
    """Interactively walk directories and assign access levels."""
    from vero.core.veroaccess import _AccessEntry, _format_veroaccess
    from vero.filesystem import AccessType

    project_root = Path.cwd()

    # Gather top-level entries, dirs first then files
    dirs = sorted(
        [
            d
            for d in project_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ],
        key=lambda p: p.name,
    )
    hidden_dirs = sorted(
        [d for d in project_root.iterdir() if d.is_dir() and d.name.startswith(".")],
        key=lambda p: p.name,
    )

    entries: list[_AccessEntry] = []

    access_map = {"e": AccessType.EXCLUDE, "r": AccessType.READ, "w": AccessType.WRITE}

    click.echo("\n   For each directory, choose an access level:")
    click.echo("     [e]xclude  [r]ead  [w]rite  [s]kip (omit from rules)\n")

    for d in dirs:
        # Count children for context
        try:
            n_children = sum(1 for _ in d.iterdir())
        except PermissionError:
            n_children = 0
        prompt = f"   📁 {d.name}/ ({n_children} items)"
        choice = click.prompt(
            prompt, type=click.Choice(["e", "r", "w", "s"]), default="s"
        )

        if choice == "s":
            continue

        access_type = access_map[choice]
        entries.append(_AccessEntry(access_type, f"{d.name}/"))
        entries.append(_AccessEntry(access_type, f"{d.name}/**"))

    # Handle hidden dirs as a batch
    if hidden_dirs:
        hidden_names = ", ".join(d.name for d in hidden_dirs)
        click.echo(f"\n   Hidden directories: {hidden_names}")
        choice = click.prompt(
            "   Exclude all hidden directories?",
            type=click.Choice(["y", "n"]),
            default="y",
        )
        if choice == "y":
            for d in hidden_dirs:
                entries.append(_AccessEntry(AccessType.EXCLUDE, f"{d.name}/"))
                entries.append(_AccessEntry(AccessType.EXCLUDE, f"{d.name}/**"))

    # Always add noise patterns
    click.echo("")
    for pattern in [
        "**/__pycache__",
        "**/__pycache__/**",
        "**/.pytest_cache",
        "**/.pytest_cache/**",
    ]:
        entries.append(_AccessEntry(AccessType.EXCLUDE, pattern, "Noise"))

    # Always protect .veroaccess
    entries.append(
        _AccessEntry(AccessType.READ, ".veroaccess", "Access rules — protected")
    )

    return _format_veroaccess(entries)


@init.command(name="tasks")
@click.option("--task", type=str, default="main", help="Name of the task to create")
@click.option(
    "--use-pypi",
    is_flag=True,
    help="Install vero from PyPI instead of from a local directory",
)
def init_tasks(task: str, use_pypi: bool):
    """Initialize a vero_tasks module (recommended approach)."""

    package_name = _get_package_name()
    if not package_name:
        return 1

    # Convert package name to module name (replace hyphens with underscores)
    module_name = package_name.replace("-", "_")

    # Find the source directory (look for src/<module_name> or <module_name>)
    src_dir = Path("src") / module_name
    if not src_dir.exists():
        src_dir = Path(module_name)
        if not src_dir.exists():
            click.echo(
                f"Error: Could not find package directory. Tried 'src/{module_name}' and '{module_name}'"
            )
            return 1

    # Create vero_tasks directory
    vero_tasks_dir = src_dir / "vero_tasks"
    if vero_tasks_dir.exists():
        click.echo(f"vero_tasks directory already exists at {vero_tasks_dir}")
        if (vero_tasks_dir / f"{task}.py").exists():
            click.echo(f"Task '{task}' already exists. Please choose a different name.")
            return 1
    else:
        vero_tasks_dir.mkdir(exist_ok=True)

    # Create __init__.py that imports the task
    init_file = vero_tasks_dir / "__init__.py"
    if init_file.exists():
        # Append import to existing __init__.py
        existing_content = init_file.read_text()
        if f"from . import {task}" not in existing_content:
            with open(init_file, "a") as f:
                f.write(f"from . import {task}  # noqa: F401\n")
            click.echo(f"   Updated {init_file} with import for '{task}'")
    else:
        init_content = f'''"""VeroTask definitions for {module_name}."""

# Import task modules to register them
from . import {task}  # noqa: F401
'''
        init_file.write_text(init_content)

    # Create the task file from scaffold
    scaffold_src = SCAFFOLDS_DIR / "vero_tasks.py"
    task_file = vero_tasks_dir / f"{task}.py"

    if not scaffold_src.exists():
        click.echo(f"Error: Scaffold file not found at {scaffold_src}")
        return 1

    scaffold_content = scaffold_src.read_text()
    # Replace default task name if different from "main"
    if task != "main":
        task_content = scaffold_content.replace(
            'create_task("main")', f'create_task("{task}")'
        )
    else:
        task_content = scaffold_content
    task_file.write_text(task_content)

    click.echo(f"\n✅ Successfully initialized vero_tasks with task '{task}':\n")
    click.echo(f"   - {vero_tasks_dir}/__init__.py")
    click.echo(f"   - {vero_tasks_dir}/{task}.py")

    click.echo("\n📝 Next steps:")
    click.echo(
        f"   1. Edit {task_file} to implement your inference and evaluation logic"
    )
    click.echo(f'   2. Use task="{task}" in Policy')

    result = _add_vero_dependency(use_pypi)
    if result != 0:
        return result

    click.echo("")
    return 0


if __name__ == "__main__":
    main()
