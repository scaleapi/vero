"""The `vero harbor build` compiler: BuildConfig -> a runnable Harbor task dir.

Emits the environment (optimizer workbench `main` + eval `eval-sidecar`), the
protocol (instruction.md), the verifier (tests/test.sh -> `vero harbor finalize`),
and bakes the ServeConfig + dataset + baseline repo + vero source. The result runs
with `harbor run -p <task-dir> -a <optimizer> -m <model> -e docker`.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from vero.evaluation.engine import EvalRequest
from vero.harbor.build.config import BuildConfig
from vero.harbor.protocol import StatusSummary

logger = logging.getLogger(__name__)

_TEMPLATES = Path(__file__).parent / "templates"

# Container paths (must match the templates).
VERO_DIR = "/opt/vero"
AGENT_BASELINE = "/opt/agent-baseline"  # sidecar engine workspace
WORK_AGENT = "/work/agent"  # shared agent repo (main rw, sidecar ro)
VERO_HOME = "/opt/vero_home"
INNER_TASK = "/opt/inner-task"  # Mode B: baked inner Harbor task (the protected benchmark)
SERVE_JSON = "/opt/serve.json"
ADMIN_VOLUME = "/state/admin"
AGENT_VOLUME = "/state/agent-results"
TOKEN_PATH = "/state/token/admin.token"
SESSION_ID = "trial"

# vero source items copied into the build context (enough to `uv pip install`).
_VERO_COPY = ["pyproject.toml", "README.md", "uv.lock", "src"]


def _render(env: Environment, template_name: str, dest: Path, **ctx) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(env.get_template(template_name).render(**ctx))


def _copy_vero_source(vero_root: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in _VERO_COPY:
        src = vero_root / item
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest / item, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / item)


def _rewrite_vero_source_path(pyproject: Path) -> None:
    """Point a relative `scale-vero` path dependency at the baked /opt/vero so it
    resolves regardless of where the repo (or a temp worktree of it) lives."""
    if not pyproject.exists():
        return
    text = pyproject.read_text()
    new = re.sub(
        r'(scale-vero\s*=\s*\{[^}]*?path\s*=\s*")[^"]*(")',
        rf"\g<1>{VERO_DIR}\g<2>",
        text,
    )
    if new != text:
        pyproject.write_text(new)
        logger.info("Rewrote scale-vero source path -> %s", VERO_DIR)


def _prepare_baseline_repo(agent_repo: Path, dest: Path) -> str:
    """Materialize the target repo at HEAD into a clean standalone git repo
    (vero path rewritten) and return its commit sha. Copied verbatim (incl. .git)
    into both the sidecar (engine workspace) and main (seed), so they share a sha."""
    dest.mkdir(parents=True, exist_ok=True)
    toplevel = subprocess.run(
        ["git", "-C", str(agent_repo), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if toplevel.returncode == 0:
        # Extract only the target subtree at HEAD (the repo may be a monorepo and
        # agent_repo a subdirectory of it), stripping the leading path components.
        repo_root = Path(toplevel.stdout.strip())
        rel = agent_repo.relative_to(repo_root)
        strip = len(rel.parts)
        archive = subprocess.Popen(
            ["git", "-C", str(repo_root), "archive", "HEAD", str(rel)]
            if strip else ["git", "-C", str(repo_root), "archive", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            subprocess.run(
                ["tar", "xf", "-", "--strip-components", str(strip)],
                cwd=dest, stdin=archive.stdout, check=True,
            )
        finally:
            # Let git see SIGPIPE if tar died, then reap it (no zombie).
            if archive.stdout is not None:
                archive.stdout.close()
            archive_err = (archive.communicate()[1] or b"").decode(errors="replace")
        # A failed `git archive` can emit a truncated stream that `tar` still
        # accepts with exit 0, baking a near-empty baseline. Fail loudly instead.
        if archive.returncode != 0:
            raise RuntimeError(
                f"git archive failed (exit {archive.returncode}) for {repo_root}: "
                f"{archive_err.strip()}"
            )
    else:
        shutil.copytree(agent_repo, dest, dirs_exist_ok=True)

    _rewrite_vero_source_path(dest / "pyproject.toml")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=vero", "-c", "user.email=vero@localhost",
             "-C", str(dest), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "baseline")
    return git("rev-parse", "HEAD")


def _register(dataset, vero_home: Path, tmp: Path) -> str:
    """Register a dataset (path/DatasetDict) into a baked VERO_HOME; return dataset_id."""
    from vero.core.dataset.store import resolve_and_save_dataset

    sessions = vero_home / "sessions"
    datasets = vero_home / "datasets"
    (sessions / SESSION_ID).mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)
    if not isinstance(dataset, str):  # a DatasetDict -> save_to_disk first
        path = tmp / "ds"
        dataset.save_to_disk(str(path))
        dataset = str(path)
    return resolve_and_save_dataset(dataset, sessions, datasets, SESSION_ID)


def _resolve_task_source_names(task_source: str) -> set[str] | None:
    """Enumerate the registry task_source's canonical task names, or None if
    that is not possible right now (harbor not importable, offline, ...)."""
    try:
        import asyncio

        from harbor.models.job.config import DatasetConfig

        name = task_source.split("@")[0]
        coro = DatasetConfig(name=name).get_task_configs()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            cfgs = asyncio.run(coro)  # no loop running: the normal sync path
        else:
            # A loop is already running (async caller, pytest-asyncio, notebook):
            # asyncio.run would raise and the bare except below would silently
            # skip validation. Run the enumeration on its own loop in a worker.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                cfgs = pool.submit(asyncio.run, coro).result()
        return {c.name for c in cfgs}
    except Exception as e:
        logger.debug(f"task_source enumeration failed: {type(e).__name__}: {e}")
        return None


def _validate_partition_names(
    partition: dict[str, list[str]], task_source: str
) -> None:
    """Fail the build when partition task names do not exist in the task_source.

    Harbor records trial results under canonical '<org>/<name>' task names; a
    bare or misspelled name in the partition compiles fine but surfaces only at
    eval time as an all-zero experiment (every nested trial unmatched). Catch it
    here, where it costs nothing. Best-effort: when the source cannot be
    enumerated (offline compile), warn and continue.
    """
    import os

    if os.environ.get("VERO_SKIP_TASK_NAME_CHECK"):
        return
    names = _resolve_task_source_names(task_source)
    if not names:
        # None (enumeration failed: offline, harbor missing) or an empty source;
        # either way there is nothing meaningful to check against, and raising
        # "unknown names, e.g. []" would only confuse.
        logger.warning(
            f"Could not enumerate task names for task_source '{task_source}' "
            f"(offline, harbor not importable, or empty source); skipping the "
            f"check."
        )
        return
    unknown = sorted({t for tasks in partition.values() for t in tasks} - names)
    if unknown:
        sample = sorted(names)[:3]
        raise ValueError(
            f"partition contains task name(s) not found in task_source "
            f"'{task_source}': {unknown[:5]}. Task names must use harbor's "
            f"canonical '<org>/<name>' form, e.g. {sample}. "
            f"(Set VERO_SKIP_TASK_NAME_CHECK=1 to bypass.)"
        )


def _serve_config(config: BuildConfig, dataset_id: str | None, base_commit: str) -> dict:
    harbor = None
    if config.harbor is not None:
        # Local inner task -> baked sidecar-only path; registry ref -> pass through.
        harbor = {**config.harbor}
        if config.inner_task:
            harbor["task_source"] = INNER_TASK
    targets = [
        {
            "task": config.task,
            "dataset_id": dataset_id,
            "split": t.split,
            "reward_key": t.reward_key,
            "sample_ids": t.sample_ids,
        }
        for t in config.targets
    ]
    return {
        "repo_path": AGENT_BASELINE,
        "agent_repo_path": WORK_AGENT,
        "session_id": SESSION_ID,
        "dataset_id": dataset_id,
        "split_accesses": [s.model_dump() for s in config.splits],
        "budgets": [
            {"split": b.split, "dataset_id": dataset_id, **b.model_dump(exclude={"split"}, exclude_none=True)}
            for b in config.budgets
        ],
        "task": config.task,
        "task_project": config.task_project,
        "task_module": config.task_module,
        "harbor": harbor,
        "reward_mode": config.reward_mode,
        "selection_split": config.selection_split,
        "targets": targets,
        "base_commit": base_commit,
        "submit_enabled": config.submit_enabled,
        "score_baseline": config.score_baseline,
        "feedback_transcripts": config.feedback_transcripts,
        "feedback_max_bytes": config.feedback_max_bytes,
        "instruct_multifidelity": config.instruct_multifidelity,
        "agent_volume": AGENT_VOLUME,
        "admin_volume": ADMIN_VOLUME,
        "admin_token_path": TOKEN_PATH,
        "timeout": config.timeout,
        "sample_timeout": config.sample_timeout,
        "max_concurrency": config.max_concurrency,
        "host": "0.0.0.0",
        "port": 8000,
    }


def compile_task(
    config: BuildConfig, out_dir: Path | str, *, vero_root: Path | None = None
) -> Path:
    """Compile ``config`` into a Harbor task directory at ``out_dir``."""
    import json

    from vero.core.constants import PACKAGE_DIR

    vero_root = vero_root or PACKAGE_DIR
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    env_dir = out / "environment"
    env_dir.mkdir(parents=True)

    agent_repo = Path(config.agent_repo).resolve()

    # 1. vero source (both images install from here)
    _copy_vero_source(vero_root, env_dir / "vero")

    # 2. baseline repo -> sidecar engine workspace + main seed (shared sha)
    base_commit = _prepare_baseline_repo(agent_repo, env_dir / "agent-baseline")
    shutil.copytree(env_dir / "agent-baseline", env_dir / "agent-seed")

    # 3. dataset -> baked VERO_HOME.  Mode A: input+label rows.  Mode B: the
    #    {split: [task_names]} partition + the inner Harbor task baked sidecar-only.
    import tempfile

    vh = env_dir / "sidecar" / "vero_home"
    # Stage the dataset in a scratch dir that is always cleaned up (datasets can be
    # gigabytes; a leaked mkdtemp would accumulate across builds). _register copies
    # the dataset into vh before the dir is torn down, so cleanup is safe.
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        if config.mode == "A":
            if not config.dataset:
                raise ValueError("Mode A requires a dataset.")
            dataset_id = _register(config.dataset, vh, tmp)
        else:
            if not (config.partition and config.harbor):
                raise ValueError("Mode B requires partition + harbor.")
            if not (config.inner_task or config.harbor.get("task_source")):
                raise ValueError("Mode B requires inner_task (local) or harbor.task_source (registry).")
            from vero.harbor.dataset import build_harbor_dataset

            if config.harbor.get("task_source") and not config.inner_task:
                _validate_partition_names(
                    config.partition, config.harbor["task_source"]
                )
            dataset_id = _register(build_harbor_dataset(config.partition), vh, tmp)
            if config.inner_task:  # local benchmark -> bake sidecar-only
                shutil.copytree(Path(config.inner_task).resolve(), env_dir / "sidecar" / "inner-task")

    # 4. ServeConfig (compiler <-> serve contract)
    (env_dir / "sidecar" / "serve.json").write_text(
        json.dumps(_serve_config(config, dataset_id, base_commit), indent=2)
    )

    # 4b. Fail early if a declared secret is missing from the host env, so the
    #     operator finds out at build time rather than via a credential-less
    #     sidecar. The compose ${VAR:?} guard is the run-time backstop.
    import os

    if not os.environ.get("VERO_SKIP_SECRET_CHECK"):
        missing = [s for s in config.secrets if not os.environ.get(s)]
        if missing:
            raise ValueError(
                "Declared secrets missing from the host environment: "
                f"{', '.join(missing)}. Set them, or set VERO_SKIP_SECRET_CHECK=1 "
                "to defer to the run-time compose check."
            )

    # 5. render templates
    jenv = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    ctx = dict(
        name=config.name,
        description=config.description,
        mode=config.mode,
        timeout=config.timeout,
        secrets=config.secrets,
        read_only_paths=config.read_only_paths,
        base_image_main=config.base_image_main,
        base_image_sidecar=config.base_image_sidecar,
        dataset_id=dataset_id,
        selection_split=config.selection_split,
        submit_enabled=config.submit_enabled,
        eval_num_samples=None,
        bake_inner_task=bool(config.inner_task),
        # The free-baseline bullet may only render when the sidecar shipping in
        # this same tree actually grants the free eval; the feature lives on a
        # different PR chain than the compiler, and an instruction that promises
        # it without it would send the agent to burn a metered eval on a commit
        # auto_best cannot select. Introspecting the protocol keeps the
        # instruction truthful under any merge order.
        free_baseline="free_baseline_available"
        in {f.name for f in dataclasses.fields(StatusSummary)},
        # Same merge-order-truthfulness introspection for the multi-fidelity
        # section: it may only render when the sidecar shipping in this tree
        # actually accepts subset evals (sample_ids / num_samples on the eval
        # request), or the instruction would teach a knob that 400s.
        multifidelity=config.instruct_multifidelity
        and {"sample_ids", "num_samples"}
        <= {f.name for f in dataclasses.fields(EvalRequest)},
    )
    _render(jenv, "task.toml.j2", out / "task.toml", **ctx)
    _render(jenv, "instruction.md.j2", out / "instruction.md", **ctx)
    _render(jenv, "docker-compose.yaml.j2", env_dir / "docker-compose.yaml", **ctx)
    _render(jenv, "Dockerfile.main.j2", env_dir / "Dockerfile", **ctx)
    _render(jenv, "Dockerfile.sidecar.j2", env_dir / "sidecar" / "Dockerfile", **ctx)
    _render(jenv, "seed.sh.j2", env_dir / "main" / "seed.sh", **ctx)
    _render(jenv, "test.sh.j2", out / "tests" / "test.sh", **ctx)
    _render(jenv, "solve.sh.j2", out / "solution" / "solve.sh", **ctx)

    for script in [out / "tests" / "test.sh", out / "solution" / "solve.sh",
                   env_dir / "main" / "seed.sh"]:
        script.chmod(0o755)

    logger.info("Compiled Harbor task -> %s (baseline %s)", out, base_commit[:12])
    return out
