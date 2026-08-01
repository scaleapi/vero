"""A second launch of the same run has to be able to pick up the first one's work.

PR #78 hardened a resume path, and every one of those guards sits inside a
session directory that a relaunch never reaches: `vero harbor run` compiles into
a throwaway directory, the session lives on the `admin_state` compose volume
inside a per-trial Modal sandbox, and the sandbox goes away with the trial. The
one copy that outlives it is the archive on the launching host, so the tests
here are about that archive: refusing a bad one before an image is built,
carrying a good one into the next stack's sidecar without widening who can read
it, restoring it before anything opens the session directory, and not restoring
it over a session that is currently being written.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vero.evaluation import (
    DisclosureLevel,
    EvaluationAccessPolicy,
    EvaluationBudget,
    EvaluationCost,
    EvaluationSet,
    MetricSelector,
    ObjectiveSpec,
)
from vero.harbor import (
    AgentAccessSpec,
    HarborBackendConfig,
    HarborBuildConfig,
    VerificationTargetSpec,
    build_harbor_components,
    compile_harbor_task,
)
from vero.layout import LAYOUT
from vero.sidecar import SidecarEvaluationPolicy, VerificationTarget
from vero.sidecar.session import (
    HarborSessionManifest,
    create_harbor_session_archive,
    read_harbor_session_archive_manifest,
    restore_harbor_session_archive,
)
from vero.sidecar.verifier import VerificationSelection

_VERO_ROOT = Path(__file__).parents[1]


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _repo(path: Path) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "VeRO Test")
    _git(path, "config", "user.email", "vero@example.test")
    (path / "program.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "pyproject.toml").write_text(
        '[project]\nname="target"\nversion="0.1.0"\n', encoding="utf-8"
    )
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "baseline")
    return _git(path, "rev-parse", "HEAD")


def _session_archive(root: Path, *, payload: str = "restored") -> Path:
    """The smallest thing that satisfies the archive contract, plus a marker.

    A real archive is whatever a run left behind; what every consumer here keys
    off is the manifest at the root, so this stands in for one wherever the test
    is about transport rather than about the state itself.
    """

    session = root / "session"
    session.mkdir(parents=True)
    manifest = HarborSessionManifest(
        id="trial",
        task_name="org/task",
        created_at=datetime.now(UTC),
        backends={},
        selection=VerificationSelection(mode="submit"),
        targets=[],
    )
    (session / "harbor-session.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )
    (session / "marker.txt").write_text(payload, encoding="utf-8")
    return create_harbor_session_archive(session, root / "session.tar.gz")


def _build_config(root: Path) -> HarborBuildConfig:
    target = root / "target"
    _repo(target)
    tasks = root / "tasks"
    for name in ("task-a", "task-hidden"):
        (tasks / name).mkdir(parents=True)
        (tasks / name / "task.toml").write_text(
            f'[task]\nname="org/{name}"\n', encoding="utf-8"
        )
    return HarborBuildConfig(
        name="org/optimize-program",
        description="Improve the program",
        agent_repo=str(target),
        task_source=str(tasks),
        agent_import_path="target.agent:Agent",
        harbor_requirement="harbor==0.1.17",
        partitions={"validation": ["task-a"], "test": ["task-hidden"]},
        agent_access=[AgentAccessSpec(partition="validation", total_runs=5)],
        selection_partition="validation",
        targets=[VerificationTargetSpec(partition="test")],
    )


def test_a_compile_carries_the_session_archive_only_when_a_launch_asked_for_one(
    tmp_path,
):
    """The archive has to travel in the sidecar image, and only on request.

    Baking it in is the only transport there is: the session directory is a
    compose volume inside the sandbox, so a relaunch has nothing to mount. That
    also means the default must stay clean, because every launch that is not a
    resume would otherwise pay the image size and, worse, adopt a previous run's
    scores without anyone having asked for it.
    """

    config = _build_config(tmp_path / "build")
    archive = _session_archive(tmp_path / "previous")

    plain = compile_harbor_task(
        config, tmp_path / "plain", vero_root=_VERO_ROOT
    )
    sidecar = plain / "environment/sidecar"
    assert not (sidecar / "session-seed.tar.gz").exists()
    assert json.loads((sidecar / "serve.json").read_text())[
        "session_seed_archive"
    ] is None
    assert "session-seed.tar.gz" not in (sidecar / "Dockerfile").read_text()

    resumed = compile_harbor_task(
        config,
        tmp_path / "resumed",
        vero_root=_VERO_ROOT,
        session_seed_archive=archive,
    )
    sidecar = resumed / "environment/sidecar"
    assert (sidecar / "session-seed.tar.gz").read_bytes() == archive.read_bytes()
    assert (
        json.loads((sidecar / "serve.json").read_text())["session_seed_archive"]
        == LAYOUT.session_seed_archive
    )
    dockerfile = (sidecar / "Dockerfile").read_text()
    copy = f"COPY sidecar/session-seed.tar.gz {LAYOUT.session_seed_archive}"
    assert copy in dockerfile
    # The archive carries database.json, so it discloses held-out membership and
    # per-case scores exactly as the case lists do. It gets the same lock, or the
    # unprivileged harness that runs candidate code can read the answers.
    assert f"chmod 600 {LAYOUT.session_seed_archive}" in dockerfile


def test_a_launch_refuses_an_archive_it_could_not_have_restored(tmp_path):
    """Reject on the host, before an image build, not inside the sandbox.

    The failure this prevents is silent and expensive: a wrong path compiles
    fine, builds three images, brings the stack up, and only then does the
    sidecar refuse, tens of minutes into a run whose logs nobody is tailing.
    """

    config = _build_config(tmp_path / "build")
    plain = tmp_path / "not-an-archive.tar.gz"
    with tarfile.open(plain, "w:gz") as archive:
        source = tmp_path / "loose.txt"
        source.write_text("no manifest here\n", encoding="utf-8")
        archive.add(source, arcname="session/loose.txt")

    with pytest.raises(ValueError, match="no session manifest"):
        compile_harbor_task(
            config,
            tmp_path / "out",
            vero_root=_VERO_ROOT,
            session_seed_archive=plain,
        )
    assert not (tmp_path / "out").exists()

    escaping = tmp_path / "escaping.tar.gz"
    with tarfile.open(escaping, "w:gz") as archive:
        archive.add(tmp_path / "loose.txt", arcname="../escape.txt")
    with pytest.raises(ValueError, match="unsafe Harbor session archive member"):
        read_harbor_session_archive_manifest(escaping)


def test_a_restore_seeds_an_empty_session_and_spares_a_live_one(tmp_path):
    """The two boots that reach this code are indistinguishable from inside it.

    A relaunch boots against a fresh volume and must be seeded. A sidecar restart
    inside a live run boots against the volume it has been writing all along, and
    seeding that would roll the run back to an older attempt's state. The
    manifest is what tells them apart.
    """

    archive = _session_archive(tmp_path / "previous")

    fresh = tmp_path / "state/session"
    assert restore_harbor_session_archive(archive, fresh) is True
    assert (fresh / "marker.txt").read_text() == "restored"
    assert json.loads((fresh / "harbor-session.json").read_text())["id"] == "trial"
    # Staged into a sibling and moved in, so nothing is left behind to be picked
    # up as session state on a later boot.
    assert sorted(path.name for path in (tmp_path / "state").iterdir()) == ["session"]

    live = tmp_path / "live/session"
    live.mkdir(parents=True)
    (live / "harbor-session.json").write_text(
        (fresh / "harbor-session.json").read_text(), encoding="utf-8"
    )
    (live / "marker.txt").write_text("work since the restart", encoding="utf-8")
    assert restore_harbor_session_archive(archive, live) is False
    assert (live / "marker.txt").read_text() == "work since the restart"


@pytest.mark.asyncio
async def test_a_relaunch_resumes_the_previous_run_s_ledger_and_candidates(tmp_path):
    """The end of the line: a fresh volume that behaves like the old one.

    Everything the sidecar builds below the session directory reopens what is on
    disk, so restoring the directory before any of them is constructed is the
    whole of the resume. The budget ledger is the proof that carries the least
    ceremony and the most meaning: a relaunch that did not resume hands the
    optimizer its agent budget back, and a run that has already spent hours can
    then spend it a second time.
    """

    trusted = tmp_path / "trusted"
    agent = tmp_path / "agent"
    _repo(trusted)
    _repo(agent)
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps({"id": "task", "task_name": "org/task"}) + "\n", encoding="utf-8"
    )
    evaluation_set = EvaluationSet(name="benchmark", partition="validation")
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"), direction="maximize"
    )
    backend = HarborBackendConfig(
        task_source="org/benchmark@1.0",
        agent_import_path="program:Agent",
        cases_path=str(cases),
        harbor_requirement="harbor==0.1.17",
        evaluation_set_name="benchmark",
        partition="validation",
        uv_executable=sys.executable,
    )

    def _config(session_dir: Path, seed: Path | None = None) -> dict:
        return {
            "repo_path": str(trusted),
            "agent_repo_path": str(agent),
            "session_dir": str(session_dir),
            "session_id": "trial",
            **({"session_seed_archive": str(seed)} if seed is not None else {}),
            "backends": {"validation": backend.model_dump(mode="json")},
            "access_policies": [
                SidecarEvaluationPolicy(
                    backend_id="validation",
                    evaluation_set_name="benchmark",
                    partition="validation",
                    objective=objective,
                    access=EvaluationAccessPolicy(
                        disclosure=DisclosureLevel.AGGREGATE
                    ),
                ).model_dump(mode="json")
            ],
            "budgets": [
                EvaluationBudget(
                    backend_id="validation",
                    evaluation_set_key=evaluation_set.budget_key("validation"),
                    total_runs=4,
                ).model_dump(mode="json")
            ],
            "selection": {
                "mode": "auto_best",
                "backend_id": "validation",
                "evaluation_set": evaluation_set.model_dump(mode="json"),
                "objective": objective.model_dump(mode="json"),
                "baseline_version": "HEAD",
            },
            "targets": [
                VerificationTarget(
                    reward_key="reward",
                    backend_id="validation",
                    evaluation_set=evaluation_set,
                    objective=objective,
                ).model_dump(mode="json")
            ],
            "agent_volume": str(session_dir.parent / "agent"),
            "admin_volume": str(session_dir.parent),
        }

    first = tmp_path / "attempt-one/session"
    components = await build_harbor_components(_config(first))
    ledger = components.sidecar.engine.budget_ledger
    assert ledger is not None
    await ledger.reserve("validation", evaluation_set, EvaluationCost(runs=3))
    archive = create_harbor_session_archive(first, tmp_path / "rescue.tar.gz")

    # The relaunch: a session directory that has never existed, exactly what a
    # new sandbox presents. Without the archive it is a brand new run.
    clean = await build_harbor_components(_config(tmp_path / "attempt-two/session"))
    assert clean.sidecar.status().evaluation_access[0].budget.remaining_runs == 4

    resumed = await build_harbor_components(
        _config(tmp_path / "attempt-three/session", seed=archive)
    )
    assert resumed.sidecar.status().evaluation_access[0].budget.remaining_runs == 1
    third = tmp_path / "attempt-three/session"
    assert (third / "candidates/repository.git").is_dir()
    assert (
        json.loads((third / "harbor-session.json").read_text())["created_at"]
        == json.loads((first / "harbor-session.json").read_text())["created_at"]
    )
    # Restored before the lockdown, not after: the trusted state is still closed
    # to the unprivileged harness on a resumed boot.
    assert (third.stat().st_mode & 0o777) == 0o700

    # A resume into a build that is no longer the one that produced the archive
    # has to fail, and loudly. The scores in a restored database were measured
    # against a particular backend and objective, and silently continuing under
    # a different one would mix two incomparable runs into a single ranking.
    # `initialize_harbor_session_manifest` is what catches it, on the restored
    # manifest, which only exists here because the restore ran.
    drifted = _config(tmp_path / "attempt-four/session", seed=archive)
    drifted["selection"]["objective"] = ObjectiveSpec(
        selector=MetricSelector(metric="score"), direction="minimize"
    ).model_dump(mode="json")
    with pytest.raises(ValueError, match="incompatible with deployment"):
        await build_harbor_components(drifted)


def test_the_config_starter_documents_the_local_resume_opt_in(tmp_path):
    """`vero init` never mentioned the one knob that makes a rerun resume.

    `[session] id` has always worked, and no template, doc string or build config
    emitted it, so every config-driven run took the `uuid4` fallback in
    `_session_identity` and reached none of the resume machinery. Emitting it
    commented out keeps the default untouched and makes the opt-in findable.
    """

    from click.testing import CliRunner

    from vero.cli import _CONFIG_TEMPLATE, main
    from vero.config import _session_identity, load_config

    result = CliRunner().invoke(main, ["init", str(tmp_path / "starter")])
    assert result.exit_code == 0, result.output
    emitted = (tmp_path / "starter/vero.toml").read_text()
    assert "# [session]" in emitted
    assert '# id = "my-run-2026-08-01"' in emitted
    assert emitted == _CONFIG_TEMPLATE

    # As emitted, two launches are two runs: the fallback mints a fresh uuid4.
    (tmp_path / "starter/target").mkdir()
    (tmp_path / "starter/harness").mkdir()
    config_path = tmp_path / "starter/vero.toml"
    first = _session_identity(load_config(config_path))[1]
    assert _session_identity(load_config(config_path))[1] != first

    # Uncommented, they are one run over one directory, which is what every
    # guard in the resume path needs in order to ever run.
    config_path.write_text(
        emitted.replace("# [session]", "[session]").replace(
            '# id = "my-run-2026-08-01"', 'id = "my-run-2026-08-01"'
        ),
        encoding="utf-8",
    )
    identity, directory = _session_identity(load_config(config_path))
    assert identity == "my-run-2026-08-01"
    assert directory == _session_identity(load_config(config_path))[1]
    assert directory.name == "my-run-2026-08-01"


def test_harbor_run_hands_the_resume_archive_to_the_compile(tmp_path, monkeypatch):
    """The flag is only worth anything if it reaches the compiler.

    Asserted rather than assumed because the wiring is invisible at run time: the
    launch prints the same command line either way, and a dropped argument shows
    up as a silently fresh run hours later.
    """

    from vero.harbor import build as harbor_build
    from vero.harbor import cli as harbor_cli

    config_path = tmp_path / "build.yaml"
    config_path.write_text("name: org/task\n", encoding="utf-8")
    archive = _session_archive(tmp_path / "previous")

    class _Config:
        harbor_requirement = "harbor[modal]==0.20.0"
        agent_env: dict[str, str] = {}
        optimizer_harbor_args: list[str] = []
        extra_harbor_args: list[str] = []
        name = "vero/stub-benchmark"

    seen: dict[str, object] = {}

    def _compile(config, output, **keywords):
        seen.update(keywords)
        output.mkdir(parents=True)
        return output

    monkeypatch.setattr(
        harbor_build, "load_harbor_build_config", lambda *a, **k: _Config()
    )
    monkeypatch.setattr(harbor_build, "compile_harbor_task", _compile)
    monkeypatch.setattr(harbor_cli.shutil, "which", lambda name: "/usr/bin/uvx")
    monkeypatch.setattr(
        harbor_cli, "_compiled_run_environment", lambda task, overrides: {}
    )
    monkeypatch.setattr(
        harbor_cli.subprocess,
        "run",
        lambda command, env=None: subprocess.CompletedProcess(command, 0),
    )

    from click.testing import CliRunner

    from vero.cli import main

    result = CliRunner().invoke(
        main,
        [
            "harbor",
            "run",
            "--config",
            str(config_path),
            "--agent",
            "codex",
            "--resume-from",
            str(archive),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["session_seed_archive"] == archive
    assert str(archive) in result.output

    seen.clear()
    result = CliRunner().invoke(
        main,
        ["harbor", "run", "--config", str(config_path), "--agent", "codex"],
    )
    assert result.exit_code == 0, result.output
    assert seen["session_seed_archive"] is None
