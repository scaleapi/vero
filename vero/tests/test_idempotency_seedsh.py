"""Boot-twice behaviour of the compiled seed script.

The seed script is the main service's compose `command`, so Docker runs it again
every time it restarts a crashed main container, against the same `agent_repo`
volume. That makes it a first-boot script that in practice boots many times, and
the two things it did unconditionally both compounded across a long run: the
`.git/info/exclude` appends duplicated themselves on every boot, and the baked
workspace overlay was copied back over work the optimizer had already done.

These tests therefore execute the rendered script rather than grep its text: the
only way to tell "applied once" from "applied again" is to run it twice and look
at the workspace afterwards. The container paths are rewritten onto a sandbox
under tmp_path, and `chown` / `git` are stubbed on PATH because the real ones
need root and a system gitconfig that a test must not touch.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from vero.harbor import (
    AgentAccessSpec,
    HarborBuildConfig,
    VerificationTargetSpec,
    WorkspaceOverlaySpec,
    compile_harbor_task,
)
from vero.layout import LAYOUT

_OVERLAY_SKILL = "# baked overlay skill\n"


def _git(path: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments], cwd=path, check=True, text=True, capture_output=True
    )


def _config(tmp_path: Path) -> HarborBuildConfig:
    """The smallest build that still bakes an overlay into the workspace."""
    target = tmp_path / "target"
    target.mkdir(parents=True)
    _git(target, "init", "-q")
    _git(target, "config", "user.name", "VeRO Test")
    _git(target, "config", "user.email", "vero@example.test")
    (target / "README.md").write_text("# Target\n", encoding="utf-8")
    _git(target, "add", ".")
    _git(target, "commit", "-q", "-m", "target baseline")

    task_source = tmp_path / "tasks"
    task_source.mkdir()
    for name in ("task-a", "task-b", "task-c", "task-d", "task-e", "task-hidden"):
        task = task_source / name
        task.mkdir()
        (task / "task.toml").write_text(
            f'[task]\nname="org/{name}"\n', encoding="utf-8"
        )

    bundle = tmp_path / "bundle" / "skills" / "insights"
    bundle.mkdir(parents=True)
    (bundle / "SKILL.md").write_text(_OVERLAY_SKILL, encoding="utf-8")

    return HarborBuildConfig(
        name="org/optimize-program",
        description="Improve the program",
        agent_repo=str(target),
        task_source=str(task_source),
        agent_import_path="target.agent:Agent",
        harbor_requirement="harbor==0.1.17",
        partitions={
            "validation": ["task-a", "task-b", "task-c", "task-d", "task-e"],
            "test": ["task-hidden"],
        },
        agent_access=[
            AgentAccessSpec(
                partition="validation",
                expose_case_resources=True,
                total_runs=5,
                total_cases=25,
            )
        ],
        selection_partition="validation",
        targets=[VerificationTargetSpec(partition="test")],
        workspace_overlays=[
            WorkspaceOverlaySpec(
                source=str(tmp_path / "bundle" / "skills"), dest="skills"
            )
        ],
    )


def _sandbox(tmp_path: Path, seed_script: str) -> tuple[Path, Callable[[], None]]:
    """Stage a runnable copy of the rendered seed script and return (work, boot).

    The script's paths are absolute container paths, so the rewrite below is what
    makes it executable on the host at all. `exec sleep infinity` is dropped for
    the obvious reason: the real script never returns.
    """
    root = tmp_path / "sandbox"
    seed_repo = root / "seed"
    overlay = root / "overlay"
    work = root / "work"
    stubs = root / "bin"
    for directory in (overlay, work, stubs):
        directory.mkdir(parents=True)

    # The seed repo carries the .git the first-boot guard keys on, including the
    # info/exclude that git itself creates, so the appends land in a real file.
    (seed_repo / ".git" / "info").mkdir(parents=True)
    (seed_repo / ".git" / "info" / "exclude").write_text(
        "# git ls-files --others --exclude-from=.git/info/exclude\n", encoding="utf-8"
    )
    (seed_repo / "README.md").write_text("# Target\n", encoding="utf-8")
    (overlay / "skills" / "insights").mkdir(parents=True)
    (overlay / "skills" / "insights" / "SKILL.md").write_text(
        _OVERLAY_SKILL, encoding="utf-8"
    )
    for stub in ("chown", "git"):
        path = stubs / stub
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    body = (
        seed_script.replace(LAYOUT.overlay, str(overlay))
        .replace(LAYOUT.seed_repo, str(seed_repo))
        .replace(LAYOUT.target_repo, str(work))
        .replace("exec sleep infinity\n", "")
    )
    assert "/work/agent" not in body and "/opt/" not in body
    script = root / "seed.sh"
    script.write_text(body, encoding="utf-8")

    def boot() -> None:
        subprocess.run(
            ["/bin/sh", str(script)],
            check=True,
            text=True,
            capture_output=True,
            env={"PATH": f"{stubs}{os.pathsep}{os.defpath}", "HOME": str(root)},
        )

    return work, boot


def test_seed_script_appends_git_excludes_once_across_boots(tmp_path):
    """A restart must not append the same ignore lines a second time.

    Before this was guarded, a run that bounced the main container a dozen times
    left a dozen copies of /.evals/ and /skills/ in .git/info/exclude, and the
    file grew for as long as the run lasted.
    """
    output = compile_harbor_task(_config(tmp_path), tmp_path / "compiled")
    seed_script = (output / "environment/main/seed.sh").read_text(encoding="utf-8")
    work, boot = _sandbox(tmp_path, seed_script)

    boot()
    exclude = work / ".git" / "info" / "exclude"
    first = exclude.read_text(encoding="utf-8").splitlines()
    assert first.count("/.evals/") == 1
    assert first.count("/skills/") == 1

    boot()
    boot()
    after = exclude.read_text(encoding="utf-8").splitlines()
    assert after.count("/.evals/") == 1
    assert after.count("/skills/") == 1
    # Nothing else in the file moved either, so the guard is not rewriting it.
    assert after == first


def test_seed_script_does_not_reapply_the_overlay_after_a_restart(tmp_path):
    """A restart must leave the optimizer's edits under overlay paths alone.

    The overlay copy used to sit outside the first-boot guard, so a container
    that came back after a crash re-applied the baked skills over whatever the
    optimizer had written there, reverting its work without a word.
    """
    output = compile_harbor_task(_config(tmp_path), tmp_path / "compiled")
    seed_script = (output / "environment/main/seed.sh").read_text(encoding="utf-8")
    work, boot = _sandbox(tmp_path, seed_script)

    boot()
    skill = work / "skills" / "insights" / "SKILL.md"
    assert skill.read_text(encoding="utf-8") == _OVERLAY_SKILL

    # Stand in for the optimizer editing an overlay-provided file, which is
    # exactly what a second copy of the baked overlay would overwrite.
    skill.write_text("# edited by the optimizer\n", encoding="utf-8")
    boot()

    assert skill.read_text(encoding="utf-8") == "# edited by the optimizer\n"
