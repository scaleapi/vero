"""Container integration test for harness-isolation filesystem invariants.

Runs inside a throwaway Linux container (the CI/dev host may be macOS, where the
uid-drop is unavailable) and asserts, against the *real* privilege drop
(``vero.sandbox.LocalSandbox.run(run_as=...)``) and the *real* provisioning
commands (``vero.sidecar.isolation``), that:

* a candidate checkout under a ``mktemp -d`` (0700 root) parent is UNreachable to
  the dropped user when only the leaf is chowned (the regression sentinel — this
  is the bug that shipped),
* it becomes reachable after ``harness_grant_commands`` (and the reachability
  probe agrees), and
* trusted root-only state stays unreadable to the dropped user.

The container loads ``sandbox.py``/``isolation.py`` standalone (they are
stdlib-only) so no vero install or heavy dependency is needed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
_VERO_SRC = Path(__file__).resolve().parents[1] / "src"


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_usable(), reason="requires a usable docker daemon"
)


# Executed as root inside the container.
_CONTAINER_SCRIPT = r'''
import asyncio, importlib.util, os, subprocess, sys


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass/typing introspection reads sys.modules
    spec.loader.exec_module(module)
    return module


# Real modules, loaded standalone (stdlib-only) — no vero install needed.
sbx = _load("_vero_sandbox", "/vero-src/vero/sandbox.py")
iso = _load("_vero_isolation", "/vero-src/vero/sidecar/isolation.py")

subprocess.run(["useradd", "-m", "-u", "1002", "harness"], check=True)
sandbox = sbx.LocalSandbox("/")  # run_as drops privileges from this root process
MARKER = os.path.join("src", "tinyagent", "__init__.py")


def make_checkout():
    # Mirror candidate_repository.git.checkout: <mktemp -d>/repository. mktemp -d
    # leaves the parent 0700 root — the traversal the harness needs.
    parent = subprocess.run(
        ["mktemp", "-d", "/tmp/vero-candidate-checkoutXXXXXX"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    repo = os.path.join(parent, "repository")
    os.makedirs(os.path.join(repo, "src", "tinyagent"))
    open(os.path.join(repo, MARKER), "w").close()
    return parent, repo


async def as_harness(command):
    return await sandbox.run(command, run_as="harness")


async def main():
    failures = []

    # Trusted seal: a root-only 0700 dir + 0600 file (stands in for the session
    # dir / serve.json / admin token).
    os.makedirs("/trusted", exist_ok=True)
    with open("/trusted/secret", "w") as handle:
        handle.write("held-out")
    os.chmod("/trusted/secret", 0o600)
    os.chmod("/trusted", 0o700)

    # 1) Regression sentinel: chown only the leaf (the shipped bug). The dropped
    #    user must NOT be able to reach a file under the still-0700 parent.
    _parent, repo = make_checkout()
    subprocess.run(["chown", "-R", "harness:harness", repo], check=True)
    if (await as_harness(["test", "-r", os.path.join(repo, MARKER)])).returncode == 0:
        failures.append(
            "regression sentinel: workspace reachable WITHOUT the traversal grant"
        )

    # 2) Positive: the real provisioning makes it reachable, and the probe agrees.
    _parent, repo = make_checkout()
    for command in iso.harness_grant_commands(
        "harness", chown_paths=[repo], checkout_root=repo
    ):
        subprocess.run(command, check=True)
    if (await as_harness(iso.harness_reachability_probe(repo))).returncode != 0:
        failures.append("positive: reachability probe failed after provisioning")
    if (await as_harness(["test", "-r", os.path.join(repo, MARKER)])).returncode != 0:
        failures.append("positive: provisioned workspace unreadable to the harness")

    # 3) Negative: trusted state stays sealed from the dropped user.
    if (await as_harness(["cat", "/trusted/secret"])).returncode == 0:
        failures.append("negative: harness read the trusted /trusted/secret")

    if failures:
        print("ISOLATION INVARIANTS VIOLATED:")
        for item in failures:
            print("  -", item)
        sys.exit(1)
    print("all isolation invariants hold")


asyncio.run(main())
'''


def test_harness_isolation_invariants_on_container(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(_CONTAINER_SCRIPT)
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{_VERO_SRC}:/vero-src:ro",
            "-v", f"{script}:/probe.py:ro",
            _IMAGE,
            "python3", "/probe.py",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"isolation invariants failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "all isolation invariants hold" in result.stdout
