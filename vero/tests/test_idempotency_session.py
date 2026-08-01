"""Regressions for the session-level idempotency of a rerun.

Every case here stands for a way a relaunch after a crash used to lose work: a
first durable write that erased the previous death's diagnosis, and two live
processes silently sharing one session directory.

The deterministic default session identity that belongs beside these was backed
out of this branch: making it stable is correct, but it turns every rerun into a
resume, and on a session whose only stored baseline record is unusable the
resumed run adopts that record and measures nothing. That needs the baseline
reuse guards, which change what gets measured, so both land together or not at
all.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from tests.test_v05_runtime_session import StubOptimizer
from vero.runtime import OptimizationSession, SessionStatus


@pytest.mark.asyncio
async def test_running_manifest_keeps_the_previous_failure(tmp_path: Path):
    """A rerun must not erase how the previous attempt died.

    The RUNNING write is a rerun's first durable act, so clearing `failure`
    there destroyed the only recorded explanation of the death before anyone
    read it. Only a run that actually completed may drop it.
    """

    session_dir = tmp_path / "sessions" / "kept-failure"
    failing = OptimizationSession(
        id="kept-failure",
        session_dir=session_dir,
        optimizer=StubOptimizer(session_dir, failure=RuntimeError("producer exploded")),
    )
    with pytest.raises(RuntimeError, match="producer exploded"):
        await failing.run()
    assert failing.load_manifest().failure.message == "producer exploded"

    observed: list[dict] = []

    class _ObservingOptimizer(StubOptimizer):
        """Read the manifest exactly as the resumed run starts working."""

        async def run(self, **kwargs):
            observed.append(
                json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
            )
            return await super().run(**kwargs)

    resumed = OptimizationSession(
        id="kept-failure",
        session_dir=session_dir,
        optimizer=_ObservingOptimizer(session_dir),
    )
    result = await resumed.run(skip_baseline_evaluation=True)

    assert observed[0]["status"] == SessionStatus.RUNNING.value
    assert observed[0]["failure"]["message"] == "producer exploded"
    # The completed write is where the obsolete explanation goes away, because
    # only by then is there a result that supersedes it.
    assert resumed.load_manifest().failure is None
    assert result.best is not None


@pytest.mark.asyncio
async def test_session_refuses_a_second_process_on_one_directory(tmp_path: Path):
    """A relaunch while the first run is still alive must be refused.

    Two processes over one session directory evaluate the same pending set
    against separately loaded budget ledgers, so the budget is spent twice and
    the manifests overwrite each other.
    """

    session_dir = tmp_path / "sessions" / "contended"
    session = OptimizationSession(
        id="contended",
        session_dir=session_dir,
        optimizer=StubOptimizer(session_dir),
    )
    # Stand in for the live process: an flock held on another open file
    # description is exactly what a second `vero run` would find.
    holder = os.open(session_dir / "run.lock", os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(holder, b"4242\n")

    with pytest.raises(RuntimeError, match="already being run by another process"):
        await session.run()
    # Refused before any durable write, so the live run's state is untouched.
    assert not session.manifest_path.exists()

    # Closing is what a crash does too, so the relaunch this whole tranche
    # exists to support proceeds, over a lock file that is still lying there.
    os.close(holder)
    result = await session.run()

    assert result.best is not None
    assert session.load_manifest().status == SessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_session_run_lock_names_the_holder_and_is_released_at_exit(
    tmp_path: Path,
):
    """The refusal has to answer "is a run still alive?" without guesswork."""

    session_dir = tmp_path / "sessions" / "diagnosable"
    session = OptimizationSession(
        id="diagnosable",
        session_dir=session_dir,
        optimizer=StubOptimizer(session_dir),
    )

    await session.run()

    # The lock is released once run() returns, so a sequential rerun is free to
    # take it; the pid breadcrumb left behind belongs to this process.
    lock_path = session_dir / "run.lock"
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)
