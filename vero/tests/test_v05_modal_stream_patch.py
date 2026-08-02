"""The Modal stdio reconnect budget vero applies to its harbor subprocess.

Modal budgets stdio reconnects for the LIFE of a stream: the count is set once
above the read loop and a successful chunk resets only the backoff delay, so an
hours-long harbor phase spends the budget on blips unrelated to the drop that
eventually kills it. vero reaches into the subprocess with a `sitecustomize` on
PYTHONPATH and fixes two things there: the three numeric knobs (constructor
keywords that `TaskCommandRouterClient._connect` never forwards and
`modal/config.py` never exposes), and the missing per-chunk reset of the count,
which lives inside the method body and so needs modal's own source recompiled.

These tests pin both seams, because a silent no-op in either reads exactly like a
budget that survives the run.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import tempfile
import textwrap

from vero.harbor import cli as harbor_cli

PATCH_DIRECTORY = harbor_cli._STREAM_PATCH_DIRECTORY
KNOBS = (
    "stream_stdio_retry_delay_secs",
    "stream_stdio_retry_delay_factor",
    "stream_stdio_max_retries",
)


def _run_patch(environment: dict[str, str], modal_source: str) -> tuple[str, str]:
    """Execute the sitecustomize against a stand-in modal package.

    A stand-in rather than the real client: the point is the seam (keyword
    defaults on `__init__`), and pinning it against a fake keeps the test honest
    when modal is absent from the test environment. The end-to-end check against
    the installed modal lives in `test_patch_applies_to_the_real_modal_client`.
    """

    # Written to a file rather than passed with `-c`, because half the patch
    # recompiles modal's own source and `inspect.getsource` needs a real file to
    # read it from. A `-c` stand-in would decline the rewrite for a reason that
    # has nothing to do with the seam under test. (The same decline is the right
    # behaviour against a sourceless modal install, and is asserted separately.)
    directory = tempfile.mkdtemp(prefix="vero-stream-patch-")
    script = os.path.join(directory, "drive_fake_modal.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(modal_source))
    process = subprocess.run(
        [sys.executable, script],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(PATCH_DIRECTORY), os.environ.get("PYTHONPATH", "")]
            ).strip(os.pathsep),
            **environment,
        },
        capture_output=True,
        text=True,
    )
    return process.stdout, process.stderr


# A miniature of modal 1.5.3's `_stream_stdio_with_retries`, kept faithful in the
# ways the patch depends on: the delay assigned twice (once above `while True`,
# once per successful chunk) and the count assigned once, above the loop. Both
# seams are pinned against this rather than against the real client so the tests
# still run where modal is not installed; `test_patch_applies_to_the_real_modal_
# client` is what checks the shape has not drifted from the dependency.
FAKE_MODAL_MODULE = """
    import asyncio, sys, types
    package = types.ModuleType("modal")
    utils = types.ModuleType("modal._utils")
    module = types.ModuleType("modal._utils.task_command_router_client")

    class TaskCommandRouterClient:
        def __init__(
            self,
            server_client,
            *,
            stream_stdio_retry_delay_secs: float = 0.01,
            stream_stdio_retry_delay_factor: float = 2,
            stream_stdio_max_retries: int = 10,
        ) -> None:
            self.stream_stdio_retry_delay_secs = stream_stdio_retry_delay_secs
            self.stream_stdio_retry_delay_factor = stream_stdio_retry_delay_factor
            self.stream_stdio_max_retries = stream_stdio_max_retries

        def _get_metadata(self):
            return {}

        async def _stream_stdio_with_retries(
            self, *, stub_method, request_factory, deadline_label, deadline=None
        ):
            offset = 0
            delay_secs = self.stream_stdio_retry_delay_secs
            delay_factor = self.stream_stdio_retry_delay_factor
            num_retries_remaining = self.stream_stdio_max_retries

            async def sleep_and_update(e):
                nonlocal delay_secs, num_retries_remaining
                await asyncio.sleep(delay_secs)
                delay_secs *= delay_factor
                num_retries_remaining -= 1

            while True:
                try:
                    stream = stub_method.open(timeout=None, metadata=self._get_metadata())
                    async with stream as s:
                        await s.send_message(request_factory(offset), end=True)
                        async for item in s:
                            # Reset retry backoff after any successful chunk.
                            delay_secs = self.stream_stdio_retry_delay_secs
                            offset += len(item.data)
                            yield item
                    return
                except OSError as e:
                    if num_retries_remaining > 0:
                        await sleep_and_update(e)
                    else:
                        raise

    module.TaskCommandRouterClient = TaskCommandRouterClient
    sys.modules["modal"] = package
    sys.modules["modal._utils"] = utils
    sys.modules["modal._utils.task_command_router_client"] = module
"""

FAKE_MODAL = (
    FAKE_MODAL_MODULE
    + """
    import sitecustomize
    sitecustomize._apply()
    print({k: v for k, v in TaskCommandRouterClient.__init__.__kwdefaults__.items()})
"""
)

# Scripted outage sequence: three attempts that deliver one chunk and then drop,
# then one that delivers a chunk and ends. Every outage is separated by a
# success, so a lifetime budget of 2 is exhausted by the third drop while a
# per-outage budget of 2 is never below 2 when a drop arrives. The client is
# built with no keyword arguments on purpose: the budget it runs on has to come
# from the patched `__kwdefaults__`, the way `_connect` builds it.
FAKE_MODAL_DRIVEN = (
    FAKE_MODAL_MODULE
    + """
    import json, sitecustomize
    sitecustomize._apply()

    ATTEMPTS = 4

    class Chunk:
        def __init__(self, data):
            self.data = data

    class Attempt:
        def __init__(self, index, drops):
            self.index, self.drops = index, drops

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send_message(self, request, end=True):
            return None

        def __aiter__(self):
            async def chunks():
                yield Chunk(b"chunk\\n")
                if self.drops:
                    raise OSError("simulated drop %d" % self.index)
            return chunks()

    class Stub:
        def __init__(self):
            self.opened = 0

        def open(self, timeout=None, metadata=None):
            self.opened += 1
            return Attempt(self.opened, drops=self.opened < ATTEMPTS)

    async def drive():
        client = TaskCommandRouterClient(None)
        delivered, outcome = 0, "completed"
        stream = client._stream_stdio_with_retries(
            stub_method=Stub(), request_factory=lambda offset: None,
            deadline_label="test",
        )
        try:
            async for item in stream:
                delivered += 1
        except OSError as error:
            outcome = "died: %s" % error
        return {"delivered": delivered, "outcome": outcome, "attempts": ATTEMPTS}

    print(json.dumps(asyncio.run(drive())))
"""
)


def test_patch_widens_the_reconnect_budget_when_vero_asks() -> None:
    stdout, stderr = _run_patch(
        {
            "VERO_MODAL_STREAM_RETRY_DELAY_SECS": "2.0",
            "VERO_MODAL_STREAM_RETRY_FACTOR": "1.0",
            "VERO_MODAL_STREAM_MAX_RETRIES": "60",
        },
        FAKE_MODAL,
    )
    defaults = eval(stdout.strip().splitlines()[-1])
    assert defaults["stream_stdio_retry_delay_secs"] == 2.0
    assert defaults["stream_stdio_retry_delay_factor"] == 1.0
    assert defaults["stream_stdio_max_retries"] == 60
    # A flat factor is what makes the tolerated outage a simple product; a
    # doubling one would outgrow the run instead of pacing it. "per outage" is
    # load-bearing in this message: the same numbers spent per stream are the bug
    # this patch was rewritten to fix, so the job log has to name the scope.
    assert "120s tolerated per outage" in stderr


def _drive(environment: dict[str, str], source: str = FAKE_MODAL_DRIVEN) -> tuple:
    import json

    stdout, stderr = _run_patch(environment, source)
    return json.loads(stdout.strip().splitlines()[-1]), stderr


PER_OUTAGE_ENVIRONMENT = {
    "VERO_MODAL_STREAM_RETRY_DELAY_SECS": "0.0",
    "VERO_MODAL_STREAM_RETRY_FACTOR": "1.0",
    "VERO_MODAL_STREAM_MAX_RETRIES": "2",
}


def test_a_lifetime_budget_dies_on_outages_it_has_already_survived() -> None:
    """The defect, stated as a test: this is modal 1.5.3 without the rewrite.

    Three drops, each separated by a chunk that arrived fine, against a budget of
    two. The count is set once above the read loop and only the delay is reset
    per chunk, so the two are spent by the second drop and the third is fatal
    even though the stream has been proving itself healthy in between. Scaled up,
    that is a 3h19m optimizer phase dying to a blip because of blips hours
    earlier.
    """

    result, _ = _drive(PER_OUTAGE_ENVIRONMENT, _reshaped_loop())
    assert result["delivered"] == 3
    assert "died: simulated drop 3" in result["outcome"]


def test_patch_makes_the_retry_budget_per_outage() -> None:
    """The fix: a successful chunk restores the count, not just the delay.

    Same script, same budget of two, and now every drop is met with a full
    budget because the stream delivered a chunk since the last one. What the
    patch buys is not more retries, it is retries that belong to the outage in
    front of them.
    """

    result, stderr = _drive(PER_OUTAGE_ENVIRONMENT)
    assert result["delivered"] == result["attempts"] == 4
    assert result["outcome"] == "completed"
    assert "per outage" in stderr


def _reshaped_loop() -> str:
    """`FAKE_MODAL_DRIVEN` with the per-chunk delay reset written differently.

    Same behaviour, different AST, which is exactly the modal upgrade this patch
    has to survive: the rewrite must decline rather than guess where the count
    reset belongs. The last occurrence is the in-loop one.
    """

    head, tail = FAKE_MODAL_DRIVEN.rsplit(
        "delay_secs = self.stream_stdio_retry_delay_secs", 1
    )
    return head + "delay_secs = max(0.0, self.stream_stdio_retry_delay_secs)" + tail


def test_patch_declines_a_reshaped_retry_loop_and_says_so() -> None:
    """Recompiling a pinned dependency's private coroutine has to fail closed.

    A reshaped loop leaves modal's own method installed and untouched (the run
    dies at three chunks, exactly as unpatched), warns loudly, and still applies
    the numeric knobs, since a flat 2s budget is strictly wider than the 0.01s
    doubling modal ships and withholding it would punish the run for the
    rewrite's failure rather than protect it.
    """

    result, stderr = _drive(PER_OUTAGE_ENVIRONMENT, _reshaped_loop())
    assert result["delivered"] == 3
    assert "changed shape" in stderr
    assert "budget stays lifetime-scoped" in stderr
    # The knobs landed even though the rewrite did not, and the message says
    # which scope is actually in force rather than implying the good one.
    assert "per stream, lifetime" in stderr


def test_patch_is_inert_when_vero_does_not_ask() -> None:
    """Importable anywhere without changing behaviour, so it cannot surprise."""

    stdout, stderr = _run_patch({}, FAKE_MODAL)
    defaults = eval(stdout.strip().splitlines()[-1])
    assert defaults["stream_stdio_max_retries"] == 10
    assert stderr == ""


def test_patch_is_loud_when_modal_renames_the_knobs() -> None:
    """The failure that matters: a modal upgrade turning this into a no-op.

    Silence would read as "the budget is wide" while the run is back on ten
    seconds of tolerance, so an absent knob has to reach the job log.
    """

    renamed = FAKE_MODAL.replace("stream_stdio_max_retries", "stream_stdio_max_attempts")
    _, stderr = _run_patch({"VERO_MODAL_STREAM_MAX_RETRIES": "60"}, renamed)
    assert "modal API changed" in stderr
    assert "stream_stdio_max_retries" in stderr


def test_environment_puts_the_patch_on_the_subprocess_path() -> None:
    environment = harbor_cli._modal_stream_patch_environment({})
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(PATCH_DIRECTORY)
    assert (PATCH_DIRECTORY / "sitecustomize.py").is_file()

    retries = int(
        harbor_cli.MODAL_STREAM_RECONNECT_WINDOW_SECONDS
        / harbor_cli.MODAL_STREAM_RECONNECT_DELAY_SECONDS
    )
    assert environment["VERO_MODAL_STREAM_MAX_RETRIES"] == str(retries)


def test_environment_preserves_an_inherited_pythonpath() -> None:
    """Prepend, never replace: a caller's own PYTHONPATH has to survive."""

    environment = harbor_cli._modal_stream_patch_environment({"PYTHONPATH": "/opt/mine"})
    assert environment["PYTHONPATH"] == os.pathsep.join(
        [str(PATCH_DIRECTORY), "/opt/mine"]
    )


def test_environment_defers_to_an_explicit_window() -> None:
    """A run can widen or narrow the window without a code change."""

    environment = harbor_cli._modal_stream_patch_environment(
        {"VERO_MODAL_STREAM_MAX_RETRIES": "5"}
    )
    assert "VERO_MODAL_STREAM_MAX_RETRIES" not in environment


def test_patch_applies_to_the_real_modal_client() -> None:
    """End to end against whichever modal is installed, or skipped if absent.

    The fake pins the seam's shape; only this pins that the shape still matches
    the dependency vero actually patches.
    """

    import pytest

    pytest.importorskip("modal._utils.task_command_router_client")
    stdout, _ = _run_patch(
        {
            "VERO_MODAL_STREAM_RETRY_DELAY_SECS": "2.0",
            "VERO_MODAL_STREAM_RETRY_FACTOR": "1.0",
            "VERO_MODAL_STREAM_MAX_RETRIES": "60",
        },
        """
        from modal._utils.task_command_router_client import TaskCommandRouterClient
        state = {k: v for k, v in TaskCommandRouterClient.__init__.__kwdefaults__.items()
                 if 'stdio' in k}
        state['recompiled'] = (
            TaskCommandRouterClient._stream_stdio_with_retries.__code__.co_filename
        )
        print(state)
        """,
    )
    defaults = eval(stdout.strip().splitlines()[-1])
    assert defaults["stream_stdio_max_retries"] == 60
    assert defaults["stream_stdio_retry_delay_secs"] == 2.0
    # The rewrite is the half that cannot be expressed as a keyword default, and
    # the only proof it took is that the installed method now comes from vero's
    # compile rather than modal's file.
    assert defaults["recompiled"] == "<vero: modal per-outage retry budget>"


def test_sitecustomize_chains_to_a_shadowed_module(tmp_path) -> None:
    """Prepending our directory must not disable an interpreter's own.

    Python imports exactly one module named `sitecustomize`, so shadowing one
    silently is a real risk of this mechanism rather than a hypothetical.
    """

    shadowed = tmp_path / "sitecustomize.py"
    shadowed.write_text("import sys; print('shadowed ran', file=sys.stderr)\n")
    process = subprocess.run(
        [sys.executable, "-c", "pass"],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(PATCH_DIRECTORY), str(tmp_path)]),
        },
        capture_output=True,
        text=True,
    )
    assert "shadowed ran" in process.stderr


def test_module_is_importable_without_side_effects() -> None:
    """`runpy` it directly: no modal, no environment, no exception."""

    runpy.run_path(str(PATCH_DIRECTORY / "sitecustomize.py"), run_name="not_main")


def test_one_bad_value_leaves_every_setting_at_modal_defaults() -> None:
    """All or nothing: the three are only meaningful together.

    Committing the valid ones would pair a raised retry count with modal's
    exponential default, and that does not fail closed. Sleeps double from the
    delay, so ~60 retries means sleeps longer than the run: the crash this fixes
    becomes a hang, which is harder to diagnose than what it replaced.
    """

    stdout, stderr = _run_patch(
        {
            "VERO_MODAL_STREAM_RETRY_DELAY_SECS": "2.0",
            "VERO_MODAL_STREAM_RETRY_FACTOR": "not-a-float",
            "VERO_MODAL_STREAM_MAX_RETRIES": "60",
        },
        FAKE_MODAL,
    )
    defaults = eval(stdout.strip().splitlines()[-1])
    assert defaults["stream_stdio_max_retries"] == 10
    assert defaults["stream_stdio_retry_delay_secs"] == 0.01
    assert defaults["stream_stdio_retry_delay_factor"] == 2
    assert "leaving ALL reconnect settings" in stderr


def test_a_growing_factor_is_refused_rather_than_accepted() -> None:
    """Parseable is not safe: a factor above 1.0 makes the budget unbounded.

    Greptile flagged this on PR #79. With modal's shipped factor of 2 and a
    raised count, the sixtieth sleep is 2 * 2**59 seconds, so the "wider window"
    is really no window at all. It does not fail closed either: the run neither
    finishes nor errors, which is harder to diagnose than the crash this module
    prevents. Refused as a set, so no half-applied policy survives.
    """

    stdout, stderr = _run_patch(
        {
            "VERO_MODAL_STREAM_RETRY_DELAY_SECS": "2.0",
            "VERO_MODAL_STREAM_RETRY_FACTOR": "2",
            "VERO_MODAL_STREAM_MAX_RETRIES": "60",
        },
        FAKE_MODAL,
    )
    defaults = eval(stdout.strip().splitlines()[-1])
    assert defaults["stream_stdio_max_retries"] == 10
    assert defaults["stream_stdio_retry_delay_secs"] == 0.01
    assert defaults["stream_stdio_retry_delay_factor"] == 2
    assert "unbounded rather than paced" in stderr
    assert "Use 1.0 for a flat delay" in stderr


def test_a_flat_factor_is_still_accepted() -> None:
    """The guard must not reject the value the module actually ships."""

    stdout, _ = _run_patch(
        {
            "VERO_MODAL_STREAM_RETRY_DELAY_SECS": "2.0",
            "VERO_MODAL_STREAM_RETRY_FACTOR": "1.0",
            "VERO_MODAL_STREAM_MAX_RETRIES": "60",
        },
        FAKE_MODAL,
    )
    assert eval(stdout.strip().splitlines()[-1])["stream_stdio_max_retries"] == 60
