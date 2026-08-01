"""The Modal stdio reconnect budget vero applies to its harbor subprocess.

Modal budgets stdio reconnects per stream (10, never replenished) with a 0.01s
doubling backoff, so the whole budget is spent in ~10s of outage and the next
drop kills the trial. The three knobs are constructor keywords that
`TaskCommandRouterClient._connect` never forwards and `modal/config.py` never
exposes, so vero reaches them through a `sitecustomize` on the subprocess's
PYTHONPATH. These tests pin the seam, because a silent no-op here reads exactly
like a widened budget.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
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

    script = textwrap.dedent(modal_source)
    process = subprocess.run(
        [sys.executable, "-c", script],
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


FAKE_MODAL = """
    import sys, types
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
            pass

    module.TaskCommandRouterClient = TaskCommandRouterClient
    sys.modules["modal"] = package
    sys.modules["modal._utils"] = utils
    sys.modules["modal._utils.task_command_router_client"] = module

    import sitecustomize
    sitecustomize._apply()
    print({k: v for k, v in TaskCommandRouterClient.__init__.__kwdefaults__.items()})
"""


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
    # doubling one would outgrow the run instead of pacing it.
    assert "120s outage tolerated" in stderr


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
        print({k: v for k, v in TaskCommandRouterClient.__init__.__kwdefaults__.items()
               if 'stdio' in k})
        """,
    )
    defaults = eval(stdout.strip().splitlines()[-1])
    assert defaults["stream_stdio_max_retries"] == 60
    assert defaults["stream_stdio_retry_delay_secs"] == 2.0


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
