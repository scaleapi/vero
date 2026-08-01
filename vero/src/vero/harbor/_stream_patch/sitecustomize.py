"""Widen Modal's stdio reconnect budget inside the harbor subprocess.

Harbor reads a whole agent phase through one Modal stdio stream. Modal gives that
stream a reconnect budget of ``stream_stdio_max_retries`` (10) for the *life of
the stream*, never replenished: on a successful chunk only the backoff delay
resets, not the count (``modal/_utils/task_command_router_client.py``, the
``delay_secs = self.stream_stdio_retry_delay_secs`` line inside the read loop).
With the shipped defaults of 0.01s and a doubling factor, those ten attempts are
spent in 10.23 seconds of continuous outage, so a multi-hour optimization is
protected against ten seconds of network trouble and the next drop is fatal.

The three knobs are constructor keywords, and ``TaskCommandRouterClient._connect``
never passes them, so nothing in Modal's public surface can change them: not
``init()``, not ``init_v2()``, and there is no entry for them in ``modal/config.py``
so no environment variable either. They are keyword-only parameters, which means
their defaults live in a plain writable dict on the function object, and that is
the seam this module uses.

Loaded by being on ``PYTHONPATH`` when vero spawns harbor (see
``vero.harbor.cli``). Inert unless vero sets the environment variables below, so
importing it in any other context does nothing.

Raising the count alone would not help: the delay doubles, so past roughly the
seventeenth attempt a single sleep already outlasts the run. Pacing is the point,
hence a factor of 1.0 and a flat delay.
"""

import os
import sys

_DELAY_ENV = "VERO_MODAL_STREAM_RETRY_DELAY_SECS"
_FACTOR_ENV = "VERO_MODAL_STREAM_RETRY_FACTOR"
_RETRIES_ENV = "VERO_MODAL_STREAM_MAX_RETRIES"

_SETTINGS = (
    ("stream_stdio_retry_delay_secs", _DELAY_ENV, float),
    ("stream_stdio_retry_delay_factor", _FACTOR_ENV, float),
    ("stream_stdio_max_retries", _RETRIES_ENV, int),
)


def _warn(message: str) -> None:
    """Report to stderr, which harbor captures into the run's job log.

    A patch of a private module in a pinned dependency has to be loud when it
    stops applying. Silence would read as "the wider budget is in effect" while
    the run is back on ten seconds of tolerance.
    """

    print(f"vero: modal stream patch: {message}", file=sys.stderr, flush=True)


def _chain_to_shadowed_sitecustomize() -> None:
    """Run any ``sitecustomize`` this module is shadowing on ``sys.path``.

    Python imports exactly one module by that name. Prepending our directory to
    PYTHONPATH would otherwise silently disable an interpreter's own, so find the
    next one along the path and execute it first.
    """

    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        try:
            if not entry or os.path.abspath(entry) == here:
                continue
            candidate = os.path.join(entry, "sitecustomize.py")
            if not os.path.isfile(candidate):
                continue
            spec = importlib.util.spec_from_file_location("_vero_shadowed", candidate)
            if spec is None or spec.loader is None:
                continue
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return
        except Exception as error:  # never let chaining break the run
            _warn(f"could not chain to {candidate}: {type(error).__name__}: {error}")
            return


def _apply() -> None:
    requested = {
        name: os.environ[env]
        for name, env, _ in _SETTINGS
        if os.environ.get(env) not in (None, "")
    }
    if not requested:
        return  # vero did not ask; stay inert

    try:
        from modal._utils.task_command_router_client import TaskCommandRouterClient
    except Exception as error:
        _warn(f"modal client not importable, defaults unchanged: {error}")
        return

    defaults = getattr(TaskCommandRouterClient.__init__, "__kwdefaults__", None)
    if not isinstance(defaults, dict):
        _warn("client __init__ has no keyword defaults; modal API changed")
        return

    missing = [name for name, _, _ in _SETTINGS if name not in defaults]
    if missing:
        _warn(f"modal API changed, absent knobs {missing}; defaults unchanged")
        return

    applied = {}
    for name, env, cast in _SETTINGS:
        raw = os.environ.get(env)
        if raw in (None, ""):
            continue
        try:
            defaults[name] = cast(raw)
        except ValueError:
            _warn(f"{env}={raw!r} is not a valid {cast.__name__}; left at default")
            continue
        applied[name] = defaults[name]

    if applied:
        delay = applied.get("stream_stdio_retry_delay_secs")
        factor = applied.get("stream_stdio_retry_delay_factor")
        tries = applied.get("stream_stdio_max_retries")
        window = (
            f", ~{delay * tries:.0f}s outage tolerated"
            if None not in (delay, tries) and factor == 1.0
            else ""
        )
        _warn(f"applied {applied}{window}")


_chain_to_shadowed_sitecustomize()
_apply()
