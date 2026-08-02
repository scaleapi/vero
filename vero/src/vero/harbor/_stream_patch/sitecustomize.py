"""Make Modal's stdio reconnect budget per-outage inside the harbor subprocess.

Harbor reads a whole agent phase through one Modal stdio stream. A measured
optimizer phase held that single stream open for 3h19m.

Modal gives the stream a reconnect budget of ``stream_stdio_max_retries`` and
then never replenishes it. In modal 1.5.3,
``modal/_utils/task_command_router_client.py:769`` sets
``num_retries_remaining = self.stream_stdio_max_retries`` once, above the
``while True``; the success path at :801-802 resets only the backoff delay
(``# Reset retry backoff after any successful chunk.``), not the count. Every
reconnect over the life of the stream draws down the same counter, so what reads
like "retries per problem" is really a LIFETIME cap on a generator that lives as
long as the exec. Hours of healthy streaming in between earn nothing back.

That is the defect. An earlier version of this patch raised the count instead,
which treats a symptom: any finite lifetime cap is the wrong shape for a stream
measured in hours, because it is spent by unrelated blips spread across the run
rather than by the outage that actually kills it. The fix is per-outage
semantics, and the correct edit is one line next to :802 restoring the count
alongside the delay.

Two seams, applied together:

1. The three numeric knobs are constructor keywords, and
   ``TaskCommandRouterClient._connect`` never passes them, so nothing in Modal's
   public surface can change them: not ``init()``, not ``init_v2()``, and
   ``modal/config.py`` has no entry so no environment variable either. They are
   keyword-only, so their defaults live in a plain writable dict on the function
   object, which is the seam used here.
2. The count reset lives inside the method body, where no keyword default can
   reach it. So this module reads the shipped source of
   ``_stream_stdio_with_retries``, verifies its shape against the AST it expects,
   inserts the one missing line, and rebinds the recompiled coroutine.

Seam 2 is the fragile one and deserves a plain statement of that: it patches a
private async generator of a pinned dependency by recompiling its source. It is
structured to fail loudly and completely rather than partially. Every assumption
(the method exists, is undecorated, has no closure, assigns the delay exactly
twice with exactly one of those inside the read loop, assigns the count exactly
once outside it) is asserted before anything is written, and any surprise leaves
Modal's own method installed untouched. A modal upgrade that reshapes the loop
therefore degrades to seam 1 alone with a warning in the job log, never to a
silently mismatched hybrid.

Loaded by being on ``PYTHONPATH`` when vero spawns harbor (see
``vero.harbor.cli``). Inert unless vero sets the environment variables below, so
importing it in any other context does nothing.
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

# Names this module has to recognise in modal's source to place the reset. Kept
# as constants because they are simultaneously the thing asserted and the thing
# written: if a modal upgrade renames either local, the assertions below fail and
# nothing is written, rather than an edit landing next to a stale name.
_METHOD_NAME = "_stream_stdio_with_retries"
_COUNT_LOCAL = "num_retries_remaining"
_COUNT_ATTRIBUTE = "stream_stdio_max_retries"
_DELAY_LOCAL = "delay_secs"
_DELAY_ATTRIBUTE = "stream_stdio_retry_delay_secs"


def _warn(message: str) -> None:
    """Report to stderr, which harbor captures into the run's job log.

    A patch of a private module in a pinned dependency has to be loud when it
    stops applying. Silence would read as "the budget is per-outage" while the
    run is back on a lifetime budget it can exhaust in its first hour.
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


def _assigns_attribute(node, local: str, attribute: str) -> bool:
    """True for exactly ``<local> = self.<attribute>``, nothing looser.

    Matched on the AST rather than on text so that reformatting, a reworded
    comment, or a changed indent cannot silently move where the reset lands.
    """

    import ast

    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == local
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == attribute
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    )


def _within_read_loop(tree) -> set:
    """Ids of nodes under an ``async for``, i.e. modal's chunk-consuming loop.

    This is how the initialisation of a local is told apart from its per-chunk
    reset without depending on line numbers or on the comment above it.
    """

    import ast

    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFor):
            inside.update(id(child) for child in ast.walk(node))
    return inside


def _rewrite_source(source: str):
    """Insert the count reset beside modal's delay reset, or return None.

    None means an assumption failed and the caller must leave modal's own method
    in place. Every check here guards a shape this module would otherwise be
    editing blind.
    """

    import ast

    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.AsyncFunctionDef):
        _warn(f"{_METHOD_NAME} is not a lone async def; budget stays lifetime-scoped")
        return None
    function = tree.body[0]
    if function.name != _METHOD_NAME or function.decorator_list:
        # A decorator would be dropped by recompiling the def alone, which would
        # quietly change behaviour rather than fail. Refuse instead.
        _warn(f"{_METHOD_NAME} is decorated or renamed; budget stays lifetime-scoped")
        return None

    in_loop = _within_read_loop(tree)
    nodes = list(ast.walk(tree))
    delays = [n for n in nodes if _assigns_attribute(n, _DELAY_LOCAL, _DELAY_ATTRIBUTE)]
    counts = [n for n in nodes if _assigns_attribute(n, _COUNT_LOCAL, _COUNT_ATTRIBUTE)]
    resets = [n for n in delays if id(n) in in_loop]
    # modal 1.5.3: the delay is assigned twice (once above `while True`, once per
    # successful chunk) and the count exactly once, above the loop. Anything else
    # means the loop was reshaped and the premise of this patch no longer holds.
    if (
        len(delays) != 2
        or len(resets) != 1
        or len(counts) != 1
        or id(counts[0]) in in_loop
    ):
        _warn(
            f"modal's retry loop changed shape ({len(delays)} delay assignments, "
            f"{len(resets)} of them per-chunk, {len(counts)} count assignments); "
            "budget stays lifetime-scoped"
        )
        return None

    reset = resets[0]
    lines = source.splitlines(keepends=True)
    lines.insert(
        reset.end_lineno,
        " " * reset.col_offset
        + f"{_COUNT_LOCAL} = self.{_COUNT_ATTRIBUTE}"
        + "  # vero: per-outage budget, see vero/harbor/_stream_patch\n",
    )
    patched = "".join(lines)

    # Re-read what was actually produced. Cheap, and it is the difference between
    # "we intended to add a reset" and "a reset exists inside the read loop".
    verify = ast.parse(patched)
    verify_loop = _within_read_loop(verify)
    landed = [
        n
        for n in ast.walk(verify)
        if _assigns_attribute(n, _COUNT_LOCAL, _COUNT_ATTRIBUTE) and id(n) in verify_loop
    ]
    if len(landed) != 1:
        _warn("rewrite did not land the count reset in the read loop; not applying")
        return None
    return patched


def _make_budget_per_outage(client) -> bool:
    """Rebind ``_stream_stdio_with_retries`` so a good chunk restores the count.

    Recompiled against modal's own module globals (not a copy) so the private
    names the body reads -- ``sr_pb2``, ``RETRYABLE_GRPC_STATUS_CODES``,
    ``ExecTimeoutError``, the module ``logger`` -- resolve to the live objects,
    and so a later modal-side rebinding of any of them is still seen.
    """

    import inspect
    import textwrap

    original = client.__dict__.get(_METHOD_NAME)
    if original is None or not inspect.isasyncgenfunction(original):
        _warn(
            f"{_METHOD_NAME} absent or not an async generator; "
            "budget stays lifetime-scoped"
        )
        return False
    if original.__closure__ is not None:
        # A closure cell (``super()``, ``__class__``) cannot be rebuilt by
        # recompiling the source standalone; the result would raise at call time.
        _warn(f"{_METHOD_NAME} closes over cells; budget stays lifetime-scoped")
        return False

    try:
        source = textwrap.dedent(inspect.getsource(original))
    except (OSError, TypeError) as error:
        _warn(
            f"{_METHOD_NAME} source unavailable ({error}); budget stays lifetime-scoped"
        )
        return False

    patched_source = _rewrite_source(source)
    if patched_source is None:
        return False

    globals_ = original.__globals__
    if _METHOD_NAME in globals_:
        # Executing the def would clobber a module-level name of the same spelling.
        _warn(f"{_METHOD_NAME} shadows a module global; budget stays lifetime-scoped")
        return False
    try:
        code = compile(patched_source, "<vero: modal per-outage retry budget>", "exec")
        exec(code, globals_)
        patched = globals_.pop(_METHOD_NAME)
    except Exception as error:
        globals_.pop(_METHOD_NAME, None)
        _warn(
            f"could not recompile {_METHOD_NAME} "
            f"({type(error).__name__}: {error}); budget stays lifetime-scoped"
        )
        return False

    if not inspect.isasyncgenfunction(patched):
        _warn(f"recompiled {_METHOD_NAME} is not an async generator; not applying")
        return False
    setattr(client, _METHOD_NAME, patched)
    return True


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

    # Parse every value before writing any, because the three are only meaningful
    # as a set. Committing the valid ones would pair a raised retry count with
    # Modal's exponential default, and that combination does not fail closed: the
    # sleeps double from the delay, so ~60 retries means sleeps longer than the
    # run. A bad factor would turn the crash this fixes into a hang.
    applied = {}
    for name, env, cast in _SETTINGS:
        raw = os.environ.get(env)
        if raw in (None, ""):
            continue
        try:
            applied[name] = cast(raw)
        except ValueError:
            _warn(
                f"{env}={raw!r} is not a valid {cast.__name__}; "
                "leaving ALL reconnect settings at modal's defaults"
            )
            return

    # Parseable is not the same as safe. A factor above 1.0 makes each sleep grow
    # from the one before, so a raised retry count stops being a wider window and
    # becomes an unbounded one: at the shipped factor of 2, the sixtieth sleep is
    # 2 * 2**59 seconds. That does not fail closed. It converts the crash this
    # module exists to prevent into a hang, which is strictly harder to diagnose
    # because the run neither finishes nor reports an error.
    #
    # Rejected rather than clamped. The whole premise here is that a long-lived
    # stream needs PACING, and a flat delay is the only shape whose worst case is
    # readable off the two numbers. Silently rewriting a caller's explicit choice
    # would leave the log agreeing with a setting that is not in force.
    factor = applied.get("stream_stdio_retry_delay_factor")
    if factor is not None and factor > 1.0:
        _warn(
            f"{_FACTOR_ENV}={factor} grows every sleep from the last, so the "
            "budget is unbounded rather than paced; leaving ALL reconnect "
            "settings at modal's defaults. Use 1.0 for a flat delay."
        )
        return

    defaults.update(applied)

    # After the knobs, and deliberately not gated on each other. If the rewrite
    # fails, the knobs alone still leave a strictly wider budget than modal ships
    # (a flat 2s beats a 0.01s doubling spent in 10.23s), so withholding them on
    # that failure would be worse for the run, not safer. They are independent
    # improvements, and the reported scope below says which one is in force.
    per_outage = _make_budget_per_outage(TaskCommandRouterClient)

    if applied:
        delay = applied.get("stream_stdio_retry_delay_secs")
        factor = applied.get("stream_stdio_retry_delay_factor")
        tries = applied.get("stream_stdio_max_retries")
        scope = "per outage" if per_outage else "per stream, lifetime"
        window = (
            f", ~{delay * tries:.0f}s tolerated {scope}"
            if None not in (delay, tries) and factor == 1.0
            else f", budget is {scope}"
        )
        _warn(f"applied {applied}{window}")


_chain_to_shadowed_sitecustomize()
_apply()
