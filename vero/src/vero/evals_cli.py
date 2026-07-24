"""`evals` — the agent-facing CLI over the `.evals/` evaluation context.

One well-named entry point for the whole evaluation loop: run an evaluation
(`evals run`, delegating to the metered sidecar endpoint), then navigate the
disclosure-projected results on disk (`evals list/show/cases/trace/diff`),
browse exposed task resources (`evals tasks/task`), and check what may be run
(`evals plan`).

The viewers are deliberately dumb and unprivileged: `.evals/` is written by the
trusted control plane already projected to the authorized disclosure, so this
module only reads JSON from disk — stdlib + click, no other vero imports. The
runner subcommands import the sidecar client lazily so the viewers work in
contexts without the harbor extra installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

CONTEXT_DIRECTORY = ".evals"
_CELL_WIDTH = 48


# --------------------------------------------------------------------------
# Context discovery and shared helpers
# --------------------------------------------------------------------------


def _find_context(explicit: str | None) -> Path:
    """Locate the `.evals` context directory.

    Order: --context flag, $VERO_CONTEXT_PATH, then walk up from the working
    directory looking for a `.evals/manifest.json`.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if path.name != CONTEXT_DIRECTORY and (path / CONTEXT_DIRECTORY).is_dir():
            path = path / CONTEXT_DIRECTORY
        if not path.is_dir():
            raise click.ClickException(f"context directory not found: {path}")
        return path
    env = os.environ.get("VERO_CONTEXT_PATH")
    if env:
        path = Path(env)
        if path.is_dir():
            return path
        raise click.ClickException(f"$VERO_CONTEXT_PATH is not a directory: {path}")
    current = Path.cwd()
    for candidate in (current, *current.parents):
        path = candidate / CONTEXT_DIRECTORY
        if (path / "manifest.json").is_file():
            return path
    raise click.ClickException(
        f"no {CONTEXT_DIRECTORY}/ directory found from {current} upward; "
        "pass --context or set $VERO_CONTEXT_PATH"
    )


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise click.ClickException(f"missing context file: {path}")
    except json.JSONDecodeError as error:
        raise click.ClickException(f"invalid JSON in {path}: {error}")


def _result_index(context: Path) -> list[dict]:
    document = _load_json(context / "results" / "index.json")
    return list(document.get("evaluations", []))


def _resolve_result(context: Path, identifier: str) -> dict:
    """Match an index entry by evaluation id (or unique prefix) or path digest."""
    entries = _result_index(context)
    matches = [
        entry
        for entry in entries
        if entry.get("evaluation_id") == identifier
        or str(entry.get("evaluation_id", "")).startswith(identifier)
        or str(entry.get("path", "")).startswith(identifier)
    ]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(str(entry.get("evaluation_id")) for entry in entries) or "none"
    kind = "ambiguous" if matches else "unknown"
    raise click.ClickException(
        f"{kind} evaluation {identifier!r}; available ids: {available}"
    )


def _result_document(context: Path, entry: dict) -> dict:
    return _load_json(context / "results" / str(entry["path"]))


def _dig(value: object, *keys: str, default: object = None) -> object:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _result_row(context: Path, entry: dict) -> dict:
    """One `evals list` row, tolerant of every disclosure shape."""
    document = _result_document(context, entry)
    result = document.get("result", {})
    score = _dig(result, "objective", "value")
    if score is None:
        score = _dig(result, "metrics", "score")
    if score is None:
        score = _dig(result, "report", "metrics", "score")
    case_files = result.get("case_files")
    cases = (
        len(case_files)
        if isinstance(case_files, list)
        else _dig(result, "total_cases")
    )
    return {
        "id": entry.get("evaluation_id"),
        "evaluation": entry.get("evaluation"),
        "partition": entry.get("partition"),
        "candidate": entry.get("candidate_id"),
        "disclosure": entry.get("disclosure"),
        "status": _dig(result, "status") or _dig(result, "report", "status"),
        "score": score,
        "cases": cases,
        "errored": _dig(result, "errored_cases"),
        "completed_at": _dig(result, "completed_at"),
    }


def _case_rows(context: Path, entry: dict) -> list[dict]:
    document = _result_document(context, entry)
    result = document.get("result", {})
    case_files = result.get("case_files")
    if not isinstance(case_files, list):
        raise click.ClickException(
            f"evaluation {entry.get('evaluation_id')!r} has disclosure "
            f"{document.get('disclosure')!r}: per-case results are not available"
        )
    root = context / "results" / str(entry["path"])
    rows = []
    for case_file in case_files:
        case_document = _load_json(root.parent / str(case_file["path"]))
        case = case_document.get("result", {})
        errors = case.get("errors") or []
        rows.append(
            {
                "case_id": case.get("case_id"),
                "task": _dig(case, "input", "task_name"),
                "status": case.get("status"),
                "score": _dig(case, "metrics", "score"),
                "error": (
                    errors[0].get("code")
                    if errors
                    else _dig(case, "output", "error_category")
                ),
                "trace": bool(
                    case_document.get("execution_trace_path")
                    or case_document.get("evaluation_trace_path")
                ),
                "_path": str(case_file["path"]),
            }
        )
    return rows


def _resolve_case(context: Path, entry: dict, case_identifier: str) -> dict:
    rows = _case_rows(context, entry)
    matches = [
        row
        for row in rows
        if str(row["case_id"]) == case_identifier
        or str(row["case_id"]).startswith(case_identifier)
    ]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(str(row["case_id"]) for row in rows) or "none"
    kind = "ambiguous" if matches else "unknown"
    raise click.ClickException(
        f"{kind} case {case_identifier!r}; available case ids: {available}"
    )


def _format_cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= _CELL_WIDTH else text[: _CELL_WIDTH - 1] + "…"


def _print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        click.echo("(no rows)")
        return
    cells = [[_format_cell(row.get(column)) for column in columns] for row in rows]
    widths = [
        max(len(column), max((len(line[index]) for line in cells), default=0))
        for index, column in enumerate(columns)
    ]
    click.echo("  ".join(column.ljust(widths[i]) for i, column in enumerate(columns)))
    for line in cells:
        click.echo("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)))


def _paginate(rows: list[dict], sort: str | None, desc: bool, limit: int, offset: int):
    if sort:
        rows = sorted(
            rows,
            key=lambda row: (row.get(sort) is None, row.get(sort)),
            reverse=desc,
        )
    total = len(rows)
    window = rows[offset : offset + limit if limit else None]
    return window, total


def _emit(rows: list[dict], columns: list[str], total: int, offset: int, as_json: bool):
    public = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    if as_json:
        click.echo(json.dumps(public, indent=2, default=str))
        return
    _print_table(public, columns)
    if total > len(rows):
        click.echo(
            f"({total} total; showing {len(rows)} from offset {offset}; "
            "use --limit/--offset)"
        )


_CONTEXT_OPTION = click.option(
    "--context",
    "context_path",
    help=f"Path to the {CONTEXT_DIRECTORY}/ directory (default: discovered).",
)
_JSON_OPTION = click.option("--json", "as_json", is_flag=True, help="Emit JSON rows.")


def _pagination_options(command):
    for option in (
        click.option("--sort", help="Column to sort by."),
        click.option("--desc", is_flag=True, help="Sort descending."),
        click.option("--limit", default=20, show_default=True, type=click.IntRange(0)),
        click.option("--offset", default=0, show_default=True, type=click.IntRange(0)),
    ):
        command = option(command)
    return command


# --------------------------------------------------------------------------
# The command group (runner subcommands attach lazily)
# --------------------------------------------------------------------------


_LAZY_RUNNERS = {"run", "result", "submit"}


class _EvalsGroup(click.Group):
    """Attach sidecar runner commands only when asked for, so the on-disk
    viewers work without the harbor extra installed."""

    def list_commands(self, ctx):
        return sorted({*super().list_commands(ctx), *_LAZY_RUNNERS})

    def get_command(self, ctx, name):
        command = super().get_command(ctx, name)
        if command is not None or name not in _LAZY_RUNNERS:
            return command
        try:
            from vero.harbor import cli as harbor_cli
        except ImportError as error:
            raise click.ClickException(
                f"`evals {name}` needs the evaluation sidecar client "
                f"(vero[harbor]): {error}"
            )
        return {
            "run": harbor_cli.evaluate_command,
            "result": harbor_cli.evaluation_result_command,
            "submit": harbor_cli.submit_command,
        }[name]


@click.group(cls=_EvalsGroup)
def evals() -> None:
    """Run evaluations and navigate their results.

    Everything you are authorized to see lives in the read-only `.evals/`
    directory: `results/` (past evaluation results), `tasks/` (exposed task
    resources), `candidates/` (prior program versions), and `plan.json`
    (what you may evaluate, and remaining budget).

    Typical loop: `evals plan` -> edit + commit -> `evals run --detach` ->
    `evals status JOB` -> `evals list` -> `evals diff BASELINE CANDIDATE` ->
    `evals cases ID --sort score` -> `evals trace ID CASE`.
    """


@evals.command("status")
@click.argument("job_id", required=False)
def status_command(job_id):
    """Show evaluation access and budgets, or one detached job's status."""
    try:
        from vero.harbor.cli import _request
    except ImportError as error:
        raise click.ClickException(
            f"`evals status` needs the evaluation sidecar client (vero[harbor]): {error}"
        )
    path = f"/eval/jobs/{job_id}" if job_id else "/status"
    click.echo(json.dumps(_request("GET", path), indent=2))


@evals.command("list")
@_CONTEXT_OPTION
@_pagination_options
@_JSON_OPTION
def list_command(context_path, sort, desc, limit, offset, as_json):
    """List past evaluation results, one row per evaluation."""
    context = _find_context(context_path)
    rows = [_result_row(context, entry) for entry in _result_index(context)]
    window, total = _paginate(rows, sort, desc, limit, offset)
    _emit(
        window,
        [
            "id",
            "evaluation",
            "partition",
            "candidate",
            "disclosure",
            "status",
            "score",
            "cases",
            "completed_at",
        ],
        total,
        offset,
        as_json,
    )


@evals.command("show")
@click.argument("evaluation_id")
@_CONTEXT_OPTION
@click.option("--raw", is_flag=True, help="Dump the full stored document.")
def show_command(evaluation_id, context_path, raw):
    """Show one evaluation result in detail."""
    context = _find_context(context_path)
    entry = _resolve_result(context, evaluation_id)
    document = _result_document(context, entry)
    if raw:
        click.echo(json.dumps(document, indent=2, default=str))
        return
    compact = json.loads(json.dumps(document, default=str))
    result = compact.get("result", {})
    case_files = result.get("case_files")
    if isinstance(case_files, list):
        result["case_files"] = (
            f"({len(case_files)} cases; use `evals cases {entry['evaluation_id']}`)"
        )
    artifacts = _dig(result, "report", "artifacts")
    if isinstance(artifacts, list) and artifacts:
        result["report"]["artifacts"] = (
            f"({len(artifacts)} artifacts under "
            f"results/{entry['path'].rsplit('/', 1)[0]}/artifacts/)"
        )
    click.echo(json.dumps(compact, indent=2))


@evals.command("cases")
@click.argument("evaluation_id")
@_CONTEXT_OPTION
@_pagination_options
@_JSON_OPTION
def cases_command(evaluation_id, context_path, sort, desc, limit, offset, as_json):
    """List one evaluation's per-case results (full disclosure only)."""
    context = _find_context(context_path)
    entry = _resolve_result(context, evaluation_id)
    rows = _case_rows(context, entry)
    window, total = _paginate(rows, sort, desc, limit, offset)
    _emit(
        window,
        ["case_id", "task", "status", "score", "error", "trace"],
        total,
        offset,
        as_json,
    )


@evals.command("trace")
@click.argument("evaluation_id")
@click.argument("case_id")
@_CONTEXT_OPTION
@click.option("--span", type=click.IntRange(0), help="Show one span by index.")
@click.option("--chars", default=10_000, show_default=True, type=click.IntRange(1))
@click.option("--char-offset", default=0, show_default=True, type=click.IntRange(0))
def trace_command(evaluation_id, case_id, context_path, span, chars, char_offset):
    """Summarize a case's trace, or window one span of it.

    Without --span: span count, shapes, and sizes, plus the case's artifact
    files (read those directly with your file tools). With --span N: that
    span's JSON, windowed by --char-offset/--chars.
    """
    context = _find_context(context_path)
    entry = _resolve_result(context, evaluation_id)
    case_row = _resolve_case(context, entry, case_id)
    case_root = (context / "results" / str(entry["path"])).parent / Path(
        str(case_row["_path"])
    ).parent
    case_document = _load_json(case_root / "result.json")

    trace = []
    trace_name = case_document.get("execution_trace_path")
    if trace_name:
        loaded = _load_json(case_root / str(trace_name))
        trace = loaded if isinstance(loaded, list) else [loaded]

    if span is not None:
        if not trace:
            raise click.ClickException("this case has no execution trace file")
        if span >= len(trace):
            raise click.ClickException(
                f"span {span} out of range; trace has {len(trace)} spans"
            )
        text = json.dumps(trace[span], indent=2, default=str)
        window = text[char_offset : char_offset + chars]
        click.echo(
            f"span {span}/{len(trace) - 1}, chars {char_offset}-"
            f"{char_offset + len(window)} of {len(text)}"
        )
        click.echo(window)
        return

    summary: dict[str, object] = {"case_id": case_row["case_id"]}
    if trace:
        shapes: dict[str, dict[str, int]] = {}
        total_chars = 0
        for item in trace:
            if isinstance(item, dict):
                key = "dict(" + ",".join(sorted(item)) + ")"
            elif isinstance(item, list):
                key = f"list(len={len(item)})"
            else:
                key = type(item).__name__
            size = len(json.dumps(item, default=str))
            total_chars += size
            shape = shapes.setdefault(key, {"count": 0, "chars": 0})
            shape["count"] += 1
            shape["chars"] += size
        summary["execution_trace"] = {
            "spans": len(trace),
            "total_chars": total_chars,
            "shapes": shapes,
            "read_with": f"evals trace {entry['evaluation_id']} "
            f"{case_row['case_id']} --span N",
        }
    else:
        summary["execution_trace"] = None

    artifacts = []
    for artifact in _dig(case_document, "result", "artifacts", default=[]) or []:
        relative = str(artifact.get("path", ""))
        on_disk = (
            context / "results" / str(entry["path"])
        ).parent / "artifacts" / relative
        artifacts.append(
            {
                "path": f"results/{str(entry['path']).rsplit('/', 1)[0]}"
                f"/artifacts/{relative}",
                "bytes": on_disk.stat().st_size if on_disk.is_file() else None,
                "description": artifact.get("description"),
            }
        )
    summary["artifacts"] = artifacts
    click.echo(json.dumps(summary, indent=2))


@evals.command("diff")
@click.argument("baseline_id")
@click.argument("candidate_id")
@_CONTEXT_OPTION
@_JSON_OPTION
def diff_command(baseline_id, candidate_id, context_path, as_json):
    """Per-case score deltas between two evaluations (matched by case id)."""
    context = _find_context(context_path)
    baseline_entry = _resolve_result(context, baseline_id)
    candidate_entry = _resolve_result(context, candidate_id)
    baseline = {row["case_id"]: row for row in _case_rows(context, baseline_entry)}
    candidate = {row["case_id"]: row for row in _case_rows(context, candidate_entry)}
    rows = []
    for case_id in sorted({*baseline, *candidate}, key=str):
        before = _dig(baseline.get(case_id, {}), "score")
        after = _dig(candidate.get(case_id, {}), "score")
        if case_id not in baseline or case_id not in candidate:
            verdict = "unmatched"
        elif before is None or after is None:
            verdict = "unscored"
        elif after > before:
            verdict = "improved"
        elif after < before:
            verdict = "regressed"
        else:
            verdict = "unchanged"
        rows.append(
            {
                "case_id": case_id,
                "baseline": before,
                "candidate": after,
                "delta": (after - before)
                if isinstance(before, (int, float)) and isinstance(after, (int, float))
                else None,
                "verdict": verdict,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    if as_json:
        click.echo(json.dumps({"cases": rows, "summary": counts}, indent=2))
        return
    _print_table(rows, ["case_id", "baseline", "candidate", "delta", "verdict"])
    click.echo(json.dumps(counts))


@evals.command("plan")
@_CONTEXT_OPTION
@_JSON_OPTION
def plan_command(context_path, as_json):
    """Show the evaluations you may run, their rules, and remaining budget."""
    context = _find_context(context_path)
    document = _load_json(context / "plan.json")
    rows = []
    for evaluation in document.get("evaluations", []):
        budget = evaluation.get("budget") or {}
        rows.append(
            {
                "evaluation": evaluation.get("name"),
                "partition": evaluation.get("partition"),
                "can_evaluate": evaluation.get("agent_can_evaluate"),
                "selection": evaluation.get("agent_selection"),
                "disclosure": evaluation.get("disclosure"),
                "runs_left": budget.get("remaining_runs"),
                "cases_left": budget.get("remaining_cases"),
            }
        )
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    _print_table(
        rows,
        [
            "evaluation",
            "partition",
            "can_evaluate",
            "selection",
            "disclosure",
            "runs_left",
            "cases_left",
        ],
    )


def _task_sets(context: Path) -> list[dict]:
    document = _load_json(context / "tasks" / "index.json")
    return list(document.get("case_resources", []))


@evals.command("tasks")
@click.argument("evaluation_set", required=False)
@_CONTEXT_OPTION
@_JSON_OPTION
def tasks_command(evaluation_set, context_path, as_json):
    """List exposed task sets, or one set's task ids and resource paths."""
    context = _find_context(context_path)
    sets = _task_sets(context)
    if evaluation_set is None:
        rows = []
        for item in sets:
            resources = _load_json(
                context / "tasks" / str(item["path"]) / "resources" / "index.json"
            )
            rows.append(
                {
                    "evaluation": _dig(item, "evaluation_set", "name"),
                    "partition": _dig(item, "evaluation_set", "partition"),
                    "tasks": len(resources.get("cases", [])),
                    "path": f"tasks/{item['path']}/resources/",
                }
            )
        if as_json:
            click.echo(json.dumps(rows, indent=2))
        else:
            _print_table(rows, ["evaluation", "partition", "tasks", "path"])
        return
    matches = [
        item
        for item in sets
        if evaluation_set
        in (
            _dig(item, "evaluation_set", "name"),
            _dig(item, "evaluation_set", "partition"),
            str(item.get("path")),
        )
    ]
    if len(matches) != 1:
        available = ", ".join(
            f"{_dig(item, 'evaluation_set', 'name')}/"
            f"{_dig(item, 'evaluation_set', 'partition')}"
            for item in sets
        )
        raise click.ClickException(
            f"unknown or ambiguous task set {evaluation_set!r}; available: "
            f"{available or 'none'}"
        )
    root = f"tasks/{matches[0]['path']}/resources"
    resources = _load_json(context / "tasks" / str(matches[0]["path"]) / "resources" / "index.json")
    rows = [
        {"case_id": case.get("case_id"), "path": f"{root}/{case.get('path')}"}
        for case in resources.get("cases", [])
    ]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    else:
        _print_table(rows, ["case_id", "path"])
        click.echo("(read task files directly with your file tools)")


def main() -> None:
    evals()
