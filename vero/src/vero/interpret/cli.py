"""`vero interpret` — extract, decompose, and report on optimization runs.

Stages write JSONL and are independently resumable, so a long extraction is never
repeated to re-run a cheap downstream step.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click

from vero.interpret.artifacts.harbor import build as build_harbor
from vero.interpret.artifacts.harbor.adapter import latest_verifier_dir
from vero.interpret.artifacts.harbor import session as session_mod
from vero.interpret.artifacts.harbor.repo import CandidateRepo
from vero.interpret.cache import Cache
from vero.interpret.edits import decompose
from vero.interpret.models import CellRef, Edit, Trajectory

DEFAULT_CACHE = Path.home() / ".cache" / "vero-interpret"


def _load_scope(path: Path | None) -> dict[str, set[str]] | None:
    """{benchmark: {cell, ...}} restricting which runs are in scope."""
    if path is None:
        return None
    raw = json.loads(Path(path).read_text())
    return {bench: set(cells) for bench, cells in raw.items()}


def _in_scope(ref: CellRef, scope: dict[str, set[str]] | None) -> bool:
    if scope is None:
        return True
    return ref.cell in scope.get(ref.benchmark, set())


@click.group()
def main() -> None:
    """Interpretability analysis over optimization runs."""


@main.command()
@click.option("--runs", "roots", multiple=True, required=True, type=click.Path(path_type=Path),
              help="Directory to search for runs. Repeatable.")
@click.option("--cells-file", type=click.Path(path_type=Path),
              help='JSON {"benchmark": ["cell", ...]} restricting scope.')
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE)
@click.option("--out", type=click.Path(path_type=Path), default=Path("trajectories.jsonl"))
@click.option("--refresh", is_flag=True, help="Bypass the cache and re-extract.")
def extract(roots, cells_file, cache_dir, out, refresh) -> None:
    """Canonicalise runs into trajectories."""
    cache = Cache(cache_dir, "harbor-unpack", refresh=refresh)
    adapter = build_harbor(cache)
    scope = _load_scope(cells_file)

    refs = [r for r in adapter.discover(roots) if _in_scope(r, scope)]
    click.echo(f"{len(refs)} cells in scope")

    out = Path(out)
    with out.open("w") as fh:
        for i, ref in enumerate(refs, 1):
            traj = adapter.load(ref)
            fh.write(traj.model_dump_json() + "\n")
            click.echo(
                f"  [{i}/{len(refs)}] {ref.key} "
                f"cands={len(traj.candidates)} evals={len(traj.evaluations)}"
            )
    click.echo(f"wrote {out}  ({cache.stats()})")


@main.command()
@click.option("--in", "src", type=click.Path(path_type=Path), default=Path("trajectories.jsonl"))
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE)
@click.option("--out", type=click.Path(path_type=Path), default=Path("edits.jsonl"))
def edits(src, cache_dir, out) -> None:
    """Split every candidate into symbol-scoped edits."""
    cache = Cache(cache_dir, "harbor-unpack")
    total = 0
    with Path(out).open("w") as fh:
        for line in Path(src).read_text().splitlines():
            traj = Trajectory.model_validate_json(line)
            verifier = latest_verifier_dir(Path(traj.ref.cell_dir))
            if verifier is None:
                continue
            archive = verifier / "session.tar.gz"
            if not archive.is_file():
                continue
            root = session_mod.unpack(archive, cache)
            if root is None:
                continue
            repo_dir = session_mod.find_repo(root)
            if repo_dir is None:
                continue
            repo = CandidateRepo(repo_dir)
            n = 0
            for cand in traj.candidates:
                for edit in decompose(repo, traj.ref.key, cand):
                    fh.write(edit.model_dump_json() + "\n")
                    n += 1
            total += n
            click.echo(f"  {traj.ref.key}: {n} edits")
    click.echo(f"wrote {out}  ({total} edits)")


@main.command()
@click.option("--edits-file", type=click.Path(path_type=Path), default=Path("edits.jsonl"))
@click.option("--trajectories", type=click.Path(path_type=Path),
              default=Path("trajectories.jsonl"))
@click.option("--model", default=None, help="Defaults to a cheap model.")
@click.option("--concurrency", default=16)
@click.option("--limit", default=0, help="Label only the first N edits (a dry run).")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE)
@click.option("--out", type=click.Path(path_type=Path), default=Path("labels.jsonl"))
def label(edits_file, trajectories, model, concurrency, limit, cache_dir, out) -> None:
    """Assign facets to edits. Cached and resumable; re-running costs nothing."""
    import asyncio as _asyncio

    from vero.interpret.config import Settings
    from vero.interpret.labeling.client import AsyncLLM
    from vero.interpret.labeling.labeler import Labeler

    subjects: dict[str, str] = {}
    if Path(trajectories).is_file():
        for line in Path(trajectories).read_text().splitlines():
            traj = Trajectory.model_validate_json(line)
            for cand in traj.candidates:
                subjects[cand.sha] = cand.subject

    rows = [Edit.model_validate_json(l) for l in Path(edits_file).read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    click.echo(f"labelling {len(rows)} edits with {model or 'default model'}")

    settings = Settings.from_env(model=model, concurrency=concurrency,
                                 cache_dir=Path(cache_dir))
    if not settings.api_key:
        raise click.ClickException(
            "no OPENAI_API_KEY found; put it in .env or secrets.env"
        )

    async def run():
        llm = AsyncLLM(settings)
        labeler = Labeler(llm, Cache(Path(cache_dir), "labels"))
        try:
            return await labeler.label_all(
                [(e, subjects.get(e.candidate_sha, "")) for e in rows],
                progress=lambda i, n: click.echo(f"  {i}/{n}"),
            ), labeler
        finally:
            await llm.close()

    labels, labeler = _asyncio.run(run())
    with Path(out).open("w") as fh:
        for lab in labels:
            fh.write(lab.model_dump_json() + "\n")
    click.echo(f"wrote {out} ({len(labels)} labels)  {labeler.stats()}")


@main.command()
@click.option("--labels-file", type=click.Path(path_type=Path), default=Path("labels.jsonl"))
@click.option("--edits-file", type=click.Path(path_type=Path), default=Path("edits.jsonl"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("analysis"))
def report(labels_file, edits_file, out) -> None:
    """Render the figures as one self-contained HTML page."""
    from vero.interpret.analysis import figures

    edits = {}
    for line in Path(edits_file).read_text().splitlines():
        e = json.loads(line)
        edits[e["id"]] = e
    # A label carries only edit_id; the aggregations key on the edit's cell and
    # symbol, so join here rather than duplicating those fields into every label.
    rows = []
    orphans = 0
    for line in Path(labels_file).read_text().splitlines():
        lab = json.loads(line)
        edit = edits.get(lab["edit_id"])
        if edit is None:
            orphans += 1
            continue
        rows.append({**lab, "cell_key": edit["cell_key"], "symbol": edit["symbol"],
                     "symbol_kind": edit["symbol_kind"], "path": edit["path"]})
    if orphans:
        click.echo(f"warning: {orphans} labels had no matching edit and were dropped")
    cells = len({r["cell_key"] for r in rows})
    meta = {"cells": cells, "edits": len(edits), "labels": len(rows)}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    page.write_text(figures.render(rows, edits, meta))
    (out / "labels.jsonl").write_text(Path(labels_file).read_text())
    (out / "edits.jsonl").write_text(Path(edits_file).read_text())
    click.echo(f"wrote {page}  ({cells} cells, {len(rows)} labels)")


@main.command()
@click.option("--in", "src", type=click.Path(path_type=Path), default=Path("edits.jsonl"))
@click.option("--top", default=40, help="Symbols to show per benchmark.")
def symbols(src, top) -> None:
    """Symbol frequency, the input to designing a role vocabulary."""
    per_bench: dict[str, Counter] = {}
    kinds: Counter = Counter()
    cells: dict[str, set[str]] = {}
    for line in Path(src).read_text().splitlines():
        e = json.loads(line)
        bench = e["cell_key"].split("/")[1]
        name = f"{e['path'].split('/')[-1]}::{e['symbol']}"
        per_bench.setdefault(bench, Counter())
        cells.setdefault(f"{bench}|{name}", set()).add(e["cell_key"])
        per_bench[bench][name] += 1
        kinds[e["symbol_kind"]] += 1

    for bench, counter in sorted(per_bench.items()):
        click.echo(f"\n=== {bench}: {sum(counter.values())} edits, {len(counter)} symbols")
        for name, n in counter.most_common(top):
            ncells = len(cells[f"{bench}|{name}"])
            click.echo(f"   {n:>4} edits  {ncells:>3} cells  {name}")
    click.echo("\nsymbol kinds: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))


if __name__ == "__main__":
    main()
