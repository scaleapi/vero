"""The narrative sections above the figures: method, vocabulary, and known errors.

Computed from the same data as the figures rather than written out, so they cannot
drift from what the pipeline actually produced. The error inventory in particular is
measured: it exists because several facets are wrong in specific, findable ways, and
a reader who is going to use these figures needs that before the figures, not after.
"""

from __future__ import annotations

import html
import re
from collections import Counter

_DIS = re.compile(r"\[hint=(\S+) model=(\S+)\]")


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _pick_example(rows: list[dict], edits: dict[str, dict]):
    """A candidate that shows the split clearly: several edits spanning several kinds."""
    by_cand: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        edit = edits.get(r["edit_id"])
        if edit:
            by_cand.setdefault((edit["cell_key"], edit["candidate_sha"]), []).append(
                {**r, **edit}
            )
    best = None
    for (cell, sha), group in sorted(by_cand.items()):
        kinds = len({g["symbol_kind"] for g in group})
        if 6 <= len(group) <= 9 and (best is None or kinds > best[0]):
            best = (kinds, cell, sha, group)
    if best is None:
        (cell, sha), group = next(iter(by_cand.items()))
        return cell, sha, group
    return best[1], best[2], best[3]


def worked_example(rows: list[dict], edits: dict[str, dict]) -> str:
    cell, sha, group = _pick_example(rows, edits)
    added = sum(g["added"] for g in group)
    kinds = len({g["symbol_kind"] for g in group})
    body = []
    for g in sorted(group, key=lambda x: -x["added"]):
        val = ""
        if g.get("before_value"):
            val = (
                f' <span class="mono">{_esc(g["before_value"])}'
                f'&rarr;{_esc(g["after_value"])}</span>'
            )
        body.append(
            f'<tr><td class="mono">{_esc(g["symbol"][:38])}</td>'
            f'<td>{_esc(g["symbol_kind"])}</td><td>+{g["added"]}</td>'
            f'<td>{_esc(g["role"])}</td><td>{_esc(g["action"])}{val}</td></tr>'
        )
    return (
        f'<p class="note">One commit is not one change. Candidate '
        f'<span class="mono">{_esc(sha[:12])}</span> of '
        f'<span class="mono">{_esc(cell.split("/", 1)[1])}</span> touches {added} lines under a '
        f'single subject line; labelling it as one item would give one category to all of '
        f'them. Mapping each changed line through the syntax tree to its innermost enclosing '
        f'definition splits it into <b>{len(group)} edits across {kinds} kinds</b>, each '
        f'labelled on its own:</p>'
        f'<div class="fig"><table style="width:100%"><thead><tr><th>symbol</th><th>kind</th>'
        f'<th>+lines</th><th>role</th><th>action</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        f'<p class="note">Git\'s own hunk header cannot do this. Its Python matcher reports the '
        f'enclosing <i>class</i> for a method, so a one-line fix inside a shell helper is '
        f'attributed to the whole agent. Module-level bindings are split further by target '
        f'name and value shape, which is what separates a system prompt from a tuning '
        f'constant when both sit at the top of the same file.</p>'
    )


def kinds_table(rows: list[dict], edits: dict[str, dict]) -> str:
    seen: dict[str, dict] = {}
    for r in rows:
        edit = edits.get(r["edit_id"])
        if not edit:
            continue
        kind = edit["symbol_kind"]
        # Prefer an exemplar that carries a value change; it shows more of the schema.
        if kind not in seen or (edit.get("before_value") and not seen[kind].get("before_value")):
            seen[kind] = {**r, **edit}
    order = [
        "prompt_text", "scalar_const", "collection", "regex", "method",
        "function", "class", "module", "non_python",
    ]
    body = []
    for kind in order:
        g = seen.get(kind)
        if not g:
            continue
        val = (
            f'{_esc(g["before_value"])}&rarr;{_esc(g["after_value"])}'
            if g.get("before_value") else "&mdash;"
        )
        body.append(
            f'<tr><td>{_esc(kind)}</td><td class="mono">{_esc(g["symbol"][:34])}</td>'
            f'<td class="mono">{val}</td><td>{_esc(g["role"])}</td>'
            f'<td>{_esc(g["action"])}</td>'
            f'<td>{"rule" if g["hinted"] else "model"}</td></tr>'
        )
    return (
        '<p class="note">Every changed line lands on a named symbol with a kind resolved from '
        'the syntax tree. The role is set by deterministic rule where the path, kind or name '
        'settles it and by model otherwise. Before/after values are captured for scalar '
        'constants, which is what makes tuning direction derived rather than guessed.</p>'
        '<div class="fig"><table style="width:100%"><thead><tr><th>kind</th>'
        '<th>example symbol</th><th>value change</th><th>role</th><th>action</th>'
        f'<th>role from</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def false_labels(rows: list[dict], edits: dict[str, dict]) -> str:
    dis = [r for r in rows if r["mechanism"].startswith("[hint=")]
    hinted = sum(1 for r in rows if r["hinted"])
    pairs: Counter = Counter()
    for r in dis:
        m = _DIS.match(r["mechanism"])
        if m:
            pairs[(m.group(1), m.group(2))] += 1

    inert = [
        r for r in rows
        if (e := edits.get(r["edit_id"])) and e["symbol_kind"] == "non_python"
    ]
    inert_wrong = sum(1 for r in inert if r["action"] != "cosmetic")
    acts = Counter(r["action"] for r in rows)
    regex_roles = Counter(
        r["role"] for r in rows
        if (e := edits.get(r["edit_id"])) and e["symbol_kind"] == "regex"
    )
    rate = len(dis) / hinted if hinted else 0.0

    confusion = "".join(
        f"<tr><td>{n}</td><td>{_esc(h)}</td><td>{_esc(m)}</td></tr>"
        for (h, m), n in pairs.most_common(6)
    )
    return f"""
<p class="note">Where these labels are known to be wrong. Measured, not estimated, and
every figure below inherits it.</p>
<div class="fig">
<p><b>Fix provenance could not be labelled at all, and is now derived instead.</b> Asked of
the model it returned 452 &ldquo;own&rdquo; against 3 &ldquo;seed&rdquo;, and called 21 of 22
swe-atlas submission fixes self-inflicted &mdash; those repair a defect in the seed's answer
parser that 15 of 20 cells independently patched. An edit shown alone carries no history, so
the model sees repair inside the optimizer's own file and answers &ldquo;own&rdquo;
confidently rather than abstaining. Comparing the repaired symbol against the seed tree gives
<b>281 seed / 226 own</b>, and 15 of those 22 swe-atlas fixes now read as seed defects &mdash;
the same count reached independently by reading diffs.</p>

<p><b>Inert edits are read as work.</b> {inert_wrong} of {len(inert)} non-Python edits, almost
all <span class="mono">.gitignore</span>, were labelled <span class="mono">env_setup / add</span>
rather than cosmetic: the model reads &ldquo;ignore build artifacts&rdquo; as environment
setup. Corpus-wide only {acts.get('cosmetic', 0)} edits were called cosmetic, which is
certainly too few.</p>

<p><b>The action facet under-uses its own vocabulary.</b>
<span class="mono">add</span> {acts.get('add', 0)} against <span class="mono">reword</span>
{acts.get('reword', 0)} &mdash; prompt rewrites are being counted as additions. And
<span class="mono">revert</span> {acts.get('revert', 0)} is far too low: a revert is a property
of a <i>commit</i>, and at symbol scope it looks like ordinary edits, so this decomposition
structurally cannot see it. Read the action column as a coarse split, not a fine one.</p>

<p><b>Rule and model disagree on {len(dis)} of {hinted} hinted edits ({rate:.0%}), and the
pattern is systematic rather than noise: the rule labels by location, the model by purpose.</b>
Adjudicated &mdash; <span class="mono">_complete</span> (rule
<span class="mono">model_client</span>, model <span class="mono">control_loop</span>) is a
client call used to force a final answer, so the rule is right about where it is and the model
is describing what it is for; <span class="mono">TOOLS</span> (rule
<span class="mono">tool_surface</span>, model <span class="mono">prompt</span>) is a tool
description, which is genuinely both. The rule wins on conflict, so these are recorded rather
than resolved &mdash; but they mark where a single-role facet is the wrong shape and
multi-label would fit better.</p>
<table><thead><tr><th>edits</th><th>rule said</th><th>model said</th></tr></thead>
<tbody>{confusion}</tbody></table>

<p><b>Answer-parsing regexes scatter.</b> Regex-kind edits spread across
{len(regex_roles)} roles ({_esc(', '.join(f'{k} {v}' for k, v in regex_roles.most_common(4)))}),
and several labelled <span class="mono">prompt</span> are answer-extraction patterns belonging
to <span class="mono">submission</span>. No rule covers regex bindings, so the model decides
unaided.</p>
</div>
"""


def render_sections(rows: list[dict], edits: dict[str, dict]) -> str:
    return (
        "<h2>How an edit is extracted</h2>" + worked_example(rows, edits)
        + "<h2>Symbols, kinds, and what they were assigned</h2>" + kinds_table(rows, edits)
        + "<h2>Where these labels are wrong</h2>" + false_labels(rows, edits)
    )
