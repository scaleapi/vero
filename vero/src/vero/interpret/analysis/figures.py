"""Render the analysis as one self-contained HTML page.

Inline SVG, no build step and no CDN: the output is a single file that opens from
disk and can be handed to someone. Colour follows the validated reference palette —
sequential blue for magnitude, the fixed categorical order for identity, blue/red
diverging for polarity — and every chart carries a table view, because several
palette steps sit below 3:1 on the light surface and the relief rule applies.
"""

from __future__ import annotations

import html
import json
from collections import Counter

from vero.interpret.analysis import stats

CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
POS, NEG = "#2a78d6", "#e34948"

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f4f3f0;--text-primary:#0b0b0b;
--text-secondary:#52514e;--text-muted:#7a7975;--grid:#e6e5e1;--pos:#2a78d6;--neg:#e34948;
background:var(--surface-1);color:var(--text-primary);
font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;padding:28px 32px;max-width:1180px;margin:0 auto}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--surface-2:#242422;--text-primary:#fff;--text-secondary:#c3c2b7;
--text-muted:#8f8e86;--grid:#343430;--pos:#3987e5;--neg:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;
--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;--grid:#343430;--pos:#3987e5;--neg:#e66767}
h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:34px 0 2px}
p.note{color:var(--text-secondary);margin:2px 0 14px;max-width:74ch;font-size:13px}
.fig{background:var(--surface-1);border:1px solid var(--grid);border-radius:10px;padding:14px 16px;margin-bottom:6px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 0;font-size:12px;color:var(--text-secondary)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
text{font:11px ui-sans-serif,-apple-system,sans-serif}
.axis{fill:var(--text-secondary)} .muted{fill:var(--text-muted)} .ink{fill:var(--text-primary)}
.grid{stroke:var(--grid);stroke-width:1}
details{margin:6px 0 0} summary{cursor:pointer;color:var(--text-secondary);font-size:12px}
table{border-collapse:collapse;font-size:12px;margin-top:8px} th,td{border:1px solid var(--grid);padding:3px 8px;text-align:right}
th:first-child,td:first-child{text-align:left} th{color:var(--text-secondary);font-weight:600}
#tip{position:fixed;pointer-events:none;opacity:0;background:var(--surface-2);color:var(--text-primary);
border:1px solid var(--grid);border-radius:6px;padding:6px 9px;font-size:12px;max-width:320px;
box-shadow:0 2px 10px rgba(0,0,0,.16);z-index:50;transition:opacity .08s}
.hit{cursor:crosshair}
.toggle{float:right;font-size:12px;color:var(--text-secondary);cursor:pointer;border:1px solid var(--grid);
border-radius:6px;padding:3px 9px;background:var(--surface-2)}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mousemove',e=>{tip.innerHTML=el.dataset.tip;tip.style.opacity=1;
    const p=12;let x=e.clientX+p,y=e.clientY+p;
    if(x+tip.offsetWidth>innerWidth)x=e.clientX-tip.offsetWidth-p;
    if(y+tip.offsetHeight>innerHeight)y=e.clientY-tip.offsetHeight-p;
    tip.style.left=x+'px';tip.style.top=y+'px';});
  el.addEventListener('mouseleave',()=>tip.style.opacity=0);});
document.getElementById('themebtn').onclick=()=>{
  const r=document.documentElement;
  const dark=r.getAttribute('data-theme')==='dark';
  r.setAttribute('data-theme',dark?'light':'dark');};
"""


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _seq(frac: float) -> str:
    return SEQ[max(0, min(len(SEQ) - 1, round(frac * (len(SEQ) - 1))))]


def _table(headers: list[str], rows: list[list], caption: str) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return (
        f"<details><summary>{_esc(caption)}</summary>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></details>"
    )


def fig_prevalence(rows: list[dict]) -> str:
    roles, table = stats.prevalence(rows)
    benches = [b for b in stats.BENCH_ORDER if b in next(iter(table.values()))]
    cw, ch, lw, top = 132, 26, 150, 46
    w, h = lw + cw * len(benches) + 30, top + ch * len(roles) + 26
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for j, b in enumerate(benches):
        x = lw + j * cw + cw / 2
        mark = " ‡" if b in stats.CONSTRUCTED_SEED else ""
        out.append(f'<text class="axis" x="{x}" y="{top-24}" text-anchor="middle">{_esc(b[:15])}{mark}</text>')
    for i, role in enumerate(roles):
        y = top + i * ch
        out.append(f'<text class="axis" x="{lw-10}" y="{y+17}" text-anchor="end">{_esc(role)}</text>')
        for j, b in enumerate(benches):
            hit, tot = table[role][b]
            frac = hit / tot if tot else 0
            x = lw + j * cw
            tip = f"<b>{_esc(role)}</b><br>{_esc(b)}<br>{hit} of {tot} cells ({frac:.0%})"
            # 2px surface gap between adjacent fills.
            out.append(
                f'<rect class="hit" data-tip="{tip}" x="{x+1}" y="{y+1}" width="{cw-2}" '
                f'height="{ch-2}" rx="4" fill="{_seq(frac)}"/>'
            )
            ink = "#0b0b0b" if frac < 0.62 else "#ffffff"
            out.append(
                f'<text x="{x+cw/2}" y="{y+17}" text-anchor="middle" fill="{ink}">{hit}/{tot}</text>'
            )
    out.append("</svg>")
    tbl = _table(
        ["role"] + benches,
        [[r] + [f"{table[r][b][0]}/{table[r][b][1]}" for b in benches] for r in roles],
        "Table view",
    )
    return "".join(out) + tbl


def fig_rarefaction(rows: list[dict]) -> str:
    curves = stats.rarefaction(rows)
    benches = [b for b in stats.BENCH_ORDER if b in curves]
    w, h, pad = 760, 300, 46
    maxy = max(max(v) for v in curves.values())
    maxx = max(len(v) for v in curves.values())
    sx = lambda i: pad + i * (w - pad - 150) / max(maxx - 1, 1)
    sy = lambda v: h - pad - v * (h - 2 * pad) / maxy
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for g in range(0, int(maxy) + 1, 2):
        out.append(f'<line class="grid" x1="{pad}" y1="{sy(g)}" x2="{w-150}" y2="{sy(g)}"/>')
        out.append(f'<text class="muted" x="{pad-8}" y="{sy(g)+4}" text-anchor="end">{g}</text>')
    for k, b in enumerate(benches):
        pts = curves[b]
        d = " ".join(f"{'M' if i==0 else 'L'}{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{CAT_LIGHT[k]}" stroke-width="2"/>')
        for i, v in enumerate(pts):
            tip = f"<b>{_esc(b)}</b><br>after {i+1} cells: {v:.1f} distinct roles"
            out.append(f'<circle class="hit" data-tip="{tip}" cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="5" fill="transparent"/>')
        out.append(
            f'<text x="{w-144}" y="{sy(pts[-1])+4}" fill="{CAT_LIGHT[k]}">{_esc(b)}</text>'
        )
    out.append(f'<text class="muted" x="{(w-150+pad)/2}" y="{h-12}" text-anchor="middle">cells sampled</text>')
    out.append("</svg>")
    tbl = _table(
        ["cells"] + benches,
        [[i + 1] + [f"{curves[b][i]:.1f}" if i < len(curves[b]) else "" for b in benches]
         for i in range(maxx)],
        "Table view",
    )
    return "".join(out) + tbl


def fig_jaccard(rows: list[dict]) -> str:
    data = stats.jaccard(rows)
    benches = [b for b in stats.BENCH_ORDER if b in data]
    rh, lw, w = 42, 150, 760
    h = 40 + rh * len(benches)
    x0, x1 = lw, w - 130
    lo = min(min(d["null_lo"], d["observed"]) for d in data.values()) - 0.04
    hi = max(max(d["null_hi"], d["observed"]) for d in data.values()) + 0.04
    sx = lambda v: x0 + (v - lo) * (x1 - x0) / (hi - lo)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for t in [round(lo + i * (hi - lo) / 4, 2) for i in range(5)]:
        out.append(f'<line class="grid" x1="{sx(t)}" y1="26" x2="{sx(t)}" y2="{h-16}"/>')
        out.append(f'<text class="muted" x="{sx(t)}" y="20" text-anchor="middle">{t:.2f}</text>')
    for i, b in enumerate(benches):
        d = data[b]
        y = 40 + i * rh
        out.append(f'<text class="axis" x="{lw-10}" y="{y+16}" text-anchor="end">{_esc(b)}</text>')
        tip_null = f"<b>{_esc(b)}</b><br>null 95%: {d['null_lo']:.3f}–{d['null_hi']:.3f}<br>null mean {d['null_mean']:.3f}"
        out.append(
            f'<rect class="hit" data-tip="{tip_null}" x="{sx(d["null_lo"]):.1f}" y="{y+5}" '
            f'width="{max(sx(d["null_hi"])-sx(d["null_lo"]),2):.1f}" height="22" rx="4" '
            f'fill="var(--grid)"/>'
        )
        col = POS if d["verdict"] == "converged" else (NEG if d["verdict"] == "diverged" else "#7a7975")
        tip = (f"<b>{_esc(b)}</b><br>observed {d['observed']:.3f} ({d['n_cells']} cells)"
               f"<br>null {d['null_lo']:.3f}–{d['null_hi']:.3f}<br><b>{_esc(d['verdict'])}</b>")
        out.append(
            f'<circle class="hit" data-tip="{tip}" cx="{sx(d["observed"]):.1f}" cy="{y+16}" '
            f'r="6" fill="{col}" stroke="var(--surface-1)" stroke-width="2"/>'
        )
        out.append(f'<text class="muted" x="{x1+10}" y="{y+20}">{d["verdict"]}</text>')
    out.append("</svg>")
    tbl = _table(
        ["benchmark", "cells", "observed", "null 2.5%", "null 97.5%", "verdict"],
        [[b, data[b]["n_cells"], f"{data[b]['observed']:.3f}", f"{data[b]['null_lo']:.3f}",
          f"{data[b]['null_hi']:.3f}", data[b]["verdict"]] for b in benches],
        "Table view",
    )
    return "".join(out) + tbl


def fig_action_role(rows: list[dict]) -> str:
    roles, actions, counts = stats.action_by_role(rows)
    lw, w, rh = 150, 760, 30
    h = 40 + rh * len(roles)
    maxv = max(sum(counts.get((r, a), 0) for a in actions) for r in roles)
    bw = w - lw - 60
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i, role in enumerate(roles):
        y = 34 + i * rh
        out.append(f'<text class="axis" x="{lw-10}" y="{y+16}" text-anchor="end">{_esc(role)}</text>')
        x = lw
        total = sum(counts.get((role, a), 0) for a in actions)
        for k, a in enumerate(actions):
            n = counts.get((role, a), 0)
            if not n:
                continue
            seg = n * bw / maxv
            tip = f"<b>{_esc(role)}</b><br>{_esc(a)}: {n} edits ({n/total:.0%} of this role)"
            out.append(
                f'<rect class="hit" data-tip="{tip}" x="{x:.1f}" y="{y+3}" '
                f'width="{max(seg-2,1):.1f}" height="20" rx="4" fill="{CAT_LIGHT[k % 8]}"/>'
            )
            x += seg
        out.append(f'<text class="muted" x="{x+8:.1f}" y="{y+18}">{total}</text>')
    out.append("</svg>")
    leg = '<div class="legend">' + "".join(
        f'<span><i style="background:{CAT_LIGHT[k%8]}"></i>{_esc(a)}</span>'
        for k, a in enumerate(actions)
    ) + "</div>"
    tbl = _table(["role"] + actions,
                 [[r] + [counts.get((r, a), 0) for a in actions] for r in roles],
                 "Table view")
    return "".join(out) + leg + tbl


def fig_direction(rows: list[dict], edits: dict[str, dict]) -> str:
    data = stats.tuning_direction(rows, edits)
    if not data:
        return "<p class='note'>No scalar constants changed value.</p>"
    lw, w, rh = 210, 760, 28
    h = 46 + rh * len(data)
    mx = max(max(u, d) for _, u, d in data) or 1
    mid = lw + (w - lw - 40) / 2
    half = (w - lw - 60) / 2
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    out.append(f'<line class="grid" x1="{mid}" y1="24" x2="{mid}" y2="{h-14}"/>')
    out.append(f'<text class="muted" x="{mid-half/2}" y="18" text-anchor="middle">lowered</text>')
    out.append(f'<text class="muted" x="{mid+half/2}" y="18" text-anchor="middle">raised</text>')
    for i, (sym, up, dn) in enumerate(data):
        y = 32 + i * rh
        out.append(f'<text class="axis" x="{lw-10}" y="{y+16}" text-anchor="end">{_esc(sym[:26])}</text>')
        if dn:
            wpx = dn * half / mx
            out.append(f'<rect class="hit" data-tip="<b>{_esc(sym)}</b><br>lowered in {dn} edits" '
                       f'x="{mid-wpx:.1f}" y="{y+4}" width="{max(wpx-2,1):.1f}" height="18" rx="4" fill="{NEG}"/>')
            out.append(f'<text class="muted" x="{mid-wpx-8:.1f}" y="{y+17}" text-anchor="end">{dn}</text>')
        if up:
            wpx = up * half / mx
            out.append(f'<rect class="hit" data-tip="<b>{_esc(sym)}</b><br>raised in {up} edits" '
                       f'x="{mid+2}" y="{y+4}" width="{max(wpx-2,1):.1f}" height="18" rx="4" fill="{POS}"/>')
            out.append(f'<text class="muted" x="{mid+wpx+8:.1f}" y="{y+17}">{up}</text>')
    out.append("</svg>")
    leg = ('<div class="legend"><span><i style="background:'+NEG+'"></i>lowered</span>'
           '<span><i style="background:'+POS+'"></i>raised</span></div>')
    tbl = _table(["constant", "raised", "lowered"], [[s, u, d] for s, u, d in data], "Table view")
    return "".join(out) + leg + tbl


def fig_provenance(rows: list[dict]) -> str:
    """Whose defect each fix repaired. Derived from the seed tree, not model-assigned."""
    data = stats.provenance_of_fixes(rows)
    benches = [b for b in stats.BENCH_ORDER if b in data]
    if not benches:
        return "<p class='note'>No fixes labelled.</p>"
    order = ["seed", "own", "unknown"]
    cols = {"seed": CAT_LIGHT[0], "own": CAT_LIGHT[1], "unknown": "#7a7975"}
    lw, w, rh = 150, 760, 32
    h = 34 + rh * len(benches)
    maxv = max(sum(data[b].values()) for b in benches)
    bw = w - lw - 70
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i, b in enumerate(benches):
        y = 22 + i * rh
        total = sum(data[b].values())
        out.append(f'<text class="axis" x="{lw-10}" y="{y+17}" text-anchor="end">{_esc(b)}</text>')
        x = lw
        for key in order:
            n = data[b].get(key, 0)
            if not n:
                continue
            seg = n * bw / maxv
            tip = f"<b>{_esc(b)}</b><br>{_esc(key)}: {n} of {total} fixes ({n/total:.0%})"
            out.append(f'<rect class="hit" data-tip="{tip}" x="{x:.1f}" y="{y+4}" '
                       f'width="{max(seg-2,1):.1f}" height="20" rx="4" fill="{cols[key]}"/>')
            x += seg
        out.append(f'<text class="muted" x="{x+8:.1f}" y="{y+19}">{total}</text>')
    out.append("</svg>")
    leg = '<div class="legend">' + "".join(
        f'<span><i style="background:{cols[k]}"></i>{k} defect</span>' for k in order) + "</div>"
    tbl = _table(["benchmark"] + order,
                 [[b] + [data[b].get(k, 0) for k in order] for b in benches], "Table view")
    return "".join(out) + leg + tbl


def render(rows: list[dict], edits: dict[str, dict], meta: dict) -> str:
    ag = stats.hint_agreement(rows)
    figs = [
        ("Which kinds of edit did optimizers make, and how universally?",
         "Share of cells in each benchmark that ever made an edit of this kind. Counted per "
         "cell, not per edit: cells produced between 1 and 18 candidates, so edit-weighted "
         "counts would measure which cells were prolific. ‡ gaia-shell's seed is an empty "
         "shell, so every role is present there by construction — do not pool that column.",
         fig_prevalence(rows)),
        ("Does the next optimizer try anything new?",
         "Distinct roles discovered as cells are added, averaged over 200 random orderings. "
         "A curve that flattens says the repertoire was exhausted early; one still climbing "
         "at the right edge says it was not.",
         fig_rarefaction(rows)),
        ("Do different optimizers explore different things?",
         "Mean pairwise Jaccard distance between cells' role repertoires (dot) against a "
         "permutation null holding each cell's repertoire size and the corpus role "
         "frequencies fixed (grey band, 95%). Left of the band means cells are more alike "
         "than chance — convergence. The raw distance alone says nothing without this.",
         fig_jaccard(rows)),
        ("What was done to each part of the harness?",
         "Edits per role, split by action. This is the one view where edit counts are the "
         "right unit, since the question is about the composition of the work.",
         fig_action_role(rows)),
        ("Whose defect did each fix repair?",
         "Fixes split by whether the repaired code was still exactly as the seed wrote it "
         "(seed defect) or had already been rewritten by an earlier candidate in the same "
         "cell (the optimizer's own). Derived by comparing trees, not asked of the model: "
         "asking returned 452 own against 3 seed and called 21 of 22 swe-atlas submission "
         "fixes self-inflicted, when 15 of them repair a defect in the seed's answer parser.",
         fig_provenance(rows)),
        ("Which way did the knobs go?",
         "Scalar constants whose value actually changed, by direction. Constants touched by "
         "reformatting without a value change are excluded.",
         fig_direction(rows, edits)),
    ]
    body = "".join(
        f'<h2>{_esc(t)}</h2><p class="note">{_esc(n)}</p><div class="fig">{svg}</div>'
        for t, n, svg in figs
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>What optimizers modified</title><style>{CSS}</style></head>
<body><div class="viz-root">
<button class="toggle" id="themebtn">light / dark</button>
<h1>What optimizers modified</h1>
<p class="note">{_esc(meta['cells'])} cells, {_esc(meta['edits'])} symbol-scoped edits,
{_esc(meta['labels'])} labelled. Roles were assigned by deterministic rule where the
file, symbol kind or name settles it ({ag['hinted']} edits) and by model otherwise
({ag['model_decided']}); {ag['disagreements']} audited edits disagreed and both readings
are kept in the record. Reward is deliberately absent: measurement noise in this corpus
makes category-versus-score comparisons unsupportable.</p>
{body}
<p class="note">Generated by <code>vero interpret report</code>. Colour: sequential blue for
magnitude, fixed categorical order for identity, blue/red diverging for polarity; palette
validated for colour-vision deficiency. Every figure has a table view.</p>
</div><div id="tip"></div><script>{JS}</script></body></html>"""
