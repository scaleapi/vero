#!/usr/bin/env python3
"""Render a run's search progress and its best packing as one SVG.

    python make_figure.py [session-dir] [-o results/progress.svg]

Reads the evaluation records a run leaves behind — `sum_radii` per development
evaluation, and the circle layout the trusted harness recorded alongside each —
and draws two panels: how the score moved as the agent worked, and what the best
layout looks like.

Standard library only, matching the harness, so the figure regenerates from a
session directory with no extra install.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

W, H = 860, 360
PAD = 52
PANEL = 336
PUBLISHED_BEST = 2.635  # best known packing for 26 circles in a unit square


def load(session: Path) -> tuple[list[dict], list[dict]]:
    """Return (development evaluations, final evaluations), oldest first."""
    development: list[dict] = []
    final: list[dict] = []
    for record in sorted(session.glob("evaluations/*/evaluation.json")):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        report = data.get("report") or {}
        metrics = report.get("metrics") or {}
        score = metrics.get("sum_radii")
        if score is None:
            continue
        # Artifacts are namespaced by backend id (artifacts/<backend>/layout.json),
        # so search rather than assuming the flat path.
        layouts = sorted((record.parent / "artifacts").rglob("layout.json"))
        entry = {
            "score": float(score),
            "valid": float(metrics.get("valid") or 0.0),
            "created": ((data.get("request") or {}).get("candidate") or {}).get(
                "created_at"
            )
            or "",
            "layout": layouts[0] if layouts else None,
        }
        name = ((data.get("request") or {}).get("evaluation_set") or {}).get("name")
        (final if name == "final" else development).append(entry)
    development.sort(key=lambda e: e["created"])
    final.sort(key=lambda e: e["created"])
    return development, final


def circles(layout: Path | None) -> list[dict]:
    if layout is None:
        return []
    try:
        return json.loads(layout.read_text(encoding="utf-8")).get("circles") or []
    except (OSError, ValueError):
        return []


def progress_panel(points: list[dict], baseline: float | None) -> list[str]:
    x0, y0 = PAD, PAD
    x1, y1 = PAD + PANEL, PAD + PANEL - 24
    out = [
        f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
        'fill="#fbfbfd" stroke="#d6d6de"/>',
        f'<text x="{x0}" y="{y0 - 14}" class="h">Search progress</text>',
    ]
    if not points:
        return out + [
            f'<text x="{x0 + 12}" y="{(y0 + y1) // 2}" class="l">no evaluations yet</text>'
        ]

    top = max(PUBLISHED_BEST, max(p["score"] for p in points)) * 1.04
    bottom = min(baseline or points[0]["score"], min(p["score"] for p in points)) * 0.96

    def sx(i: int) -> float:
        span = max(len(points) - 1, 1)
        return x0 + 10 + (x1 - x0 - 20) * i / span

    def sy(v: float) -> float:
        return y1 - (y1 - y0) * (v - bottom) / max(top - bottom, 1e-9)

    for value, label, cls in (
        (PUBLISHED_BEST, f"published best {PUBLISHED_BEST}", "ref"),
        (baseline, f"baseline {baseline:.4f}" if baseline else "", "base"),
    ):
        if value is None or not (bottom <= value <= top):
            continue
        y = sy(value)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="{cls}"/>')
        out.append(f'<text x="{x0 + 6}" y="{y - 5:.1f}" class="l">{label}</text>')

    best = -1e9
    ridge: list[str] = []
    for i, p in enumerate(points):
        best = max(best, p["score"])
        ridge.append(f"{sx(i):.1f},{sy(best):.1f}")
    out.append(f'<polyline points="{" ".join(ridge)}" class="ridge"/>')
    for i, p in enumerate(points):
        cls = "pt" if p["valid"] >= 1.0 else "bad"
        out.append(f'<circle cx="{sx(i):.1f}" cy="{sy(p["score"]):.1f}" r="3" class="{cls}"/>')

    out.append(
        f'<text x="{(x0 + x1) // 2}" y="{y1 + 26}" class="l mid">'
        f"development evaluation (1–{len(points)})</text>"
    )
    out.append(
        f'<text x="{x0 - 8}" y="{y0 + 4}" class="l end">{top:.2f}</text>'
        f'<text x="{x0 - 8}" y="{y1}" class="l end">{bottom:.2f}</text>'
    )
    return out


def packing_panel(items: list[dict], score: float | None, title: str) -> list[str]:
    x0, y0 = PAD + PANEL + 92, PAD
    side = PANEL - 24
    out = [
        f'<text x="{x0}" y="{y0 - 14}" class="h">{title}</text>',
        f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" '
        'fill="#fbfbfd" stroke="#d6d6de"/>',
    ]
    for c in items:
        try:
            cx = x0 + float(c["x"]) * side
            cy = y0 + (1.0 - float(c["y"])) * side
            r = float(c["radius"]) * side
        except (KeyError, TypeError, ValueError):
            continue
        out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" class="disc"/>')
    if score is not None:
        out.append(
            f'<text x="{x0 + side // 2}" y="{y0 + side + 26}" class="l mid">'
            f"sum of radii {score:.4f}</text>"
        )
    return out


def main() -> int:
    args = [a for a in sys.argv[1:]]
    out_path = Path("results/progress.svg")
    if "-o" in args:
        i = args.index("-o")
        out_path = Path(args[i + 1])
        del args[i : i + 2]
    session = Path(args[0]) if args else Path(".vero/session")
    if not session.exists():
        print(f"no session at {session}")
        return 1

    development, final = load(session)
    baseline = final[0]["score"] if final else None
    best = max(development, key=lambda e: e["score"]) if development else None
    shipped = max(final, key=lambda e: e["score"]) if final else None
    show = shipped if (shipped and best and shipped["score"] >= best["score"]) else best

    body = progress_panel(development, baseline)
    body += packing_panel(
        circles(show["layout"]) if show else [],
        show["score"] if show else None,
        "Best packing found",
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<style>
 text {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
 .h {{ font-size: 14px; font-weight: 600; fill: #1a1a24; }}
 .l {{ font-size: 11px; fill: #6a6a78; }}
 .mid {{ text-anchor: middle; }} .end {{ text-anchor: end; }}
 .ridge {{ fill: none; stroke: #2f6feb; stroke-width: 2; }}
 .pt {{ fill: #2f6feb; }} .bad {{ fill: #d64545; }}
 .ref {{ stroke: #8a8a99; stroke-width: 1; stroke-dasharray: 5 4; }}
 .base {{ stroke: #d64545; stroke-width: 1; stroke-dasharray: 3 3; }}
 .disc {{ fill: #2f6feb; fill-opacity: 0.18; stroke: #2f6feb; stroke-width: 1; }}
</style>
<rect width="{W}" height="{H}" fill="#ffffff"/>
{chr(10).join(body)}
</svg>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  development evaluations: {len(development)}")
    if baseline is not None:
        print(f"  baseline (final partition): {baseline:.4f}")
    if best:
        print(f"  best development score:    {best['score']:.4f}")
    if shipped:
        print(f"  selected, on final:        {shipped['score']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
