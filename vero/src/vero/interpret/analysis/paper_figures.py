"""Print figures in the Scale 2.0 brand style: vector PDF for LaTeX, PNG to review.

Separate from `figures.py` on purpose. That module builds one interactive HTML page
for colleagues to explore — hover, dark mode, table views. This one produces
caption-driven figures for a paper: no baked-in titles, sized at the real placement
width so point sizes render 1:1, muted palette on white.

Archetypes are taken from the brand skill rather than invented. Role prevalence is a
bounded-metric grid, so it is a sequential heatmap. Diversity-versus-null and
knob direction each have two values per item where the gap is the story, so both are
dumbbells. Rarefaction is a plain line plot — no archetype fits a saturation curve,
and forcing one would obscure the shape that matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

BRAND = Path(__file__).parent / "brand"
sys.path.insert(0, str(BRAND))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scale_brand_style import (  # noqa: E402
    FS,
    INK,
    INK_FAINT,
    PALETTE,
    TEXT_WIDTH_IN,
    apply_scale_style,
    family_colors,
    save,
)

from vero.interpret.analysis import display, stats  # noqa: E402

# Roles grouped so the heatmap's left colour bar means something. Rows must be
# contiguous by group, so this ordering is load-bearing, not cosmetic.
ROLE_GROUPS: list[tuple[str, list[str]]] = [
    ("Agent behaviour", ["prompt", "control_loop", "tool_surface", "tool_impl",
                         "retrieval", "submission"]),
    ("Budgets", ["budget_turns", "budget_output", "budget_wallclock", "context_mgmt"]),
    ("Plumbing", ["model_client", "initialization", "env_setup"]),
    ("Not the agent", ["tests", "metadata", "other"]),
]


def fig_prevalence(rows: list[dict], stem: Path) -> None:
    """Heatmap: share of cells per benchmark that ever made each kind of edit."""
    roles, table = stats.prevalence(rows)
    benches = [b for b in stats.BENCH_ORDER if b in next(iter(table.values()))]

    ordered: list[tuple[str, str, list[float]]] = []
    for group, members in ROLE_GROUPS:
        for role in members:
            if role in table:
                ordered.append(
                    (group, role, [table[role][b][0] / table[role][b][1] for b in benches])
                )
    counts = {
        (r, b): table[r][b] for _, r, _ in ordered for b in benches
    }

    M = np.array([r[2] for r in ordered])
    labels = [r[1] for r in ordered]
    color = family_colors([r[0] for r in ordered])
    groups: list[list] = []
    for i, (cat, _, _) in enumerate(ordered):
        if groups and groups[-1][0] == cat:
            groups[-1][2] = i
        else:
            groups.append([cat, i, i])

    cmap = LinearSegmentedColormap.from_list(
        "scale_blue", ["#FFFFFF", PALETTE["slate"], PALETTE["atlas"]]
    )
    nrow, ncol = M.shape
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 0.30 * nrow + 0.55))
    fig.subplots_adjust(left=0.32, right=0.985, top=0.93, bottom=0.015)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(nrow):
        for j in range(ncol):
            hit, tot = counts[(labels[i], benches[j])]
            ax.text(j, i, f"{hit}/{tot}", ha="center", va="center",
                    fontsize=FS["value"], color="white" if M[i, j] > 0.62 else INK)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(
        [display.benchmark(b, short=True)
         + ("\u2021" if b in stats.CONSTRUCTED_SEED else "")
         for b in benches],
        fontsize=FS["label"], color=INK,
    )
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([display.role(x) for x in labels], fontsize=FS["label"], color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)

    trans = ax.get_yaxis_transform()
    for cat, r0, r1 in groups:
        ax.add_patch(Rectangle((-0.32, r0 - 0.5), 0.024, (r1 - r0 + 1), transform=trans,
                              facecolor=color[cat], edgecolor="white", lw=0.6,
                              clip_on=False, zorder=5))
        ax.text(-0.345, (r0 + r1) / 2, cat, transform=trans, rotation=90,
                ha="center", va="center", fontsize=6.4, color=INK)
    save(fig, str(stem))
    plt.close(fig)


def fig_diversity(rows: list[dict], stem: Path) -> None:
    """Dumbbell: observed repertoire distance against the permutation null."""
    data = stats.jaccard(rows)
    benches = [b for b in stats.BENCH_ORDER if b in data][::-1]
    y = np.arange(len(benches))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 0.48 * len(benches) + 0.95))
    fig.subplots_adjust(left=0.20, right=0.965, top=0.93, bottom=0.175)
    for i, b in enumerate(benches):
        d = data[b]
        # Null interval as a light band, so the dumbbell reads against it.
        ax.plot([d["null_lo"], d["null_hi"]], [i, i], lw=6.0,
                color=PALETTE["gray_whisper"], solid_capstyle="round", zorder=1)
        ax.plot([d["observed"], d["null_mean"]], [i, i], lw=1.4,
                color=PALETTE["atlas"], zorder=2)
        ax.plot(d["null_mean"], i, "o", ms=6, mfc="white",
                mec=PALETTE["gray_soft"], mew=1.4, zorder=3)
        ax.plot(d["observed"], i, "o", ms=6.5, color=PALETTE["evergreen"], zorder=4)
        ax.text(d["observed"], i + 0.22, f"{d['observed']:.3f}", ha="center",
                va="bottom", fontsize=FS["value"], color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([display.benchmark(b, short=True) for b in benches], fontsize=FS["label"], color=INK)
    ax.set_xlabel("mean pairwise Jaccard distance between cells' edit repertoires",
                  fontsize=FS["axis"], color=INK)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=PALETTE["gray_whisper"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["gray_medium"])
    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=6, color=PALETTE["evergreen"],
                   label="observed"),
        plt.Line2D([], [], marker="o", ls="none", ms=6, mfc="white",
                   mec=PALETTE["gray_soft"], mew=1.4, label="null mean"),
        plt.Line2D([], [], lw=6, color=PALETTE["gray_whisper"], label="null 95%"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=FS["legend"], handletextpad=0.5, borderaxespad=0.2)
    save(fig, str(stem))
    plt.close(fig)


def fig_rarefaction(rows: list[dict], stem: Path) -> None:
    """Line: distinct edit kinds discovered as cells are added."""
    curves = stats.rarefaction(rows)
    benches = [b for b in stats.BENCH_ORDER if b in curves]
    colors = family_colors(benches)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.78, 2.9))
    fig.subplots_adjust(left=0.105, right=0.755, top=0.96, bottom=0.165)
    # Direct labels are the house preference but cannot work here: three curves land
    # on exactly 16 kinds and two on 15, so endpoint labels overlap into mush. A
    # legend ordered by final value keeps identity next to the visual order instead.
    for b in sorted(benches, key=lambda k: -curves[k][-1]):
        pts = curves[b]
        ax.plot(np.arange(1, len(pts) + 1), pts, lw=1.6, color=colors[b],
                label=display.benchmark(b, short=True))
    ax.set_xlabel("cells sampled", fontsize=FS["axis"], color=INK)
    ax.set_ylabel("distinct edit kinds seen", fontsize=FS["axis"], color=INK)
    ax.grid(color=PALETTE["gray_whisper"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(PALETTE["gray_medium"])
    ax.tick_params(colors=INK_FAINT, labelsize=FS["tick"], length=2)
    ax.set_xticks([1, 5, 10, 15, 20])   # cells are discrete; 2.5 is not a cell count
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=FS["legend"], handlelength=1.4, handletextpad=0.6,
              labelspacing=0.55)
    save(fig, str(stem))
    plt.close(fig)


def fig_knob_direction(rows: list[dict], edits: dict[str, dict], stem: Path) -> None:
    """Dumbbell: two directional counts per constant — raised against lowered."""
    data = stats.tuning_direction(rows, edits, top=10)
    if not data:
        return
    data = data[::-1]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 0.36 * len(data) + 0.85))
    fig.subplots_adjust(left=0.34, right=0.965, top=0.965, bottom=0.155)
    for i, (sym, up, dn) in enumerate(data):
        ax.plot([dn, up], [i, i], lw=1.3, color=PALETTE["gray_medium"], zorder=1)
        ax.plot(dn, i, "o", ms=6, mfc="white", mec=PALETTE["tan"], mew=1.5, zorder=3)
        ax.plot(up, i, "o", ms=6.5, color=PALETTE["evergreen"], zorder=4)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([s for s, _, _ in data], fontsize=FS["label"], color=INK)
    ax.set_xlabel("edits changing this constant", fontsize=FS["axis"], color=INK)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=PALETTE["gray_whisper"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["gray_medium"])
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="none", ms=6.5,
                   color=PALETTE["evergreen"], label="raised"),
        plt.Line2D([], [], marker="o", ls="none", ms=6, mfc="white",
                   mec=PALETTE["tan"], mew=1.5, label="lowered"),
    ], loc="lower right", frameon=False, fontsize=FS["legend"],
        handletextpad=0.5, borderaxespad=0.1)
    save(fig, str(stem))
    plt.close(fig)


# One directory per figure, named for its id and holding the spec beside its outputs.
# A flat directory is tolerable for four figures and unusable for forty: outputs,
# specs and stale renders interleave alphabetically and nothing travels as a unit.
FIGURES: list[tuple[str, str]] = [
    ("figure_01_prevalence", "fig_prevalence"),
    ("figure_02_diversity", "fig_diversity"),
    ("figure_03_rarefaction", "fig_rarefaction"),
    ("figure_04_knob_direction", "fig_knob_direction"),
]


def render_all(rows: list[dict], edits: dict[str, dict], out: Path) -> list[str]:
    apply_scale_style()
    written: list[str] = []
    for fid, func in FIGURES:
        folder = out / fid
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / fid
        renderer = globals()[func]
        if func == "fig_knob_direction":
            renderer(rows, edits, stem)
        else:
            renderer(rows, stem)
        written.append(f"{fid}/{fid}.{{pdf,png}}")
    return written
