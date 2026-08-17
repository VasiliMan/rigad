"""Render the figures used in the README and the project report.

Reads the metrics written by evaluate_tracks.py and evaluate.py, so the numbers
in the figures and the numbers in the text can never drift apart.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rigad.config import FIGURES_DIR

# Validated categorical slots 1-3 (all-pairs, light surface). Every bar carries
# a direct value label, which is also the required relief for the aqua slot's
# sub-3:1 contrast against the surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_MUTED = "#0b0b0b", "#52514e"
GRID = "#e4e3df"


def style(ax) -> None:
    """Recessive axes and grid: the data should be the only assertive thing."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.title.set_color(INK)


def figure_consistency(metrics: dict) -> None:
    """Two questions: is it better than chance, and is the margin informative?"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))

    left, right = axes
    for ax in axes:
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 0.52)

    # Panel 1 — against chance.
    bars = left.bar(
        ["RIGAD", "random"],
        [metrics["topic_consistency"], metrics["topic_consistency_random"]],
        color=[BLUE, GRID], width=0.5,
    )
    left.set_title(
        f"Papers on the same OpenAlex topic\nland on the same track  ({metrics['lift']:.2f}x chance)",
        fontsize=10.5, loc="left", pad=10,
    )
    left.set_ylabel("topic consistency", fontsize=9, color=INK_MUTED)

    # Panel 2 — does the reported margin mean anything?
    bars2 = right.bar(
        ["wide margin", "narrow margin"],
        [metrics["consistency_confident_half"], metrics["consistency_uncertain_half"]],
        color=[BLUE, AQUA], width=0.5,
    )
    right.set_title(
        "Drafts the tool is confident about\nare matched more consistently",
        fontsize=10.5, loc="left", pad=10,
    )

    for ax, group in ((left, bars), (right, bars2)):
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=INK)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "track_consistency.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_allocation_tradeoff(metrics: dict) -> None:
    """What the institutional tiebreaker costs, and what it buys.

    Two measures on different scales, so a scatter rather than two y-axes.
    Right is a better shared theme; low is less chance of being grouped with
    someone from your own institution. The bottom-right corner is the good
    corner.
    """
    size = metrics["headline_cohort_size"]
    points = {
        "RIGAD": ("rigad", BLUE),
        "RIGAD, tiebreaker off": ("rigad-ignoring-institution", BLUE),
        "naive similarity": ("similarity", ORANGE),
        "random": ("random", AQUA),
    }

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    style(ax)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    offsets = {"RIGAD": (10, 6), "RIGAD, tiebreaker off": (10, -14),
               "naive similarity": (-12, 10), "random": (10, 4)}
    for label, (key, colour) in points.items():
        m = metrics["strategies"][key]
        faded = "tiebreaker off" in label
        ax.scatter(m["coherence"], m["same_institution_pairs"], s=120, color=colour,
                   zorder=3, edgecolor=SURFACE, linewidth=1.6,
                   alpha=0.45 if faded else 1.0)
        ax.annotate(label, (m["coherence"], m["same_institution_pairs"]),
                    textcoords="offset points", xytext=offsets[label],
                    fontsize=9.5, color=INK_MUTED if faded else INK)

    on = metrics["strategies"]["rigad"]
    off = metrics["strategies"]["rigad-ignoring-institution"]
    ax.annotate("", xy=(on["coherence"], on["same_institution_pairs"]),
                xytext=(off["coherence"], off["same_institution_pairs"]),
                arrowprops={"arrowstyle": "->", "color": INK_MUTED, "linewidth": 1.4})

    ax.set_xlabel("shared research theme  (mean pairwise similarity within a group)",
                  fontsize=9, color=INK_MUTED)
    ax.set_ylabel("pairs from the same institution  (share)", fontsize=9, color=INK_MUTED)
    ax.set_ylim(-0.035, 0.42)
    cost = 100 * (off["coherence"] - on["coherence"]) / off["coherence"]
    ax.set_title(
        f"Mixing institutions is nearly free\n"
        f"cohort of {size}: costs {cost:.1f}% of theme quality,\n"
        f"removes every same-institution pairing",
        fontsize=11.5, loc="left", pad=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "allocation_tradeoff.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    docs = FIGURES_DIR.parent

    track_metrics = json.loads((docs / "track_metrics.json").read_text())
    figure_consistency(track_metrics)

    alloc_path = docs / "metrics.json"
    if alloc_path.exists():
        figure_allocation_tradeoff(json.loads(alloc_path.read_text()))

    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {path.name}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
