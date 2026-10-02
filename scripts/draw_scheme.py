"""Diagram of the extraction and review pipeline (Fig. 1 of the paper)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "Times New Roman"})

W, H = 0.15, 0.24
TOP, BOTTOM, MIDDLE = 0.56, 0.08, 0.32
BOXES = {
    "request": (0.00, TOP, "User request"),
    "prompt": (0.19, TOP, "Prompt with\nontology and\nexamples"),
    "model": (0.38, TOP, "Local language\nmodel"),
    "record": (0.57, TOP, "Greedy record,\ntoken\nlog-probabilities"),
    "samples": (0.57, BOTTOM, "Four sampled\nrecords"),
    "score": (0.76, MIDDLE, "Error score\nR, D or L"),
    "accept": (0.95, TOP, "Automatic\nacceptance"),
    "review": (0.95, BOTTOM, "Manual\nreview"),
}


def left(k):
    x, y, _ = BOXES[k]
    return x, y + H / 2


def right(k):
    x, y, _ = BOXES[k]
    return x + W, y + H / 2


def bottom(k):
    x, y, _ = BOXES[k]
    return x + W / 2, y


def path(ax, *points):
    for a, b in zip(points[:-2], points[1:-1], strict=True):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=11, lw=1, color="black",
                                 shrinkA=0, shrinkB=0))


def main(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    ax.set_xlim(-0.01, 1.11)
    ax.set_ylim(0.04, 0.84)
    ax.axis("off")
    for x, y, text in BOXES.values():
        ax.add_patch(Rectangle((x, y), W, H, fc="white", ec="black", lw=1))
        ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=11)
    path(ax, right("request"), left("prompt"))
    path(ax, right("prompt"), left("model"))
    path(ax, right("model"), left("record"))
    mx, my = bottom("model")
    path(ax, (mx, my), (mx, left("samples")[1]), left("samples"))
    join = 0.735
    path(ax, right("record"), (join, right("record")[1]), (join, MIDDLE + H * 0.7), (0.76, MIDDLE + H * 0.7))
    path(ax, right("samples"), (join, right("samples")[1]), (join, MIDDLE + H * 0.3), (0.76, MIDDLE + H * 0.3))
    split = 0.93
    sx, sy = right("score")
    path(ax, (sx, sy), (split, sy), (split, left("accept")[1]), left("accept"))
    path(ax, (split, sy), (split, left("review")[1]), left("review"))
    ax.text(split + 0.008, 0.49, "low", fontsize=10, ha="left", va="center")
    ax.text(split + 0.008, 0.38, "high", fontsize=10, ha="left", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "figures" / "pipeline.png")
