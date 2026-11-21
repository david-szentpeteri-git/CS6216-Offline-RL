"""Rebuild figures/cql_wins_losses.pdf from Kumar et al. (2020), Table 1, p. 8.

Gap = CQL(H) minus the best of the five baselines on each D4RL gym dataset.
Run from the repo root:  python src/make_cql_gaps_figure.py
"""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ["SAC", "BC", "BEAR", "BRAC-p", "BRAC-v"]
BLUE, ORANGE = "#2F6FB0", "#E0812B"


def load_rows():
    with open(ROOT / "data" / "cql_table1.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        scores = {b: float(r[b]) for b in BASELINES}
        r["best_name"] = max(scores, key=scores.get)
        r["best"] = scores[r["best_name"]]
        r["cql"] = float(r["CQL"])
        r["gap"] = round(r["cql"] - r["best"], 1)
    return rows


def main():
    rows = load_rows()
    groups = {"single": "Single-policy datasets", "multi": "Multi-policy datasets"}

    # Sanity checks against the numbers quoted in sections/03_practice.tex
    med = {g: statistics.median(r["gap"] for r in rows if r["group"] == g) for g in groups}
    losses = [r["task"] for r in rows if r["gap"] < 0]
    assert round(med["single"], 1) == -0.4 and round(med["multi"], 1) == 23.3, med
    assert len(losses) == 6 and max(-r["gap"] for r in rows if r["gap"] < 0) <= 3.4

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    for ax, (g, title) in zip(axes, groups.items()):
        sub = [r for r in rows if r["group"] == g][::-1]
        y = range(len(sub))
        ax.barh(y, [r["gap"] for r in sub],
                color=[BLUE if r["gap"] >= 0 else ORANGE for r in sub])
        ax.set_yticks(list(y), [r["task"] for r in sub], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-10, 100)
        ax.set_title(f"{title} (median gap {med[g]:+.1f})", fontsize=10, loc="left")
        for yi, r in zip(y, sub):
            ax.text(101, yi, f"{r['cql']:.1f} vs {r['best']:.1f} ({r['best_name']})",
                    va="center", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("CQL(H) minus best baseline (normalised return points)")
    fig.tight_layout()

    out = ROOT / "figures" / "cql_wins_losses.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {out}\nMedian gaps: {med}\nCQL behind on: {losses}")


if __name__ == "__main__":
    main()
