"""Figure 2 — Headline: BP/simulability Pareto frontier on MaxCut K₄.

Double-column layout, side-by-side panels:
  (a) 16 Nash equilibria in (M_2/n, ⟨H⟩) space as single-shape markers
      coloured by anti-sim weight w_2. Grey polyline connects the Pareto
      corners. Two endpoint callouts (Clifford / non-Clifford) sit OUTSIDE
      the data cluster, upper-right and lower-right respectively.
  (b) Circuit complexity (depth and n_ops) per corner sorted by M_2/n.

Redesign v2 (post-SWE fifth-pass critique): removed the w_1 shape encoding
(color alone now carries the weight sweep), dropped the star / "K₄ optimum"
label / ⟨H⟩=4.0 hline (all redundant with the Clifford endpoint callout),
dropped the intermediate-corner annotation (belongs in the caption), and
removed the x-jitter at M_2=0. The goal is a figure a reader can decode in
one glance at PRL print size.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting import palette as P
from plotting import style as S

S.use_style()

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_e3():
    with open(RESULTS / "exp3_weight_sweep.json") as f:
        return json.load(f)


def pareto_front_indices(xs, ys):
    """Non-dominated subset in (x ↓ preferred, y ↑ preferred) space."""
    n = len(xs)
    xs, ys = np.asarray(xs), np.asarray(ys)
    keep = []
    for j in range(n):
        dom = False
        for k in range(n):
            if k == j:
                continue
            if (xs[k] <= xs[j] and ys[k] >= ys[j]
                    and (xs[k] < xs[j] or ys[k] > ys[j])):
                dom = True
                break
        if not dom:
            keep.append(j)
    return sorted(keep, key=lambda j: xs[j])


def draw_panel_a(ax, rows):
    xs = np.asarray([r["magic_per_qubit"] for r in rows])
    ys = np.asarray([r["maxcut_value"] for r in rows])
    wsim = np.asarray([r["w_anti_sim"] for r in rows])

    sim_vals = sorted(set(wsim.tolist()))
    cmap = plt.get_cmap("viridis")
    sim_color = {v: cmap(i / max(1, len(sim_vals) - 1))
                 for i, v in enumerate(sim_vals)}

    # Pareto frontier polyline.
    pf = pareto_front_indices(xs, ys)
    ax.plot(xs[pf], ys[pf], "-", color="0.55", lw=1.4, zorder=2, alpha=0.9)

    # Single-shape markers; colour alone encodes w_2. Overlapping points
    # at M_2 ≈ 0 stack naturally (alpha + black edge = "darker = more").
    marker_colors = [sim_color[s] for s in wsim]
    ax.scatter(xs, ys, s=55, c=marker_colors, marker="o",
               edgecolors="black", linewidths=0.5, alpha=0.85, zorder=4)

    # Endpoint callouts — placed OUTSIDE the data convex hull so they
    # never cover a marker.
    cliff_idx = int(np.argmax([y if x < 1e-3 else -np.inf
                               for x, y in zip(xs, ys)]))
    ax.annotate(
        "Clifford endpoint\n" r"$M_2{=}0,\; \langle H\rangle{=}4.0$",
        xy=(xs[cliff_idx], ys[cliff_idx]),
        xytext=(0.20, 4.08),
        fontsize=S.FS_ANNOT,
        arrowprops=dict(arrowstyle="->", color="0.25", lw=0.7,
                        shrinkA=0, shrinkB=5),
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.60", lw=0.4),
    )
    high_idx = int(np.argmax(xs))
    ax.annotate(
        "non-Clifford endpoint\n"
        fr"$M_2{{=}}{xs[high_idx]:.2f},\; \langle H\rangle{{=}}{ys[high_idx]:.2f}$",
        xy=(xs[high_idx], ys[high_idx]),
        xytext=(0.14, 3.30),
        fontsize=S.FS_ANNOT,
        arrowprops=dict(arrowstyle="->", color="0.25", lw=0.7,
                        shrinkA=0, shrinkB=5),
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.60", lw=0.4),
    )

    # Inline "Pareto frontier" label on the polyline itself (no legend).
    ax.text(0.02, 3.78, "Pareto frontier",
            fontsize=S.FS_ANNOT, color="0.40",
            rotation=-70, rotation_mode="anchor",
            ha="left", va="center")

    # Discrete colorbar for w_2 — the single retained encoding.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm
    bounds = np.arange(len(sim_vals) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(len(sim_vals)),
                        pad=0.02, fraction=0.045, shrink=0.85)
    cbar.ax.set_yticklabels([f"{v:g}" for v in sim_vals], fontsize=7)
    cbar.set_label(r"anti-sim weight  $w_2$",
                    labelpad=4, fontsize=S.FS_ANNOT)

    ax.set_xlabel(r"non-stabilizerness,  $M_2/n$")
    ax.set_ylabel(r"MaxCut expectation,  $\langle H\rangle$")
    # Tight axes now that all in-plot redundancy is gone.
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(3.25, 4.18)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", length=4.0, width=0.9,
                   labelsize=S.FS_TICK)
    ax.tick_params(axis="both", which="minor", length=2.2, width=0.7)
    S.panel_label(ax, "(a)", y=1.03)


def draw_panel_b(ax, rows):
    mags = np.asarray([r["magic_per_qubit"] for r in rows])
    order = np.argsort(mags)
    depth = np.asarray([r["depth"] for r in rows])[order]
    nops = np.asarray([r["n_ops"] for r in rows])[order]

    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w/2, depth, w, color=P.PLAYER_HARDWARE, edgecolor="black",
           linewidth=0.3, label="depth")
    ax.bar(x + w/2, nops, w, color=P.PLAYER_PERFORMANCE, edgecolor="black",
           linewidth=0.3, label=r"$n_{\mathrm{ops}}$")

    tick_idx = [0, 3, 7, 11, 15]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{t+1}" for t in tick_idx], fontsize=7)
    ax.set_xlabel(r"weight corner  (sorted by $M_2/n$)")
    ax.set_ylabel("circuit complexity")
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, max(nops.max() * 1.12, 5))
    # Legend inside axes (upper-left, frameless) — matches (a)'s
    # stripped-down chrome.
    ax.legend(loc="upper left", frameon=False,
              fontsize=S.FS_ANNOT, handletextpad=0.4,
              ncol=1, borderpad=0.2)
    S.panel_label(ax, "(b)", y=1.03)


def main():
    e3 = load_e3()
    rows = e3["rows"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(S.DOUBLE_COL_IN, 2.9),
        gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.32),
    )
    draw_panel_a(ax_a, rows)
    draw_panel_b(ax_b, rows)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.18)
    S.save_figure(fig, "fig2_pareto_frontier")


if __name__ == "__main__":
    main()
