"""Figure 1 — Framework schematic + δ_Nash trace.

v4: Panel (a) completely redesigned to eliminate the label-box overlap.
Uses SHORT f_i labels inside the box with colored arrows, and a dedicated
"player legend" below. Equation placed BELOW the box with clear padding.
Graph-state and Circuit-DAG subpanels re-spaced so the map arrow and
wire labels don't collide.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from plotting import palette as P
from plotting import stats as ST
from plotting import style as S

S.use_style()

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_nash_gap():
    """Prefer multi-seed bootstrap CI; fall back to single-seed trace."""
    ms_path = RESULTS / "multi_seed_d1.json"
    if ms_path.exists():
        with open(ms_path) as f:
            ms = json.load(f)
        from collections import defaultdict
        agg = defaultdict(list)
        for _, pay in ms["per_seed"].items():
            for r in pay["rows"]:
                agg[r["n"]].append(r["delta_nash_final"])
        ns = sorted(agg)
        point = [float(np.mean(agg[n])) for n in ns]
        lo = []; hi = []
        for n in ns:
            _, l, h = ST.bootstrap_ci_95(agg[n])
            lo.append(l); hi.append(h)
        return list(ns), point, lo, hi
    with open(RESULTS / "exp_d1_tfim_scaling.json") as f:
        d1 = json.load(f)
    ns = [row["n"] for row in d1["rows"] if "error" not in row]
    gaps = [row["delta_nash_final"] for row in d1["rows"] if "error" not in row]
    return ns, gaps, gaps, gaps


def draw_panel_a(ax):
    """Four-player schematic. Short f_i labels only; descriptors below."""
    # Use the full subplot; no equal aspect so the figure fills the space.
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    # DAG box in the center (a small square with ample margins).
    bx, by, bw, bh = 3.8, 5.6, 2.4, 2.0
    box = mpatches.FancyBboxPatch(
        (bx, by), bw, bh, boxstyle="round,pad=0.1",
        linewidth=1.0, edgecolor="black", facecolor="#F2F2F2",
    )
    ax.add_patch(box)
    ax.text(bx + bw / 2, by + bh * 0.68, "Circuit DAG",
            ha="center", va="center", fontsize=9)
    ax.text(bx + bw / 2, by + bh * 0.32, r"$D(\boldsymbol{\theta})$",
            ha="center", va="center", fontsize=10)

    # Four players in the 4 corners. Label = SHORT f_i only (no descriptor).
    # Descriptors appear in the player legend below.
    players = [
        (2.0, 9.2, r"$f_1$", P.PLAYER_ANTI_BP,     (bx + 0.55 * bw, by + bh)),
        (8.0, 9.2, r"$f_2$", P.PLAYER_ANTI_SIM,    (bx + 0.45 * bw, by + bh)),
        (8.0, 6.2, r"$f_3$", P.PLAYER_PERFORMANCE, (bx + bw,       by + 0.55 * bh)),
        (2.0, 6.2, r"$f_4$", P.PLAYER_HARDWARE,    (bx,            by + 0.55 * bh)),
    ]
    for x, y, label, color, target in players:
        ax.annotate(
            "",
            xy=target, xycoords="data",
            xytext=(x, y), textcoords="data",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                            shrinkA=8, shrinkB=4),
        )
        ax.text(x, y, label, color=color, fontsize=10,
                ha="center", va="center", fontweight="bold")

    # Equation OUTSIDE the box, with generous padding. Single line — the
    # Nash-eq. condition ($\delta_{\mathrm{Nash}} \leq \varepsilon$) is
    # described in the caption, not in the panel, to keep the spacing
    # clear of the Σ-subscript descender.
    ax.text(bx + bw / 2, 4.4,
            r"$\Phi(D) \;=\; \sum_{i=1}^{4}\, s_i\, w_i\, f_i(D)$",
            ha="center", va="center", fontsize=10)
    descriptors = [
        (1.8, 2.6, r"$f_1$ anti-BP",       P.PLAYER_ANTI_BP),
        (8.2, 2.6, r"$f_2$ anti-sim",      P.PLAYER_ANTI_SIM),
        (1.8, 1.6, r"$f_3$ performance",   P.PLAYER_PERFORMANCE),
        (8.2, 1.6, r"$f_4$ hardware",      P.PLAYER_HARDWARE),
    ]
    for x, y, text, color in descriptors:
        ax.text(x, y, text, color=color, fontsize=8,
                ha="center", va="center")

    ax.text(-0.3, 9.9, "(a)", fontsize=10, fontweight="bold")


def draw_panel_b(ax):
    """Graph-state ↦ circuit-DAG embedding with clean label spacing."""
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis("off")

    # LEFT: graph state with labelled vertices.
    gx, gy, r = 2.0, 5.0, 1.2
    positions = {
        "1": (gx, gy + r),
        "2": (gx + r, gy),
        "3": (gx, gy - r),
        "4": (gx - r, gy),
    }
    for lbl, (px, py) in positions.items():
        ax.plot(px, py, "o", ms=11, mfc="white", mec="black", mew=1.0,
                zorder=3)
        ax.text(px, py, lbl, fontsize=7, ha="center", va="center", zorder=4)
    for i, j in [("1", "2"), ("2", "3"), ("3", "4"), ("1", "3")]:
        (ax_x, ay), (bx, by) = positions[i], positions[j]
        ax.plot([ax_x, bx], [ay, by], color="black", lw=0.8, zorder=2)
    ax.text(gx, gy + r + 1.3, r"Graph state  $|G\rangle$",
            ha="center", va="bottom", fontsize=9)

    # CENTER: map arrow. Label raised well above the arrow so that
    # neither it nor the arrow head collide with the wire labels on the
    # right at (dx - 0.32, gate_y[i]).
    ax.annotate("", xy=(7.0, gy), xytext=(3.8, gy),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(5.4, gy + 0.90, r"$\iota : G \mapsto D_G$",
            ha="center", fontsize=9)

    # RIGHT: circuit DAG.
    dx = 7.8
    gate_y = [gy + 1.55, gy + 0.52, gy - 0.52, gy - 1.55]

    # Wire labels clearly separated from title area.
    for i in range(4):
        ax.plot([dx, dx + 4.8], [gate_y[i], gate_y[i]],
                color="black", lw=0.7)
        ax.text(dx - 0.32, gate_y[i], str(i + 1), fontsize=7,
                ha="right", va="center")

    # Hadamards.
    for i in range(4):
        ax.add_patch(mpatches.Rectangle(
            (dx + 0.35, gate_y[i] - 0.22), 0.55, 0.44,
            linewidth=0.9, edgecolor="black", facecolor="#EAEAEA",
            zorder=3))
        ax.text(dx + 0.625, gate_y[i], "H", ha="center", va="center",
                fontsize=7, zorder=4)

    # CZs.
    for xx, a, b in [(dx + 1.80, 0, 1), (dx + 2.55, 1, 2),
                     (dx + 3.30, 2, 3), (dx + 4.05, 0, 2)]:
        ax.plot([xx, xx], [gate_y[a], gate_y[b]], color="black", lw=0.7,
                zorder=2)
        ax.plot(xx, gate_y[a], ".", ms=6, color="black", zorder=3)
        ax.plot(xx, gate_y[b], ".", ms=6, color="black", zorder=3)

    # Title of circuit-DAG subpanel, well above top wire.
    ax.text(dx + 2.4, gate_y[0] + 1.55, "Circuit DAG",
            ha="center", va="bottom", fontsize=9)
    ax.text(dx + 2.4, gate_y[0] + 0.90,
            r"$D_G \,=\, \mathrm{CZ}^{|E|}\!\cdot\! H^{\otimes n}$",
            ha="center", va="bottom", fontsize=9)

    ax.text(-0.2, 9.9, "(b)", fontsize=10, fontweight="bold")


def draw_panel_c(ax):
    ns, med, lo, hi = load_nash_gap()
    ns_arr = np.asarray(ns, dtype=float)
    ax.fill_between(ns_arr, lo, hi, color=P.NASH, alpha=0.22,
                    edgecolor="none", zorder=1,
                    label="mean ± 95% CI (5 seeds)")
    ax.plot(ns_arr, med, "-o", color=P.NASH, markersize=6,
            markerfacecolor=P.NASH, markeredgecolor="black",
            mew=0.5, lw=1.5, zorder=3, label=r"mean $\delta_{\mathrm{Nash}}$")
    ax.axhline(1e-2, color="black", ls=":", lw=0.9, alpha=0.7, zorder=1)
    ax.text(1.01, 1e-2, r" $\varepsilon$-Nash",
            transform=ax.get_yaxis_transform(),
            fontsize=S.FS_ANNOT, va="center", ha="left", color="0.25")

    ax.set_yscale("log")
    ax.set_xlabel(r"qubit count,  $n$")
    ax.set_ylabel(r"Nash-gap residual,  $\delta_{\mathrm{Nash}}$")
    ax.set_xticks(ns)
    ax.set_xlim(ns[0] - 0.5, ns[-1] + 0.5)
    # Data band occupying central 70% of the panel.
    ax.set_ylim(3e-3, 5e-1)
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    # Compact below-axes legend so the CI band is self-identifying.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              fontsize=S.FS_ANNOT, frameon=False, ncol=2,
              handlelength=1.4, handletextpad=0.4, columnspacing=0.8,
              labelspacing=0.1)
    S.panel_label(ax, "(c)", x=-0.25)


def main():
    fig = plt.figure(figsize=(S.DOUBLE_COL_IN, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0],
                          wspace=0.35, left=0.03, right=0.97,
                          top=0.95, bottom=0.26)
    draw_panel_a(fig.add_subplot(gs[0, 0]))
    draw_panel_b(fig.add_subplot(gs[0, 1]))
    draw_panel_c(fig.add_subplot(gs[0, 2]))
    S.save_figure(fig, "fig1_framework")


if __name__ == "__main__":
    main()
