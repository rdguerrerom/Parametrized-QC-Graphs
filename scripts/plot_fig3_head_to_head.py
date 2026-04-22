"""Figure 3 — Nash vs SA-DQAS head-to-head at matched budget.

Panels:
  (a) Grouped bars per topology (Nash vs SA-DQAS); reference Φ=4.10
      line with in-margin label; ΔΦ annotations placed directly above
      each bar pair.
  (b) Per-player decomposition at the grid 2×2 equilibrium as a proper
      stacked horizontal bar with all four components always rendered
      (anti-sim segment shown as a visible tiny slice, not hidden).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting import palette as P
from plotting import stats as ST
from plotting import style as S

S.use_style()

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_f6():
    with open(RESULTS / "exp_f6_vs_sa_dqas.json") as f:
        return json.load(f)


def load_f6_multi_seed():
    """Per-topology {nash: [...], sa: [...]} list of Φ across seeds."""
    ms_path = RESULTS / "multi_seed_f6.json"
    if not ms_path.exists():
        return None
    with open(ms_path) as f:
        ms = json.load(f)
    from collections import defaultdict
    agg = defaultdict(lambda: {"nash": [], "sa": []})
    for _, pay in ms["per_seed"].items():
        for r in pay["rows"]:
            agg[r["topology"]]["nash"].append(r["nash"]["phi_nash"])
            agg[r["topology"]]["sa"].append(r["sa_baseline"]["phi_nash"])
    return dict(agg)


def draw_panel_a(ax, rows):
    topos = [r["topology"] for r in rows]
    labels = [t.replace("_4q_subset", "").replace("_4", "")
               .replace("all_to_all", "all-to-all")
               .replace("heavy_hex", "heavy-hex")
               .replace("grid_2x2", "grid 2×2")
               .replace("rydberg", "Rydberg")
               .replace("_", " ")
              for t in topos]
    nash_phi = [r["nash"]["phi_nash"] for r in rows]
    sa_phi = [r["sa_baseline"]["phi_nash"] for r in rows]

    # Replace single-seed values with multi-seed MEANS + 95% bootstrap CI on
    # the mean. We use the mean (not median) as the point estimate so that
    # the ΔΦ annotations reflect the actual per-topology differences — at
    # n=5 the median quantises to one sample and can mask real gaps.
    ms = load_f6_multi_seed()
    nash_err = None; sa_err = None
    if ms is not None:
        nash_phi = []; sa_phi = []
        nash_lo = []; nash_hi = []; sa_lo = []; sa_hi = []
        for t in topos:
            n_arr = np.asarray(ms[t]["nash"]); s_arr = np.asarray(ms[t]["sa"])
            mN = float(n_arr.mean()); mS = float(s_arr.mean())
            _, lN, hN = ST.bootstrap_ci_95(ms[t]["nash"])
            _, lS, hS = ST.bootstrap_ci_95(ms[t]["sa"])
            nash_phi.append(mN); sa_phi.append(mS)
            nash_lo.append(max(0.0, mN - lN)); nash_hi.append(max(0.0, hN - mN))
            sa_lo.append(max(0.0, mS - lS));  sa_hi.append(max(0.0, hS - mS))
        nash_err = np.vstack([nash_lo, nash_hi])
        sa_err = np.vstack([sa_lo, sa_hi])

    x = np.arange(len(topos)); w = 0.34
    ax.bar(x - w/2, nash_phi, w, color=P.NASH, edgecolor="black",
           linewidth=0.4, yerr=nash_err, capsize=3,
           error_kw=dict(lw=0.7, ecolor="0.2"),
           label=("Nash (4 players)" +
                  (" · mean ±95% CI" if ms is not None else "")))
    ax.bar(x + w/2, sa_phi, w, color=P.SA_DQAS, edgecolor="black",
           linewidth=0.4, yerr=sa_err, capsize=3,
           error_kw=dict(lw=0.7, ecolor="0.2"),
           label="SA-DQAS")

    ax.axhline(4.10, ls="--", lw=0.8, color="black", alpha=0.7, zorder=1)
    # Reference-line label in the right margin
    ax.text(1.00, 4.10, r"  $\Phi{=}4.10$",
            transform=ax.get_yaxis_transform(),
            fontsize=S.FS_ANNOT, va="center", ha="left", color="0.15")

    # ΔΦ annotations above the TALLER bar + its upper whisker.
    for i, (n, s) in enumerate(zip(nash_phi, sa_phi)):
        gap = n - s
        top = max(n, s)
        # Lift past the 95%-CI whisker if present.
        whisker_h = 0.0
        if nash_err is not None:
            whisker_h = max(nash_err[1, i], sa_err[1, i])
        y_off = (0.14 if labels[i].startswith("grid") else 0.05) + whisker_h
        ax.annotate(fr"$\Delta\Phi {{=}} {gap:+.2f}$",
                    xy=(i, top + y_off), ha="center", va="bottom",
                    fontsize=S.FS_ANNOT, color="black", fontweight="bold")

    # Definition of ΔΦ in the upper-left, far from bars.
    ax.text(0.02, 0.96,
            r"$\Delta\Phi \equiv \Phi_{\mathrm{Nash}} - \Phi_{\mathrm{SA}}$",
            transform=ax.transAxes, fontsize=S.FS_ANNOT,
            va="top", ha="left", color="0.15")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=S.FS_TICK)
    ax.set_ylabel(r"Nash potential,  $\Phi$")
    # Tighten ylim once the whiskers + ΔΦ labels fit cleanly.
    ax.set_ylim(3.0, 4.30)
    # Legend OUTSIDE the plot area, below, matching Fig 4(b).
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False, fontsize=S.FS_ANNOT,
              handlelength=1.4, handletextpad=0.5, columnspacing=1.2)
    S.panel_label(ax, "(a)")


def draw_panel_b(ax, rows):
    grid_row = next(r for r in rows if "grid" in r["topology"])
    nash = grid_row["nash"]
    sa = grid_row["sa_baseline"]

    weights = {"anti_bp": 0.3, "anti_sim": 0.2, "performance": 1.0,
               "hardware": 0.2}
    player_order = ["anti_bp", "anti_sim", "performance", "hardware"]

    def breakdown(run):
        return {
            "anti_bp":     weights["anti_bp"] * run["f1_eff_dim_ratio"],
            "anti_sim":    weights["anti_sim"] * run["f2_magic_per_qubit"],
            "performance": weights["performance"] * run["f3_performance"],
            "hardware":   -weights["hardware"] * run["f4_hardware_cost"],
        }
    parts = {"Nash": breakdown(nash), "SA-DQAS": breakdown(sa)}

    y = np.arange(len(parts))[::-1]
    h = 0.55

    for i, name in enumerate(parts):
        left_pos = 0.0
        neg_left = 0.0
        row_width = {}
        for p in player_order:
            v = parts[name][p]
            color = P.PLAYER_COLORS[p]
            if v >= 0:
                ax.barh(y[i], v, left=left_pos, height=h,
                        color=color, edgecolor="black", linewidth=0.5)
                left_pos += v
            else:
                ax.barh(y[i], v, left=neg_left, height=h,
                        color=color, edgecolor="black", linewidth=0.5,
                        hatch="///", alpha=0.95)
                neg_left += v
            row_width[p] = v
        # Total Φ marker at the right edge of the bar.
        total_phi = left_pos + neg_left
        ax.plot(total_phi, y[i], marker="|", color="black",
                markersize=14, mew=1.8, zorder=5)
        ax.annotate(fr"$\Phi {{=}} {total_phi:.2f}$",
                    xy=(total_phi + 0.08, y[i]), ha="left", va="center",
                    fontsize=S.FS_ANNOT)

    # Zero baseline drawn explicitly.
    ax.axvline(0, color="black", lw=0.7, zorder=2)
    ax.tick_params(axis="x", which="both", length=3)

    ax.set_yticks(y)
    ax.set_yticklabels(list(parts.keys()), fontsize=S.FS_LABEL)
    ax.set_xlabel(r"weighted contribution to $\Phi$  at grid 2$\times$2")
    ax.set_xlim(-0.6, 5.2)
    # (The f_2 ≈ 0 note is now carried by the legend label itself, so
    # no in-panel callout is needed.)

    # Proxy-patch legend so every player's color is visible even when
    # a given run happens to contribute ≈ 0 to that player. We annotate
    # f_2 explicitly as "(≈0, not visible)" so readers don't hunt for a
    # missing green segment.
    from matplotlib.patches import Patch
    labels_override = dict(P.PLAYER_LABELS)
    labels_override["anti_sim"] = r"$f_2$ anti-sim  (${\approx}0$; not visible)"
    handles = [Patch(facecolor=P.PLAYER_COLORS[p], edgecolor="black",
                     linewidth=0.3, label=labels_override[p])
               for p in player_order]
    handles.append(Patch(facecolor="white", hatch="///",
                         edgecolor="black", linewidth=0.3,
                         label=r"penalty ($-$)"))
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=S.FS_ANNOT, frameon=False,
              handletextpad=0.45, columnspacing=1.5)
    S.panel_label(ax, "(b)")


def main():
    f6 = load_f6(); rows = f6["rows"]

    fig = plt.figure(figsize=(S.DOUBLE_COL_IN, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.30], wspace=0.36,
                          left=0.08, right=0.98, top=0.92, bottom=0.28)
    draw_panel_a(fig.add_subplot(gs[0, 0]), rows)
    draw_panel_b(fig.add_subplot(gs[0, 1]), rows)
    S.save_figure(fig, "fig3_head_to_head")


if __name__ == "__main__":
    main()
