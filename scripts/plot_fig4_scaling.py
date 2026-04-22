"""Figure 4 — H₂ VQE BP signature + TFIM scaling.

Panels:
  (a) H₂ energies — Nash (2 ops) and HEA (14 ops) vs E_HF / E_FCI with
      rotated-label reference lines. BP-signature callout inside the
      plot area, anchored to the HEA marker.
  (b) TFIM rel. error vs n with warm-start (blue) vs cold-start (grey
      dashed with filled markers). Legend OUTSIDE the plot area.
  (c) Wall-clock per Nash iter vs n with linear fit and R² annotation.

Framing note: panel (b) does NOT claim convergence to exact as n grows.
It shows structural-search VALUE (gap between warm and cold start) at
each n, and we flag this explicitly in the caption.
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


def load_d1():
    with open(RESULTS / "exp_d1_tfim_scaling.json") as f:
        return json.load(f)


def load_d1_multi_seed():
    """Return dict of n -> {'rel_err_pct': [...], 'iter_s': [...]} across seeds.

    None if the multi-seed sweep hasn't been run yet.
    """
    ms_path = RESULTS / "multi_seed_d1.json"
    if not ms_path.exists():
        return None
    with open(ms_path) as f:
        ms = json.load(f)
    from collections import defaultdict
    agg = defaultdict(lambda: {"rel_err_pct": [], "iter_s": []})
    for _, pay in ms["per_seed"].items():
        for r in pay["rows"]:
            agg[r["n"]]["rel_err_pct"].append(r["rel_error_vs_exact"] * 100)
            agg[r["n"]]["iter_s"].append(r["mean_iter_s"])
    return {n: agg[n] for n in sorted(agg)}


def load_e2():
    with open(RESULTS / "exp2_h2_heavy_hex.json") as f:
        return json.load(f)


def load_c3():
    """LiH VQE result (frozen-core 6-qubit). None if not yet generated."""
    p = RESULTS / "exp_c3_lih_vqe.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_c3_givens_nash():
    """LiH Nash-on-Givens compression result. None if not yet generated."""
    p = RESULTS / "exp_c3_lih_givens_nash.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# Pre-Path-B |+⟩ⁿ cold-start D1: Nash plateaus at E = -n. Hardcoded
# because the original JSON was overwritten by Path-B reruns.
COLD_START_REL_ERR = {4: 15.94, 6: 17.77, 8: 18.68}  # percent


def _draw_chem_row(ax, *, E_HF, E_ref, ref_label, cases,
                   x_label, pad_left=0.02, pad_right=0.04,
                   bp_marker_idx=None, panel_letter=None):
    """One chemistry sub-panel showing Nash / HEA (/ Nash-Givens) markers
    on an energy axis versus E_HF and E_ref reference lines.

    `cases` is a list of (row_label, E, n_ops, color, marker) tuples, one
    per row (top row = first entry). `bp_marker_idx` is the index of the
    HEA row whose marker gets the "BP: trapped above E_HF" callout; pass
    None to omit the callout.
    """
    n_rows = len(cases)
    ax.axvline(E_HF, ls="--", color="black", lw=0.9, zorder=1)
    ax.axvline(E_ref, ls="--", color=P.VERMILLION, lw=0.9, zorder=1)
    # Compute the x-axis extent *first*, then decide whether E_HF and
    # E_ref are too close (in panel-width fraction) to stack their
    # rotated labels at the top. Also decide per-marker whether the
    # "N ops" annotation needs to be sideways so it clears any rotated
    # reference label sitting at the top.
    E_extents = [E_ref, E_HF] + [c[1] for c in cases]
    lo = min(E_extents) - pad_left
    hi = max(E_extents) + pad_right
    x_width = hi - lo
    refs_close = abs(E_HF - E_ref) < 0.15 * x_width

    for i, (label, E, ops, color, marker) in enumerate(cases):
        y = n_rows - i - 1
        ax.plot([E], [y], marker=marker, color=color, markersize=9,
                mec="black", mew=0.7, linestyle="")
        # Marker sits on (or very close to) a vertical reference line when
        # |E − ref| is tiny relative to panel width. The default "N ops"
        # annotation then collides with the rotated reference label
        # sitting at top_y on that same x. Shift the annotation to the
        # right of the marker whenever this happens, regardless of
        # whether the two reference labels themselves are close.
        on_ref = (abs(E - E_HF) < 0.05 * x_width
                  or abs(E - E_ref) < 0.05 * x_width)
        if on_ref:
            ax.annotate(f"{ops} ops", xy=(E, y), xytext=(8, 0),
                        textcoords="offset points", fontsize=S.FS_ANNOT,
                        color="black", va="center", ha="left")
        else:
            ax.annotate(f"{ops} ops", xy=(E, y), xytext=(0, 9),
                        textcoords="offset points", fontsize=S.FS_ANNOT,
                        color="black", va="bottom", ha="center")
        ax.axhline(y, color="0.92", lw=0.5, zorder=0)

    # Reference-line labels. When E_HF and E_ref are close (the LiH case),
    # put one at the top of the panel and the other at the bottom so they
    # do not stack. Otherwise both at top (the H₂ case).
    top_y = (n_rows - 1) + 0.55
    bot_y = -0.55
    if refs_close:
        ax.text(E_HF, top_y, r" $E_{\mathrm{HF}}$",
                fontsize=S.FS_ANNOT, ha="left", va="center", rotation=90,
                color="black")
        ax.text(E_ref, bot_y, f" {ref_label}",
                fontsize=S.FS_ANNOT, ha="left", va="center", rotation=90,
                color=P.VERMILLION)
    else:
        ax.text(E_HF, top_y, r" $E_{\mathrm{HF}}$",
                fontsize=S.FS_ANNOT, ha="left", va="center", rotation=90,
                color="black")
        ax.text(E_ref, top_y, f" {ref_label}",
                fontsize=S.FS_ANNOT, ha="left", va="center", rotation=90,
                color=P.VERMILLION)

    if bp_marker_idx is not None:
        E_hea = cases[bp_marker_idx][1]
        y_hea = n_rows - bp_marker_idx - 1
        # Anchor the "BP: trapped…" text clearly to the RIGHT of the HEA
        # square (not below) so it can't be misread as pointing at any
        # other marker that happens to share the same y-row.
        ax.annotate(
            r"BP: trapped above $E_{\mathrm{HF}}$",
            xy=(E_hea, y_hea),
            xytext=(E_hea + 0.5 * pad_right, y_hea - 0.45),
            fontsize=S.FS_ANNOT, color=P.VERMILLION,
            arrowprops=dict(arrowstyle="->", color=P.VERMILLION, lw=0.8,
                            shrinkA=0, shrinkB=4,
                            connectionstyle="arc3,rad=0.15"),
            ha="center", va="top",
        )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([c[0] for c in cases][::-1], fontsize=S.FS_LABEL)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xlabel(x_label)
    E_extents = [E_ref, E_HF] + [c[1] for c in cases]
    lo = min(E_extents) - pad_left
    hi = max(E_extents) + pad_right
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.0, (n_rows - 1) + 0.85)
    if panel_letter:
        S.panel_label(ax, panel_letter, x=-0.23)


def draw_panel_a_h2(ax, e2):
    runs = e2["runs"]
    cases = [
        ("Nash",         runs["nash_free"]["final_energy_Ha"],
                         runs["nash_free"]["n_ops"], P.NASH, P.MARKER_NASH),
        ("HEA (θ-only)", runs["hea_baseline"]["final_energy_Ha"],
                         runs["hea_baseline"]["n_ops"], P.HEA, P.MARKER_HEA),
    ]
    _draw_chem_row(
        ax,
        E_HF=e2["references"]["HF_energy_Ha"],
        E_ref=e2["references"]["FCI_energy_Ha"],
        ref_label=r"$E_{\mathrm{FCI}}$",
        cases=cases,
        x_label=r"H$_2$ / STO-3G  [Ha]",
        pad_left=0.02, pad_right=0.04,
        bp_marker_idx=1, panel_letter=r"(a$_1$)",
    )


def draw_panel_a_lih(ax, c3, givens_nash=None):
    """LiH sub-panel. Shows up to three rows:
       1. Nash (Givens seed) — chemistry-aware compressed ansatz [if given]
       2. Nash (HE gates)    — hardware-generic Nash (plateaus at HF)
       3. HEA (θ-only)       — BP-trapped random-init baseline
    """
    runs = c3["runs"]
    E_ref = c3["references"]["active_ground_energy_Ha"]  # active-space GS
    cases = []
    if givens_nash is not None:
        cases.append((
            "Nash (Givens)",
            givens_nash["final_energy_Ha"],
            givens_nash["final_n_ops"],
            P.NASH, P.MARKER_COLD,   # diamond, distinguishes from HE-Nash
        ))
    cases.append((
        "Nash (HE gates)" if givens_nash is not None else "Nash",
        runs["nash_free"]["final_energy_Ha"],
        runs["nash_free"]["n_ops"],
        P.NASH, P.MARKER_NASH,
    ))
    cases.append((
        "HEA (θ-only)",
        runs["hea_baseline"]["final_energy_Ha"],
        runs["hea_baseline"]["n_ops"],
        P.HEA, P.MARKER_HEA,
    ))
    _draw_chem_row(
        ax,
        E_HF=c3["references"]["HF_energy_Ha"],
        E_ref=E_ref, ref_label=r"$E_{\mathrm{GS}}$",
        cases=cases,
        x_label=r"LiH / STO-3G (6 q)  [Ha]",
        pad_left=0.02, pad_right=0.04,
        bp_marker_idx=len(cases) - 1,  # HEA row is always last
        panel_letter=r"(a$_2$)",
    )


def draw_panel_a(ax, e2):
    """Legacy single-panel H₂ view (used when no LiH data present)."""
    draw_panel_a_h2(ax, e2)


def draw_panel_b(ax, d1):
    rows = [r for r in d1["rows"] if "error" not in r]
    ns = np.asarray([r["n"] for r in rows])
    warm = np.asarray([r["rel_error_vs_exact"] * 100 for r in rows])
    cold = np.asarray([COLD_START_REL_ERR[r["n"]] for r in rows])

    # Multi-seed bootstrap CI band for warm-start if available.
    # Use MEAN as point estimate (median quantises at n=5).
    ms = load_d1_multi_seed()
    if ms is not None:
        ns_ms = np.asarray(list(ms.keys()), dtype=float)
        lo = []; hi = []; mean_vals = []
        for n in ms:
            arr = ms[n]["rel_err_pct"]
            mean_vals.append(float(np.mean(arr)))
            _, l, h = ST.bootstrap_ci_95(arr)
            lo.append(l); hi.append(h)
        ax.fill_between(ns_ms, lo, hi, color=P.NASH, alpha=0.20,
                        edgecolor="none", zorder=1)
        warm = np.asarray(mean_vals)
        ns = ns_ms.astype(int)

    ax.plot(ns, warm, "-o", color=P.NASH, lw=1.6, markersize=6,
            mec="black", mew=0.5, zorder=3,
            label=("QAOA warm-start + Nash" +
                   (" (mean, 95% CI, 5 seeds)" if ms is not None else "")))
    # Filled grey squares for cold-start — visible at small print size.
    ax.plot(ns, cold, "--s", color=P.COLD_START, lw=1.3, markersize=6,
            markerfacecolor=P.COLD_START, mec="black", mew=0.5,
            label=r"$|+\rangle^n$ cold-start")

    ax.axhline(10, color="0.7", ls=":", lw=0.7, alpha=0.8, zorder=0)

    ax.set_xlabel(r"qubit count,  $n$")
    ax.set_ylabel(r"rel. error vs FCI / exact [%]")
    ax.set_xticks(ns); ax.set_xlim(3.5, 8.5)
    ax.set_ylim(0, 26)
    # Legend OUTSIDE the plot area, below, so it does not compete with
    # data or produce text-text overlap.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=1, frameon=False, fontsize=S.FS_ANNOT,
              handletextpad=0.5)
    S.panel_label(ax, "(b)", x=-0.25)


def draw_panel_c(ax, d1):
    rows = [r for r in d1["rows"] if "error" not in r]
    ns = np.asarray([r["n"] for r in rows])
    sec = np.asarray([r["mean_iter_s"] for r in rows])

    ms = load_d1_multi_seed()
    if ms is not None:
        ns_ms = np.asarray(list(ms.keys()), dtype=float)
        lo = []; hi = []; mean_vals = []
        for n in ms:
            arr = ms[n]["iter_s"]
            mean_vals.append(float(np.mean(arr)))
            _, l, h = ST.bootstrap_ci_95(arr)
            lo.append(l); hi.append(h)
        ax.fill_between(ns_ms, lo, hi, color=P.NASH, alpha=0.20,
                        edgecolor="none", zorder=1)
        ns = ns_ms.astype(int)
        sec = np.asarray(mean_vals)

    coef = np.polyfit(ns, sec, 1)
    xs = np.linspace(3.5, 8.5, 64)
    ax.plot(xs, np.polyval(coef, xs), ":", color="0.4", lw=1.0, zorder=1,
            label="linear fit")
    ax.plot(ns, sec, "-o", color=P.NASH, lw=1.6, markersize=6,
            mec="black", mew=0.5, zorder=3)

    # Report only the slope (R² is uninformative for a 3-point fit).
    ax.text(0.05, 0.92,
            f"slope ≈ {coef[0]:.1f} s / qubit",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color="0.2",
            va="top")

    ax.set_xlabel(r"qubit count,  $n$")
    ax.set_ylabel("wall-clock / Nash iter  [s]")
    ax.set_xticks(ns); ax.set_xlim(3.5, 8.5)
    ax.set_ylim(0, sec.max() * 1.30)
    S.panel_label(ax, "(c)", x=-0.25)


def main():
    e2 = load_e2(); d1 = load_d1(); c3 = load_c3()
    c3_givens = load_c3_givens_nash()

    # When LiH has a 3-row panel (Nash-Givens + Nash-HE + HEA), we need
    # more vertical space than the 2-row case.
    if c3 is not None:
        height = 4.8 if c3_givens is not None else 4.4
    else:
        height = 3.2
    fig = plt.figure(figsize=(S.DOUBLE_COL_IN, height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.2, 1.0], wspace=0.55,
                          left=0.08, right=0.985, top=0.96, bottom=0.16)

    if c3 is not None:
        # Panel (a) gets a 2-row sub-gridspec: top = H₂, bottom = LiH.
        # LiH gets more relative height when it has 3 rows.
        lih_relweight = 1.4 if c3_givens is not None else 1.0
        gs_a = gs[0, 0].subgridspec(
            2, 1, hspace=1.35, height_ratios=[1.0, lih_relweight],
        )
        draw_panel_a_h2(fig.add_subplot(gs_a[0, 0]), e2)
        draw_panel_a_lih(fig.add_subplot(gs_a[1, 0]), c3,
                          givens_nash=c3_givens)
    else:
        draw_panel_a(fig.add_subplot(gs[0, 0]), e2)

    draw_panel_b(fig.add_subplot(gs[0, 1]), d1)
    draw_panel_c(fig.add_subplot(gs[0, 2]), d1)
    S.save_figure(fig, "fig4_scaling")


if __name__ == "__main__":
    main()
