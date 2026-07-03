"""
Monte Carlo prior-width (sigma) sensitivity analysis.

Rescales the log-normal prior widths by x0.5 and x2 and reports how the
per-patient 95% interval width and the standard-schedule feasibility respond.
Medians stay fixed (the sigmas are mean-preserving); only the intervals move.
Amplitude and TAC sigma groups are swept independently.

Outputs:
  results/sigma_sensitivity.csv
  figures/figS1_sigma_sensitivity.{png,pdf,svg}

Run:
  python -m src.sensitivity_analysis            # full sweep + figure
  python -m src.sensitivity_analysis --figure   # rebuild figure from existing CSV
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.dosimetry import compute_patient_dose, SIGMA_DEFAULTS
from src.renal_constraint import max_constrained_activity_from_samples

REF_ACTIVITY_GBQ = 7.4
N_MC             = 1000
BASE_SEED        = 42

# Sigma configurations. Defaults come from SIGMA_DEFAULTS (baseline row).
def _scaled(keys, factor):
    return {k: SIGMA_DEFAULTS[k] * factor for k in keys}

AMP = ["amp_kidney", "amp_tumor", "amp_saliv"]
TAC = ["tac_kidney", "tac_tumor", "tac_saliv"]

CONFIGS = {
    "baseline": {},
    "amp_x0.5": _scaled(AMP, 0.5),
    "amp_x2":   _scaled(AMP, 2.0),
    "tac_x0.5": _scaled(TAC, 0.5),
    "tac_x2":   _scaled(TAC, 2.0),
}

# Figure colors.
C_KIDNEY = "#0F4D92"   # kidney
C_TUMOR  = "#B64342"   # tumor
C_MEAN   = "#42949E"   # mean MC feasibility
C_NEUT   = "#767676"   # baseline marker
C_GRID   = "#CFCECE"   # grid


def apply_publication_style(font_size=8.5, axes_linewidth=1.0):
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype":       "none",   # keep editable <text> in SVG
        "pdf.fonttype":       42,       # editable TrueType in PDF
        "font.size":          font_size,
        "axes.spines.right":  False,
        "axes.spines.top":    False,
        "axes.linewidth":     axes_linewidth,
        "axes.labelpad":      3,
        "legend.frameon":     False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  axes_linewidth,
        "ytick.major.width":  axes_linewidth,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "lines.solid_capstyle": "round",
    })


def _panel_label(ax, s):
    ax.text(-0.16, 1.05, s, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom")


def _style_key(ax, loc):
    """Frameless solid/dashed key (amplitude vs TAC sigma)."""
    handles = [Line2D([0], [0], color=C_NEUT, lw=1.6, ls="-",  label="amplitude σ"),
               Line2D([0], [0], color=C_NEUT, lw=1.4, ls=(0, (4, 2)), label="TAC σ")]
    ax.legend(handles=handles, loc=loc, fontsize=7.2, handlelength=2.0,
              labelspacing=0.3, borderaxespad=0.4)


def make_figure(df):
    """Render Fig S1 from the sweep summary table."""
    apply_publication_style(font_size=8.5, axes_linewidth=1.0)
    d = df.set_index("config")
    xp = np.array([-1.0, 0.0, 1.0])            # log2 positions of 0.5x, 1x, 2x
    amp = lambda c: [d.loc["amp_x0.5", c], d.loc["baseline", c], d.loc["amp_x2", c]]
    tac = lambda c: [d.loc["tac_x0.5", c], d.loc["baseline", c], d.loc["tac_x2", c]]

    fig, ax = plt.subplots(1, 2, figsize=(180 / 25.4, 82 / 25.4))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.86, wspace=0.34)

    common = dict(lw=1.6, markersize=4.2, clip_on=False, zorder=3)
    dash = (0, (4, 2))

    # ---------------- Panel a: per-patient 95% interval width ----------------
    a = ax[0]
    a.axvline(0, color=C_GRID, lw=0.9, ls=(0, (2, 2)), zorder=1)
    a.plot(xp, amp("kidney_95width_rel"), "-",  color=C_KIDNEY, marker="o",
           mfc=C_KIDNEY, mec=C_KIDNEY, **common)
    a.plot(xp, tac("kidney_95width_rel"), ls=dash, color=C_KIDNEY, marker="o",
           mfc="white", mec=C_KIDNEY, lw=1.4, markersize=4.2, clip_on=False, zorder=3)
    a.plot(xp, amp("tumor_95width_rel"), "-",  color=C_TUMOR, marker="s",
           mfc=C_TUMOR, mec=C_TUMOR, **common)
    a.plot(xp, tac("tumor_95width_rel"), ls=dash, color=C_TUMOR, marker="s",
           mfc="white", mec=C_TUMOR, lw=1.4, markersize=4.2, clip_on=False, zorder=3)

    a.set_ylim(0.30, 1.75)
    a.set_yticks([0.4, 0.8, 1.2, 1.6])
    a.set_ylabel("Per-patient 95% interval width\n(fraction of mean dose)")
    # direct labels at the right endpoints (amplitude curves)
    a.text(1.06, amp("tumor_95width_rel")[-1], "Tumor",  color=C_TUMOR,
           fontsize=8, fontweight="bold", va="center", ha="left")
    a.text(1.06, amp("kidney_95width_rel")[-1], "Kidney", color=C_KIDNEY,
           fontsize=8, fontweight="bold", va="center", ha="left")
    _style_key(a, "upper left")
    a.set_title("Per-patient uncertainty width", fontsize=8.8, pad=6, loc="left")
    _panel_label(a, "a")

    # ---------------- Panel b: standard-schedule feasibility -----------------
    b = ax[1]
    b.axvline(0, color=C_GRID, lw=0.9, ls=(0, (2, 2)), zorder=1)
    ge = amp("pct_ge95_feasible")
    b.plot(xp, ge, "-", color=C_KIDNEY, marker="o",
           mfc=C_KIDNEY, mec=C_KIDNEY, **common)
    b.plot(xp, tac("pct_ge95_feasible"), ls=dash, color=C_KIDNEY, marker="o",
           mfc="white", mec=C_KIDNEY, lw=1.4, markersize=4.2, clip_on=False, zorder=3)
    b.plot(xp, amp("mean_feasibility_pct"), "-", color=C_MEAN, marker="^",
           mfc=C_MEAN, mec=C_MEAN, **common)
    b.plot(xp, tac("mean_feasibility_pct"), ls=dash, color=C_MEAN, marker="^",
           mfc="white", mec=C_MEAN, lw=1.4, markersize=4.2, clip_on=False, zorder=3)

    b.set_ylim(36, 100)
    b.set_yticks([40, 55, 70, 85, 100])
    b.set_ylabel("Standard 44.4 GBq schedule\nfeasibility (%)")
    # annotate the strict >=95% line (the metric that moves)
    for x, v, dy, va in zip(xp, ge, (8, 8, -10), ("bottom", "bottom", "top")):
        b.annotate(f"{v:.1f}", (x, v), textcoords="offset points",
                   xytext=(0, dy), ha="center", va=va, fontsize=7, color=C_KIDNEY)
    b.text(1.06, amp("pct_ge95_feasible")[-1], "≥95%\nfeasible", color=C_KIDNEY,
           fontsize=7.6, fontweight="bold", va="center", ha="left", linespacing=0.95)
    b.text(1.06, amp("mean_feasibility_pct")[-1], "mean\nfeasibility", color=C_MEAN,
           fontsize=7.6, fontweight="bold", va="center", ha="left", linespacing=0.95)
    _style_key(b, "lower left")
    b.set_title("Standard-schedule feasibility", fontsize=8.8, pad=6, loc="left")
    _panel_label(b, "b")

    for axx in ax:
        axx.set_xlim(-1.35, 1.55)
        axx.set_xticks(xp)
        axx.set_xticklabels(["0.5×", "1×\n(baseline)", "2×"])
        axx.set_xlabel("Monte Carlo prior-width multiplier (σ)")
        axx.tick_params(length=3)

    Path("figures").mkdir(exist_ok=True)
    stem = "figures/figS1_sigma_sensitivity"
    for ext, dpi in (("png", 600), ("pdf", 600), ("svg", 600)):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_config(name, sigmas, cohort):
    """Return a one-row summary of interval widths + feasibility for one config."""
    k_w, t_w, k_rel, t_rel, a_w = [], [], [], [], []
    feas, k_med, t_med = [], [], []
    for i in range(len(cohort)):
        row = cohort.iloc[i].to_dict()
        rng = np.random.default_rng([BASE_SEED, i])
        dose = compute_patient_dose(row, REF_ACTIVITY_GBQ, rng, N_MC,
                                    return_samples=True, sigmas=sigmas)
        renal = max_constrained_activity_from_samples(
            dose["kidney_samples"], REF_ACTIVITY_GBQ)

        kw = dose["kidney"]["p975"] - dose["kidney"]["p025"]
        tw = dose["tumor"]["p975"]  - dose["tumor"]["p025"]
        k_w.append(kw);  t_w.append(tw)
        k_rel.append(kw / dose["kidney"]["mean"])
        t_rel.append(tw / dose["tumor"]["mean"])
        a_w.append(renal["max_cumulative_activity_gbq"]["p975"]
                   - renal["max_cumulative_activity_gbq"]["p025"])
        feas.append(renal["standard_feasibility_pct"])
        k_med.append(dose["kidney"]["mean"]); t_med.append(dose["tumor"]["mean"])

    feas = np.asarray(feas)
    return {
        "config":               name,
        "kidney_med_dose":      round(float(np.median(k_med)), 3),
        "tumor_med_dose":       round(float(np.median(t_med)), 3),
        "kidney_95width_Gy":    round(float(np.median(k_w)),   3),
        "kidney_95width_rel":   round(float(np.median(k_rel)), 3),
        "tumor_95width_Gy":     round(float(np.median(t_w)),   3),
        "tumor_95width_rel":    round(float(np.median(t_rel)), 3),
        "maxact_95width_GBq":   round(float(np.median(a_w)),   2),
        "mean_feasibility_pct": round(float(np.mean(feas)),    1),
        "pct_ge95_feasible":    round(float(np.mean(feas >= 95) * 100), 1),
    }


def run_sweep():
    cohort = pd.read_csv("results/cohort.csv")
    rows = [run_config(n, s, cohort) for n, s in CONFIGS.items()]
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/sigma_sensitivity.csv", index=False)
    return df


if __name__ == "__main__":
    if "--figure" in sys.argv:                      # rebuild figure only
        df = pd.read_csv("results/sigma_sensitivity.csv")
    else:
        df = run_sweep()
        print(df.to_string(index=False))
    make_figure(df)
    print("\nSaved -> results/sigma_sensitivity.csv, figures/figS1_sigma_sensitivity.{png,pdf,svg}")
