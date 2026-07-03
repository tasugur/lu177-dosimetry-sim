"""
Structural one-at-a-time (OAT) sensitivity analysis.

Re-runs the full cohort with each structural parameter (Hill K and gamma, tumor
sink, GFR exponent band, kidney/tumor slow half-lives) set low and high, and
reports the cohort medians, the GFR and therapeutic-ratio gradients, and the
feasibility metrics. Medians stay stable; the covariate gradients move.

Outputs:
  results/structural_sensitivity.csv
  figures/figS2_structural_sensitivity.{png,pdf,svg}

Run:
  python -m src.structural_sensitivity                 # full sweep + figure
  python -m src.structural_sensitivity baseline K_x2   # selected configs (append)
  python -m src.structural_sensitivity --figure        # rebuild figure from CSV
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from src.dosimetry import compute_patient_dose
from src.renal_constraint import max_constrained_activity_from_samples

REF_ACTIVITY_GBQ = 7.4
N_MC             = 400          # medians/gradients are stable; 400 keeps it fast
BASE_SEED        = 42

CONFIGS = {
    "baseline":          {},
    "K_x0.5":            {"k_tl": 150.0},
    "K_x2":              {"k_tl": 600.0},
    "gamma_0.6":         {"gamma": 0.6},
    "gamma_1.0":         {"gamma": 1.0},
    "sink_x0.5":         {"sink_coef": 0.04},
    "sink_x2":           {"sink_coef": 0.16},
    "alpha_lo":          {"alpha_lo": 0.2, "alpha_hi": 0.5},
    "alpha_hi":          {"alpha_lo": 0.5, "alpha_hi": 0.8},
    "kid_Tslow_x0.75":   {"tslow_scale_kidney": 0.75},
    "kid_Tslow_x1.25":   {"tslow_scale_kidney": 1.25},
    "tum_Tslow_x0.75":   {"tslow_scale_tumor": 0.75},
    "tum_Tslow_x1.25":   {"tslow_scale_tumor": 1.25},
}

# (label, low-config, high-config) for the tornado
PARAMS = [
    ("Hill K (TL-PSMA half-saturation)", "K_x0.5", "K_x2"),
    ("Hill γ (saturation steepness)",    "gamma_0.6", "gamma_1.0"),
    ("Tumor-burden sink coefficient",    "sink_x0.5", "sink_x2"),
    ("GFR clearance exponent band α",    "alpha_lo", "alpha_hi"),
    ("Kidney slow half-life",            "kid_Tslow_x0.75", "kid_Tslow_x1.25"),
    ("Tumor slow half-life",             "tum_Tslow_x0.75", "tum_Tslow_x1.25"),
]


def run_config(name, struct, cohort):
    gfr_strat = cohort["gfr_stratum"].to_numpy()
    tl_strat  = cohort["tl_psma_stratum"].to_numpy()
    km, tm, trm, feas = [], [], [], []
    for i in range(len(cohort)):
        row = cohort.iloc[i].to_dict()
        rng = np.random.default_rng([BASE_SEED, i])
        d = compute_patient_dose(row, REF_ACTIVITY_GBQ, rng, N_MC,
                                 return_samples=True, structural=struct)
        renal = max_constrained_activity_from_samples(d["kidney_samples"], REF_ACTIVITY_GBQ)
        km.append(d["kidney"]["mean"]); tm.append(d["tumor"]["mean"])
        trm.append(d["therapeutic_ratio"]["mean"]); feas.append(renal["standard_feasibility_pct"])
    km, tm, trm, feas = map(np.asarray, (km, tm, trm, feas))
    mw = lambda a, m: float(np.median(a[m]))
    kid_imp  = mw(km, gfr_strat == "impaired");  kid_pres = mw(km, gfr_strat == "preserved")
    tr_low   = mw(trm, tl_strat == "low");        tr_vh    = mw(trm, tl_strat == "very_high")
    return {
        "config":       name,
        "kidney_med":   round(float(np.median(km)),  3),
        "tumor_med":    round(float(np.median(tm)),  3),
        "tr_med":       round(float(np.median(trm)), 3),
        "kidney_imp":   round(kid_imp,  3),
        "kidney_pres":  round(kid_pres, 3),
        "gfr_gradient": round(kid_imp / kid_pres, 3),
        "tr_low":       round(tr_low, 3),
        "tr_vhigh":     round(tr_vh,  3),
        "tr_gradient":  round(tr_vh / tr_low, 2),
        "mean_feas":    round(float(np.mean(feas)),        1),
        "pct_ge95":     round(float(np.mean(feas >= 95) * 100), 1),
    }


# Tornado figure. Alpha encodes the swing magnitude; only the movers are labelled.
C_BAR  = "#3775BA"   # bars
C_BASE = "#767676"   # baseline marker
C_VAL  = "#4D4D4D"   # value labels

# Fixed display order (top -> bottom), shared by both panels for comparability.
DISPLAY = [
    ("Hill γ (saturation steepness)",    "gamma_0.6",       "gamma_1.0"),
    ("Hill K (TL-PSMA half-saturation)", "K_x0.5",          "K_x2"),
    ("GFR clearance exponent band α",    "alpha_lo",        "alpha_hi"),
    ("Tumor-burden sink coefficient",    "sink_x0.5",       "sink_x2"),
    ("Kidney slow half-life",            "kid_Tslow_x0.75", "kid_Tslow_x1.25"),
    ("Tumor slow half-life",             "tum_Tslow_x0.75", "tum_Tslow_x1.25"),
]


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 1.0, "legend.frameon": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })


def _tornado(ax, d, metric, baseline, xlabel, title, tag, label_y=True):
    n = len(DISPLAY)
    spans  = [(d.loc[lo, metric], d.loc[hi, metric]) for _, lo, hi in DISPLAY]
    swings = [abs(b - a) for a, b in spans]
    mx = max(swings) or 1.0
    ypos = np.arange(n)[::-1]                       # DISPLAY[0] at top
    for yi, (lo, hi), sw in zip(ypos, spans, swings):
        x0, x1 = min(lo, hi), max(lo, hi)
        alpha = 0.32 + 0.68 * (sw / mx)            # alpha ~ swing size
        ax.barh(yi, x1 - x0, left=x0, height=0.60, color=C_BAR, alpha=alpha,
                edgecolor="white", linewidth=0.7, zorder=3)
        if sw / mx < 0.05:                          # near-zero swing -> visible marker
            ax.plot((x0 + x1) / 2.0, yi, "s", ms=2.6, color=C_BAR,
                    alpha=max(alpha, 0.55), zorder=3)
        if sw / mx > 0.22:                          # annotate only the movers
            ax.text(x0, yi, f"{x0:.1f} ", ha="right", va="center",
                    fontsize=6.6, color=C_VAL, zorder=4)
            ax.text(x1, yi, f" {x1:.1f}", ha="left", va="center",
                    fontsize=6.6, color=C_VAL, zorder=4)
    ax.axvline(baseline, color=C_BASE, lw=1.0, ls=(0, (3, 2)), zorder=2)
    ax.text(baseline, n - 0.42, f"baseline {baseline:.1f}", color=C_BASE,
            fontsize=6.8, ha="center", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in DISPLAY] if label_y else ["" for _ in DISPLAY],
                       fontsize=7.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    allv = [v for s in spans for v in s] + [baseline]
    pad = (max(allv) - min(allv)) * 0.20 or 0.1
    ax.set_xlim(min(allv) - pad, max(allv) + pad)
    ax.set_ylim(-0.6, n - 0.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8.6, loc="left", pad=8)
    ax.text(-0.02, 1.08, tag, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="right", va="bottom")
    ax.tick_params(axis="x", length=3)


def make_figure(df):
    _style()
    d = df.drop_duplicates("config").set_index("config")
    fig, ax = plt.subplots(1, 2, figsize=(180 / 25.4, 76 / 25.4))
    fig.subplots_adjust(left=0.265, right=0.985, bottom=0.24, top=0.82, wspace=0.30)
    _tornado(ax[0], d, "tr_gradient", d.loc["baseline", "tr_gradient"],
             "Therapeutic-ratio gradient\n(very-high ÷ low TL-PSMA)",
             "Tumor burden → therapeutic ratio (§3.4)", "a", label_y=True)
    _tornado(ax[1], d, "gfr_gradient", d.loc["baseline", "gfr_gradient"],
             "Kidney-dose gradient\n(impaired ÷ preserved GFR)",
             "Renal function → kidney dose (§3.3)", "b", label_y=False)
    Path("figures").mkdir(exist_ok=True)
    stem = "figures/figS2_structural_sensitivity"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    csv = "results/structural_sensitivity.csv"
    if "--figure" in sys.argv:
        make_figure(pd.read_csv(csv))
        print("figure rebuilt from", csv)
    else:
        cohort = pd.read_csv("results/cohort.csv")
        names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CONFIGS)
        rows = [run_config(n, CONFIGS[n], cohort) for n in names]
        out = pd.DataFrame(rows)
        hdr = not os.path.exists(csv)
        out.to_csv(csv, mode="a", header=hdr, index=False)
        print(out.to_string(index=False))
        if set(CONFIGS).issubset(set(pd.read_csv(csv)["config"])):
            full = pd.read_csv(csv).drop_duplicates("config").set_index("config").loc[list(CONFIGS)].reset_index()
            make_figure(full)
            print("\nFigure -> figures/figS2_structural_sensitivity.png")
