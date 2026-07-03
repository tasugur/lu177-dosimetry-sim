"""Manuscript figures for the 177Lu-PSMA-617 simulation study."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
    "legend.fontsize":10,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

COLORS = {
    "kidney":  "#2166ac",
    "tumor":   "#d6604d",
    "parotid": "#4dac26",
    "marrow":  "#8073ac",
    "neutral": "#636363",
}

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Tumor reference dose at 7.4 GBq: 11.55 Gy / 7.8 GBq × 7.4 GBq
TUMOR_REF_GY = (11.55 / 7.8) * 7.4


def load_data():
    cohort  = pd.read_csv("results/cohort.csv")
    results = pd.read_csv("results/simulation_results.csv")
    if "activity_gbq" in results.columns:
        results = results[results["activity_gbq"] == 7.4].copy()
    return cohort, results


def fig1_cohort_characteristics(cohort):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Figure 2. Baseline Characteristics of the Synthetic mCRPC Cohort (n = 10,000)",
        fontsize=13, fontweight="bold", y=0.98
    )
    params = [
        ("age_years",    "Age (years)",                  COLORS["neutral"]),
        ("weight_kg",    "Body Weight (kg)",              COLORS["neutral"]),
        ("bsa_m2",       "Body Surface Area (m²)",        COLORS["neutral"]),
        ("gfr_ml_min",   "GFR (mL/min)",                 COLORS["kidney"]),
        ("psa_ng_ml",    "Serum PSA (ng/mL)",             COLORS["tumor"]),
        ("tl_psma_index","TL-PSMA index (SUVmean x mL)", COLORS["tumor"]),
    ]
    for ax, (col, label, color) in zip(axes.flat[:6], params):
        data = cohort[col]
        if col in ("psa_ng_ml", "tl_psma_index"):
            data  = np.log10(data + 1)
            label = f"log10({label})"
        ax.hist(data, bins=50, color=color, alpha=0.75, edgecolor="white")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Number of Patients", fontsize=10)
        ax.axvline(data.median(), color="black", lw=1.5, linestyle="--",
                   label=f"Median = {data.median():.1f}")
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    ax6    = axes.flat[6]
    counts = cohort["prior_chemo"].value_counts().sort_index()
    ax6.bar(["No Prior\nChemotherapy", "Prior\nChemotherapy"],
            counts.values,
            color=[COLORS["neutral"], COLORS["marrow"]],
            alpha=0.8, width=0.5)
    ax6.set_ylabel("Number of Patients", fontsize=10)
    ax6.set_title("Prior Chemotherapy Status", fontsize=11, pad=8)
    ax6.set_ylim(0, max(counts.values) * 1.18)
    for i, v in enumerate(counts.values):
        ax6.text(i, v + 30, f"{v:,} ({v/len(cohort)*100:.0f}%)",
                 ha="center", fontsize=9)
    sns.despine(ax=ax6)
    axes.flat[7].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=2.0)
    fig.savefig(OUT / "fig2_cohort_characteristics.png")
    plt.close()
    print("  Figure 2 saved")


def fig2_dose_distributions(results):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Figure 3. Simulated Absorbed Dose Distributions per Treatment Cycle (7.4 GBq 177Lu-PSMA-617)",
        fontsize=13, fontweight="bold"
    )
    panels = [
        ("kidney_dose_mean_gy", "Kidney Absorbed Dose (Gy/cycle)",
         COLORS["kidney"], 0.39 * 7.4,
         "Violet et al. (0.39 Gy/GBq x 7.4 GBq)"),
        ("tumor_dose_mean_gy", "Whole-Body Tumor Absorbed Dose (Gy/cycle)",
         COLORS["tumor"], TUMOR_REF_GY,
         f"Violet et al. (11.55/7.8 Gy/GBq x 7.4 GBq = {TUMOR_REF_GY:.2f} Gy)"),
        ("parotid_dose_mean_gy", "Parotid Gland Absorbed Dose (Gy/cycle)",
         COLORS["parotid"], 0.58 * 7.4,
         "Violet et al. (0.58 Gy/GBq x 7.4 GBq)"),
        ("marrow_physical_dose_mean_gy", "Bone Marrow Absorbed Dose (Gy/cycle)",
         COLORS["marrow"], 0.11 * 7.4,
         "Violet et al. (0.11 Gy/GBq x 7.4 GBq)"),
    ]
    for ax, (col, ylabel, color, ref, rlabel) in zip(axes, panels):
        if col not in results.columns:
            ax.set_title(f"[column missing: {col}]")
            continue
        ax.hist(results[col], bins=60, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(ref, color="black", lw=2, linestyle="--", label=rlabel)
        ax.axvline(results[col].median(), color=color, lw=1.5, linestyle="-",
                   label=f"Simulation median = {results[col].median():.2f} Gy")
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Number of Patients")
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(OUT / "fig3_dose_distributions.png")
    plt.close()
    print("  Figure 3 saved")


def fig3_gfr_kidney(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Figure 4. Relationship Between Renal Function and Kidney Absorbed Dose",
        fontsize=13, fontweight="bold"
    )
    ax = axes[0]
    sc = ax.scatter(results["gfr_ml_min"], results["kidney_dose_mean_gy"],
                    c=results["tl_psma_index"], cmap="YlOrRd",
                    alpha=0.15, s=4, rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("TL-PSMA index (SUVmean x mL)", fontsize=10)
    ax.set_xlabel("Glomerular Filtration Rate (mL/min)")
    ax.set_ylabel("Mean Kidney Absorbed Dose per Cycle (Gy)")
    ax.set_title("GFR vs. Kidney Dose (coloured by TL-PSMA index)")
    sns.despine(ax=ax)

    ax2 = axes[1]
    bins   = [30, 45, 60, 75, 130]
    labels = ["30-45\n(Impaired)",
              "45-60\n(Mild-Moderate)",
              "60-75\n(Mildly Reduced)",
              "75-130\n(Preserved)"]
    results = results.copy()
    results["gfr_stratum"] = pd.cut(results["gfr_ml_min"], bins=bins, labels=labels)
    data_by_stratum = [results[results["gfr_stratum"] == l]["kidney_dose_mean_gy"].dropna()
                       for l in labels]
    bp = ax2.boxplot(data_by_stratum, patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"],
                            ["#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]):
        patch.set_facecolor(color)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("GFR Stratum (mL/min)")
    ax2.set_ylabel("Mean Kidney Absorbed Dose per Cycle (Gy)")
    ax2.set_title("Kidney Dose by GFR Stratum")
    ax2.axhline(23 / 6, color="red", lw=1.5, linestyle="--",
                label="Equal-cycle equivalent of 23 Gy / 6 cycles")
    ax2.legend(fontsize=9)
    sns.despine(ax=ax2)
    plt.tight_layout()
    fig.savefig(OUT / "fig4_gfr_kidney_dose.png")
    plt.close()
    print("  Figure 4 saved")


def fig4_therapeutic_ratio(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Figure 5. Tumor-to-Kidney Therapeutic Ratio Distribution",
        fontsize=13, fontweight="bold"
    )
    ax = axes[0]
    ax.hist(results["therapeutic_ratio_mean"], bins=60,
            color=COLORS["tumor"], alpha=0.75, edgecolor="white")
    ax.axvline(results["therapeutic_ratio_mean"].median(), color="black",
               lw=2, linestyle="--",
               label=f"Median = {results['therapeutic_ratio_mean'].median():.1f}")
    ax.set_xlabel("Tumor-to-Kidney Absorbed Dose Ratio")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Distribution of Therapeutic Ratio")
    ax.legend()
    sns.despine(ax=ax)

    ax2 = axes[1]
    sc = ax2.scatter(results["tl_psma_index"], results["therapeutic_ratio_mean"],
                     c=results["gfr_ml_min"], cmap="Blues",
                     alpha=0.15, s=4, rasterized=True)
    cb = plt.colorbar(sc, ax=ax2)
    cb.set_label("GFR (mL/min)", fontsize=10)
    ax2.set_xscale("log")
    ax2.set_xlabel("TL-PSMA index (SUVmean x mL, log scale)")
    ax2.set_ylabel("Tumor-to-Kidney Therapeutic Ratio")
    ax2.set_title("Therapeutic Ratio vs. TL-PSMA index (coloured by GFR)")
    sns.despine(ax=ax2)
    plt.tight_layout()
    fig.savefig(OUT / "fig5_therapeutic_ratio.png")
    plt.close()
    print("  Figure 5 saved")


def fig5_max_activity(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Figure 6. Renal-Constrained Maximum Cumulative Administered Activity (23 Gy Kidney Dose Threshold)",
        fontsize=13, fontweight="bold"
    )
    ax = axes[0]
    ax.hist(results["max_activity_gbq_mean"], bins=40,
            color=COLORS["kidney"], alpha=0.75, edgecolor="white")
    ax.axvline(7.4 * 6, color="red", lw=2, linestyle="--",
               label="Standard 6 x 7.4 GBq = 44.4 GBq")
    ax.set_xlabel("Renal-Constrained Maximum Cumulative Activity (GBq)")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Distribution of Renal-Constrained\nMaximum Cumulative Activity")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    ax2 = axes[1]
    sc = ax2.scatter(results["gfr_ml_min"], results["max_activity_gbq_mean"],
                     c=results["tl_psma_index"], cmap="YlOrRd",
                     alpha=0.15, s=4, rasterized=True)
    cb = plt.colorbar(sc, ax=ax2)
    cb.set_label("TL-PSMA index (SUVmean x mL)", fontsize=10)
    ax2.set_xlabel("GFR (mL/min)")
    ax2.set_ylabel("Renal-Constrained Max Cumulative Activity (GBq)")
    ax2.set_title("Max Activity vs. GFR (coloured by TL-PSMA index)")
    sns.despine(ax=ax2)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_max_constrained_activity.png")
    plt.close()
    print("  Figure 6 saved")


if __name__ == "__main__":
    print("Generating manuscript figures...\n")
    cohort, results = load_data()
    fig1_cohort_characteristics(cohort)
    fig2_dose_distributions(results)
    fig3_gfr_kidney(results)
    fig4_therapeutic_ratio(results)
    fig5_max_activity(results)
    print("\nAll figures saved -> figures/")
