"""
Prespecified population weights for a sensitivity-analysis re-summary.

The primary cohort is balanced (16 strata x 625). These weights are NOT
empirical prevalence — they are clinically plausible assumptions used
to recompute cohort-level summaries under a more realistic distribution
of GFR and TL-PSMA. GFR and TL-PSMA are treated as independent, so each
cell weight = GFR weight x TL-PSMA weight.
"""

import numpy as np
import pandas as pd


GFR_WEIGHTS = {
    "impaired":         0.15,
    "mild_to_moderate": 0.25,
    "mildly_reduced":   0.30,
    "preserved":        0.30,
}

TL_PSMA_WEIGHTS = {
    "low":       0.10,
    "moderate":  0.30,
    "high":      0.40,
    "very_high": 0.20,
}


def cell_weights_table() -> pd.DataFrame:
    """Return the 16 cell weights as a long-format DataFrame (sums to 1)."""
    rows = []
    for g_lab, g_w in GFR_WEIGHTS.items():
        for t_lab, t_w in TL_PSMA_WEIGHTS.items():
            rows.append({
                "gfr_stratum":     g_lab,
                "tl_psma_stratum": t_lab,
                "cell_weight":     g_w * t_w,
            })
    df = pd.DataFrame(rows)
    total = df["cell_weight"].sum()
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"Cell weights sum to {total:.10f}, expected 1.0")
    return df


def patient_weights(results: pd.DataFrame) -> np.ndarray:
    """Per-patient weight vector aligned with results (also sums to 1)."""
    if "gfr_stratum" not in results.columns \
       or "tl_psma_stratum" not in results.columns:
        raise KeyError("results must contain gfr_stratum and tl_psma_stratum")

    cells  = cell_weights_table().set_index(["gfr_stratum", "tl_psma_stratum"])
    counts = results.groupby(["gfr_stratum", "tl_psma_stratum"]).size()

    cell_w  = cells["cell_weight"].reindex(counts.index)
    per_pat = (cell_w / counts).rename("patient_weight")
    weights = results.merge(
        per_pat.reset_index(),
        on=["gfr_stratum", "tl_psma_stratum"],
        how="left",
    )["patient_weight"].to_numpy()

    total = weights.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Patient weights sum to {total:.10f}, expected 1.0")
    return weights


def weighted_quantile(values, weights, q):
    values  = np.asarray(values,  dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~np.isnan(values)
    values  = values[mask]
    weights = weights[mask]

    order  = np.argsort(values)
    v_sort = values[order]
    w_sort = weights[order]
    cum_w  = np.cumsum(w_sort) / w_sort.sum()
    return float(np.interp(q, cum_w, v_sort))


def weighted_summary(values, weights):
    values  = np.asarray(values,  dtype=float)
    weights = np.asarray(weights, dtype=float)
    return {
        "mean":   float(np.average(values, weights=weights)),
        "median": weighted_quantile(values, weights, 0.50),
        "p025":   weighted_quantile(values, weights, 0.025),
        "p975":   weighted_quantile(values, weights, 0.975),
    }


def weighted_proportion(values, weights, threshold, op=">="):
    """Weighted % of values satisfying the threshold."""
    values  = np.asarray(values,  dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = {
        ">=": values >= threshold,
        ">":  values >  threshold,
        "<=": values <= threshold,
        "<":  values <  threshold,
    }[op]
    return float(weights[mask].sum() / weights.sum() * 100)


def summarize_outputs(results: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side unweighted vs. weighted summary for the main outputs."""
    w = patient_weights(results)

    targets = {
        "kidney_dose_mean_gy":          "Kidney dose (Gy/cycle)",
        "tumor_dose_mean_gy":           "Tumor dose (Gy/cycle)",
        "parotid_dose_mean_gy":         "Parotid dose (Gy/cycle)",
        "marrow_physical_dose_mean_gy": "Marrow physical dose (Gy/cycle)",
        "therapeutic_ratio_mean":       "Therapeutic ratio",
        "max_activity_gbq_mean":        "Renal-constrained max activity (GBq)",
        "standard_feasibility_pct":     "Standard schedule feasibility (%)",
    }

    rows = []
    for col, label in targets.items():
        if col not in results.columns:
            continue
        v   = results[col].to_numpy()
        unw = {
            "median": float(np.median(v)),
            "p025":   float(np.percentile(v, 2.5)),
            "p975":   float(np.percentile(v, 97.5)),
        }
        wsum = weighted_summary(v, w)

        rows.append({
            "Output":             label,
            "Unweighted median":  round(unw["median"],  3),
            "Unweighted p2.5":    round(unw["p025"],    3),
            "Unweighted p97.5":   round(unw["p975"],    3),
            "Weighted median":    round(wsum["median"], 3),
            "Weighted p2.5":      round(wsum["p025"],   3),
            "Weighted p97.5":     round(wsum["p975"],   3),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)

    print("Cell weights table (rows = GFR, cols = TL-PSMA):")
    cw = cell_weights_table()
    print(cw.pivot(index="gfr_stratum",
                   columns="tl_psma_stratum",
                   values="cell_weight").round(4))
    print(f"\nSum of cell weights: {cw['cell_weight'].sum():.6f}")

    results_path = Path("results/simulation_results.csv")
    if not results_path.exists():
        print("\n(No simulation_results.csv yet — run src.simulate first)")
    else:
        results = pd.read_csv(results_path)

        if "activity_gbq" in results.columns:
            results_ref = results[results["activity_gbq"] == 7.4].copy()
            print(f"Multi-activity dataset detected. "
                  f"Filtering to 7.4 GBq reference rows "
                  f"({len(results_ref):,} of {len(results):,} rows).\n")
        else:
            results_ref = results

        print("\nUnweighted vs. weighted summary:")
        summary = summarize_outputs(results_ref)
        print(summary.to_string(index=False))
        summary.to_csv("results/weighted_summary.csv", index=False)
        print("\nSaved -> results/weighted_summary.csv")

        w    = patient_weights(results_ref)
        feas = results_ref["standard_feasibility_pct"].to_numpy()

        unw_mean     = float(np.mean(feas))
        unw_pct_ge95 = float(np.mean(feas >= 95) * 100)
        w_mean       = float(np.average(feas, weights=w))
        w_pct_ge95   = weighted_proportion(feas, w, 95.0, ">=")

        print("\nFeasibility metrics:")
        print(f"  Unweighted mean MC feasibility:      {unw_mean:.2f}%")
        print(f"  Unweighted % patients >=95% feas:    {unw_pct_ge95:.2f}%")
        print(f"  Weighted mean MC feasibility:        {w_mean:.2f}%")
        print(f"  Weighted % patients >=95% feas:      {w_pct_ge95:.2f}%")
