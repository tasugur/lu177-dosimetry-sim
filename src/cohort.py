"""
Synthetic mCRPC cohort generator for the 177Lu-PSMA-617 parametric in silico simulation.

2D stratified sampling on (GFR x TL-PSMA): 4 x 4 = 16 cells, 625 patients each,
total 10,000. The balanced design is for risk-space coverage, not to mirror
real-world prevalence — apply weighting.py post-hoc if you want that.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import truncnorm

RANDOM_SEED = 42

GFR_STRATA = [
    (30.0,  45.0),
    (45.0,  60.0),
    (60.0,  75.0),
    (75.0, 130.0),
]
GFR_LABELS = ["impaired", "mild_to_moderate", "mildly_reduced", "preserved"]

TL_PSMA_STRATA = [
    (5.0,    100.0),
    (100.0,  300.0),
    (300.0, 1000.0),
    (1000.0, 5000.0),
]
TL_PSMA_LABELS = ["low", "moderate", "high", "very_high"]

PATIENTS_PER_STRATUM = 625


def _truncated_normal(mean, sd, low, high, n, rng):
    a = (low - mean) / sd
    b = (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=n,
                         random_state=int(rng.integers(0, 1_000_000)))


def _log_uniform(low, high, n, rng):
    return np.exp(rng.uniform(np.log(low), np.log(high), n))


def _sample_stratum(gfr_low, gfr_high, tl_low, tl_high, n, rng):
    gfr = rng.uniform(gfr_low, gfr_high, n)
    tl_psma = _log_uniform(tl_low, tl_high, n, rng)

    age    = _truncated_normal(70, 8,  45,  90,  n, rng)
    weight = _truncated_normal(80, 14, 50,  140, n, rng)
    height = (175 - 0.05 * (weight - 80) + rng.normal(0, 5, n)).clip(155, 200)

    # DuBois BSA
    bsa = 0.007184 * (height ** 0.725) * (weight ** 0.425)

    prior_chemo = rng.binomial(1, p=0.60, size=n)

    # PSA log-normal, partially correlated with TL-PSMA inside the stratum
    log_tl_residual = np.log(tl_psma) - np.log(np.sqrt(tl_low * tl_high))
    log_psa = np.log(80) + 0.4 * log_tl_residual + rng.normal(0, 1.0, n)
    psa     = np.exp(log_psa).clip(2, 5000)

    return pd.DataFrame({
        "age_years":     age.round(1),
        "weight_kg":     weight.round(1),
        "height_cm":     height.round(1),
        "bsa_m2":        bsa.round(3),
        "gfr_ml_min":    gfr.round(1),
        "prior_chemo":   prior_chemo,
        "psa_ng_ml":     psa.round(1),
        "tl_psma_index": tl_psma.round(1),
    })


def generate_cohort(patients_per_stratum: int = PATIENTS_PER_STRATUM) -> pd.DataFrame:
    rng    = np.random.default_rng(RANDOM_SEED)
    pieces = []

    for (g_lo, g_hi), g_lab in zip(GFR_STRATA, GFR_LABELS):
        for (t_lo, t_hi), t_lab in zip(TL_PSMA_STRATA, TL_PSMA_LABELS):
            df_cell = _sample_stratum(g_lo, g_hi, t_lo, t_hi,
                                      patients_per_stratum, rng)
            df_cell["gfr_stratum"]     = g_lab
            df_cell["tl_psma_stratum"] = t_lab
            pieces.append(df_cell)

    df = pd.concat(pieces, ignore_index=True)
    df.insert(0, "patient_id", np.arange(1, len(df) + 1))
    return df


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)

    cohort = generate_cohort()
    print(f"Total cohort size: {len(cohort):,} patients")
    print(f"  Strata: {cohort['gfr_stratum'].nunique()} GFR x "
          f"{cohort['tl_psma_stratum'].nunique()} TL-PSMA "
          f"= {cohort.groupby(['gfr_stratum','tl_psma_stratum']).ngroups} cells")
    print(f"  Patients per cell: "
          f"{cohort.groupby(['gfr_stratum','tl_psma_stratum']).size().min()} - "
          f"{cohort.groupby(['gfr_stratum','tl_psma_stratum']).size().max()}")
    print()
    print("Stratum distribution:")
    print(cohort.groupby(["gfr_stratum", "tl_psma_stratum"]).size()
                .unstack(fill_value=0))
    print()
    print("Continuous variable summary:")
    cols = ["age_years", "weight_kg", "bsa_m2", "gfr_ml_min",
            "psa_ng_ml", "tl_psma_index"]
    print(cohort[cols].describe().round(2))

    cohort.to_csv("results/cohort.csv", index=False)
    print(f"\nSaved -> results/cohort.csv  ({len(cohort):,} patients)")
