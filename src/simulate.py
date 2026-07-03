"""
Main simulation driver — runs the full cohort at 5 activity levels and
saves combined results to CSV.

    python -m src.simulate
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.dosimetry import compute_patient_dose
from src.renal_constraint import max_constrained_activity_from_samples

RANDOM_SEED     = 42
N_MC            = 1000
ACTIVITY_LEVELS = [5.0, 6.0, 7.4, 8.0, 9.0]   # GBq per cycle

Path("results").mkdir(exist_ok=True)


def run_simulation(cohort: pd.DataFrame,
                   rng: np.random.Generator) -> pd.DataFrame:
    """
    Run MC dose for every (patient, activity) pair.

    The RNG is re-seeded identically across activity levels for a patient, so
    only the physics changes with activity, not the noise realisation.
    """
    records   = []
    n_total   = len(cohort)
    BASE_SEED = 42

    print(f"  {n_total:,} patients x {len(ACTIVITY_LEVELS)} activity levels "
          f"= {n_total * len(ACTIVITY_LEVELS):,} simulations\n")

    for i, row in enumerate(cohort.itertuples(index=False)):
        if i % 1000 == 0:
            print(f"  Patient {i+1:,} / {n_total:,} ...")

        row_dict = row._asdict()

        for activity_gbq in ACTIVITY_LEVELS:
            patient_rng_act = np.random.default_rng([BASE_SEED, i])

            dose  = compute_patient_dose(row_dict, activity_gbq,
                                         patient_rng_act, N_MC,
                                         return_samples=True)
            renal = max_constrained_activity_from_samples(
                        dose["kidney_samples"], activity_gbq)

            records.append({
                "patient_id":     int(row_dict["patient_id"]),
                "activity_gbq":   activity_gbq,

                "age_years":       row_dict["age_years"],
                "gfr_ml_min":      row_dict["gfr_ml_min"],
                "weight_kg":       row_dict["weight_kg"],
                "bsa_m2":          row_dict["bsa_m2"],
                "prior_chemo":     int(row_dict["prior_chemo"]),
                "psa_ng_ml":       row_dict["psa_ng_ml"],
                "tl_psma_index":   row_dict["tl_psma_index"],
                "gfr_stratum":     row_dict.get("gfr_stratum", ""),
                "tl_psma_stratum": row_dict.get("tl_psma_stratum", ""),

                "kidney_dose_mean_gy":  dose["kidney"]["mean"],
                "kidney_dose_sd_gy":    dose["kidney"]["sd"],
                "kidney_dose_p025_gy":  dose["kidney"]["p025"],
                "kidney_dose_p975_gy":  dose["kidney"]["p975"],

                "tumor_dose_mean_gy":   dose["tumor"]["mean"],
                "tumor_dose_sd_gy":     dose["tumor"]["sd"],
                "tumor_dose_p025_gy":   dose["tumor"]["p025"],
                "tumor_dose_p975_gy":   dose["tumor"]["p975"],

                "parotid_dose_mean_gy": dose["parotid"]["mean"],
                "parotid_dose_p025_gy": dose["parotid"]["p025"],
                "parotid_dose_p975_gy": dose["parotid"]["p975"],

                "submand_dose_mean_gy": dose["submandibular"]["mean"],
                "submand_dose_p025_gy": dose["submandibular"]["p025"],
                "submand_dose_p975_gy": dose["submandibular"]["p975"],

                "marrow_physical_dose_mean_gy": dose["bone_marrow"]["mean"],
                "marrow_physical_dose_p025_gy": dose["bone_marrow"]["p025"],
                "marrow_physical_dose_p975_gy": dose["bone_marrow"]["p975"],

                "therapeutic_ratio_mean": dose["therapeutic_ratio"]["mean"],
                "therapeutic_ratio_p025": dose["therapeutic_ratio"]["p025"],
                "therapeutic_ratio_p975": dose["therapeutic_ratio"]["p975"],

                "max_activity_gbq_mean": renal["max_cumulative_activity_gbq"]["mean"],
                "max_activity_gbq_p025": renal["max_cumulative_activity_gbq"]["p025"],
                "max_activity_gbq_p975": renal["max_cumulative_activity_gbq"]["p975"],

                "standard_feasibility_pct": renal["standard_feasibility_pct"],
                "feasibility_gte95":        int(renal["standard_feasibility_pct"] >= 95.0),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    rng    = np.random.default_rng(RANDOM_SEED)
    cohort = pd.read_csv("results/cohort.csv")

    print(f"Starting multi-activity simulation\n"
          f"  Activity levels: {ACTIVITY_LEVELS} GBq\n"
          f"  MC samples per patient per activity: {N_MC}\n")

    results = run_simulation(cohort, rng)

    out = Path("results/simulation_results.csv")
    results.to_csv(out, index=False)
    print(f"\nDone. Results saved -> {out}")
    print(f"Total rows: {len(results):,}  "
          f"({len(cohort):,} patients x {len(ACTIVITY_LEVELS)} activities)")
    print(results.groupby("activity_gbq")[
        ["kidney_dose_mean_gy", "tumor_dose_mean_gy",
         "therapeutic_ratio_mean", "max_activity_gbq_mean"]
    ].median().round(2))
