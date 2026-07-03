"""
Check on why bone marrow is reported as a closed-form reference, not a surrogate.

The marrow mean carries a per-patient log-normal residual (sigma ~ 0.20) that is
independent of the six covariates, so a surrogate can't learn it: XGBoost reaches
only R2 ~ 0.5 on marrow vs ~1.0 on kidney, and recovers R2 ~ 1.0 once the residual
is removed.

Run:
  python -m src.marrow_surrogate_check
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

RANDOM_SEED = 42
REF_ACTIVITY_GBQ = 7.4

FEATURES = ["gfr_ml_min", "weight_kg", "bsa_m2",
            "log_psa", "log_tl_psma", "activity_gbq"]

XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=RANDOM_SEED, n_jobs=-1)


def _prepare(df):
    df = df.copy()
    df["log_psa"] = np.log1p(df["psa_ng_ml"])
    df["log_tl_psma"] = np.log1p(df["tl_psma_index"])
    return df


def _grouped_test_r2(X, y, groups):
    """Patient-grouped 80/20 split, same protocol as src/surrogate.py."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                 random_state=RANDOM_SEED)
    tr, te = next(splitter.split(X, y, groups=groups))
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X.iloc[tr], y.iloc[tr])
    return r2_score(y.iloc[te], model.predict(X.iloc[te]))


def run_check(results: pd.DataFrame) -> pd.DataFrame:
    results = _prepare(results)
    X = results[FEATURES]
    groups = results["patient_id"].to_numpy()

    # (1) fit on marrow mean (has the residual) vs kidney mean (deterministic).
    r2_marrow = _grouped_test_r2(X, results["marrow_physical_dose_mean_gy"], groups)
    r2_kidney = _grouped_test_r2(X, results["kidney_dose_mean_gy"], groups)

    # (2) recover the residual: marrow mean is deterministic in BSA only, so
    #     regress log-dose on log-BSA and inspect the leftover.
    ref = results[results["activity_gbq"] == REF_ACTIVITY_GBQ].copy()
    log_dose = np.log(ref["marrow_physical_dose_mean_gy"].to_numpy())
    log_bsa = np.log(ref["bsa_m2"].to_numpy())
    design = np.vstack([np.ones_like(log_bsa), log_bsa]).T
    coef, *_ = np.linalg.lstsq(design, log_dose, rcond=None)
    residual = log_dose - design @ coef
    resid_sd = float(residual.std())
    corrs = {f: abs(float(np.corrcoef(residual, ref[f].to_numpy())[0, 1]))
             for f in ["gfr_ml_min", "weight_kg", "bsa_m2",
                       "psa_ng_ml", "tl_psma_index"]}

    # (3) refit on the deterministic signal only; the surrogate recovers it.
    ref["marrow_deterministic"] = np.exp(design @ coef)
    r2_no_resid = _grouped_test_r2(_prepare(ref)[FEATURES],
                                   ref["marrow_deterministic"],
                                   ref["patient_id"].to_numpy())

    rows = [
        {"quantity": "R2 surrogate on marrow mean (with residual)", "value": round(r2_marrow, 4), "manuscript": "~0.48-0.52"},
        {"quantity": "R2 surrogate on kidney mean (contrast)",      "value": round(r2_kidney, 4), "manuscript": ">0.99"},
        {"quantity": "log-residual SD",                             "value": round(resid_sd, 4), "manuscript": "~0.20"},
        {"quantity": "max |corr| residual vs feature",             "value": round(max(corrs.values()), 4), "manuscript": "<=0.02"},
        {"quantity": "R2 surrogate on marrow with residual removed","value": round(r2_no_resid, 4), "manuscript": "~1.0"},
    ]
    summary = pd.DataFrame(rows)

    print("\nMarrow surrogate check")
    print(summary.to_string(index=False))
    print("\nPer-feature |corr| with the unexplained residual:")
    for f, c in corrs.items():
        print(f"  {f:15s} {c:.4f}")
    return summary


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    results = pd.read_csv("results/simulation_results.csv")
    summary = run_check(results)
    out = Path("results/marrow_surrogate_check.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
