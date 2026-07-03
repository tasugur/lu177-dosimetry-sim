"""
Surrogate XGBoost models that approximate the Monte Carlo dose engine.
PSA and TL-PSMA are log-transformed before training.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

OUT_FIG = Path("figures")
OUT_MOD = Path("results")

FEATURES = ["gfr_ml_min", "weight_kg", "bsa_m2",
            "log_psa", "log_tl_psma", "activity_gbq"]

FEATURE_LABELS = ["GFR (mL/min)", "Body Weight (kg)", "BSA (m2)",
                  "log(PSA+1)", "log(TL-PSMA index+1)",
                  "Administered Activity (GBq)"]

# Bone marrow is excluded here (surrogate only reaches ~R2=0.5); the calculator
# uses a deterministic marrow estimate instead.
TARGETS = {
    "kidney_dose":       "Kidney Absorbed Dose (Gy/cycle)",
    "tumor_dose":        "Whole-Body Tumor Absorbed Dose (Gy/cycle)",
    "therapeutic_ratio": "Tumor-to-Kidney Therapeutic Ratio",
    "max_activity":      "Renal-Constrained Max Cumulative Activity (GBq)",
    "parotid_dose":      "Parotid Gland Absorbed Dose (Gy/cycle)",
}

COL_MAP = {
    "kidney_dose_mean":       "kidney_dose_mean_gy",
    "kidney_dose_p025":       "kidney_dose_p025_gy",
    "kidney_dose_p975":       "kidney_dose_p975_gy",
    "tumor_dose_mean":        "tumor_dose_mean_gy",
    "tumor_dose_p025":        "tumor_dose_p025_gy",
    "tumor_dose_p975":        "tumor_dose_p975_gy",
    "therapeutic_ratio_mean": "therapeutic_ratio_mean",
    "therapeutic_ratio_p025": "therapeutic_ratio_p025",
    "therapeutic_ratio_p975": "therapeutic_ratio_p975",
    "max_activity_mean":      "max_activity_gbq_mean",
    "max_activity_p025":      "max_activity_gbq_p025",
    "max_activity_p975":      "max_activity_gbq_p975",
    "parotid_dose_mean":      "parotid_dose_mean_gy",
    "parotid_dose_p025":      "parotid_dose_p025_gy",
    "parotid_dose_p975":      "parotid_dose_p975_gy",
}


def prepare_features(results):
    results = results.copy()
    results["log_psa"]     = np.log1p(results["psa_ng_ml"])
    results["log_tl_psma"] = np.log1p(results["tl_psma_index"])
    return results


def train_surrogate(results):
    results = prepare_features(results)
    X = results[FEATURES]
    groups = results["patient_id"].to_numpy() if "patient_id" in results.columns else None
    models  = {}
    metrics = []

    for target, label in TARGETS.items():
        for stat in ["mean", "p025", "p975"]:
            col = COL_MAP[f"{target}_{stat}"]
            if col not in results.columns:
                print(f"  Skipping {col} (not in results)")
                continue

            y = results[col]

            if groups is None:
                raise KeyError("results must contain patient_id for grouped validation")
            # Group split on patient_id so all activity rows for a patient
            # land on the same side — otherwise we leak across activities.
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            groups_tr = groups[train_idx]

            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            r2  = r2_score(y_te, y_pred)
            mae = mean_absolute_error(y_te, y_pred)

            gkf = GroupKFold(n_splits=5)
            cv_r2 = cross_val_score(
                xgb.XGBRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1,
                ),
                X_tr, y_tr,
                groups=groups_tr,
                cv=gkf, scoring="r2"
            )
            cv_mean = float(cv_r2.mean())
            cv_std  = float(cv_r2.std())

            key = f"{target}_{stat}"
            models[key] = model
            metrics.append({
                "Target":     f"{label} [{stat}]",
                "R2_test":    round(r2,      4),
                "MAE_test":   round(mae,     4),
                "CV_R2_mean": round(cv_mean, 4),
                "CV_R2_std":  round(cv_std,  4),
            })
            print(f"  {label} [{stat}]: "
                  f"test R2={r2:.4f}  "
                  f"CV(train) R2={cv_mean:.4f} (+/-{cv_std:.4f})")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_MOD / "surrogate_metrics.csv", index=False)
    return models, metrics_df


def plot_shap(models, results):
    results = prepare_features(results)
    X = results[FEATURES]

    mean_models = {k: v for k, v in models.items() if k.endswith("_mean")}

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "Figure 7. SHAP Feature Importance for Surrogate Model Predictions\n"
        "(Mean |SHAP value| across 2,000 sampled virtual patients)",
        fontsize=13, fontweight="bold"
    )

    # 3 panels on top, 2 centered on the bottom
    top_axes    = [fig.add_subplot(2, 3, i+1) for i in range(3)]
    bottom_axes = [fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 6)]
    all_axes = top_axes + bottom_axes

    for ax, (key, model) in zip(all_axes, mean_models.items()):
        label  = TARGETS[key.replace("_mean", "")]
        sample = X.sample(2000, random_state=42)
        # Using XGBoost's native pred_contribs because shap.TreeExplainer
        # chokes on the base_score format from XGBoost >= 2.x.
        booster  = model.get_booster()
        contribs = booster.predict(xgb.DMatrix(sample), pred_contribs=True)
        sv       = contribs[:, :-1]
        mean_abs = np.abs(sv).mean(axis=0)
        order    = np.argsort(mean_abs)

        ax.barh([FEATURE_LABELS[i] for i in order],
                mean_abs[order],
                color="#2166ac", alpha=0.8)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(label)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(OUT_FIG / "fig7_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Figure 7 (SHAP) saved")


def save_models(models):
    joblib.dump(models, OUT_MOD / "surrogate_models.joblib")
    print("  Surrogate models saved -> results/surrogate_models.joblib")


if __name__ == "__main__":
    print("Training surrogate models...\n")
    results = pd.read_csv("results/simulation_results.csv")
    models, metrics = train_surrogate(results)
    print("\nGenerating SHAP importance plot...")
    plot_shap(models, results)
    save_models(models)
    print("\nDone.")
    print(metrics.to_string(index=False))
