"""
Bi-exponential TAC + MIRD dose calculation for 177Lu-PSMA-617.

    A(t)    = A0 * [f_fast * exp(-lam_fast * t) + f_slow * exp(-lam_slow * t)]
    A_tilde = A0 * [f_fast / lam_fast + f_slow / lam_slow]
    D       = A_tilde * S    [Gy]

S-values are calibrated against Violet et al. (J Nucl Med 2019) for a
reference patient (GFR=72 mL/min, weight=70 kg, TL-PSMA=300).
Tumor anchor: 11.55 Gy / 7.8 GBq = 1.4808 Gy/GBq → 10.958 Gy at 7.4 GBq.
"""

import numpy as np
import pandas as pd

# 177Lu physical half-life (hours)
LU177_T_PHYS_H = 159.5
LAMBDA_PHYS    = np.log(2) / LU177_T_PHYS_H

GFR_REF    = 72.0
WEIGHT_REF = 70.0
A0_MBq_REF = 7400.0   # 7.4 GBq

# Violet et al. absorbed-dose anchors for the reference patient (Gy at 7.4 GBq)
VIOLET_TARGET_GY = {
    "kidney":        0.39  * 7.4,
    "tumor":         (11.55 / 7.8) * 7.4,
    "parotid":       0.58  * 7.4,
    "submandibular": 0.44  * 7.4,
    "bone_marrow":   0.11  * 7.4,
}

# Bi-exponential TAC parameters (Kurth/Takano for fast phase, Violet/Kurth for slow)
ORGAN_BIEXP = {
    "kidney":        {"f_fast": 0.35, "T_bio_fast": 1.5, "T_bio_slow":  38.0},
    "tumor":         {"f_fast": 0.15, "T_bio_fast": 8.0, "T_bio_slow": 100.0},
    "parotid":       {"f_fast": 0.55, "T_bio_fast": 1.0, "T_bio_slow":  18.0},
    "submandibular": {"f_fast": 0.50, "T_bio_fast": 1.2, "T_bio_slow":  20.0},
    "bone_marrow":   {"f_fast": 0.25, "T_bio_fast": 2.0, "T_bio_slow":  90.0},
}

# ICRP reference organ masses (g, 70 kg adult male)
REF_MASS_G = {
    "kidney":        300.0,
    "parotid":        58.0,
    "submandibular":  29.0,
    "bone_marrow":  1170.0,
}

# 1-sigma log-normal prior widths for the Monte Carlo uncertainty layer.
# amp_* = amplitude scatter, tac_* = slow-phase half-life, bm_* = bone marrow.
SIGMA_DEFAULTS = {
    "amp_kidney": 0.12, "amp_tumor": 0.20, "amp_saliv": 0.15,
    "tac_kidney": 0.10, "tac_tumor": 0.12, "tac_saliv": 0.10,
    "bm_patient": 0.20, "bm_mc": 0.10,
}

# Structural parameters for the deterministic skeleton (Hill saturation, tumor
# sink, GFR exponent band, slow-phase half-lives). Varied by the OAT sweep.
STRUCT_DEFAULTS = {
    "k_tl": 300.0, "gamma": 0.8, "sink_coef": 0.08,
    "alpha_lo": 0.3, "alpha_hi": 0.7,
    "tslow_scale_kidney": 1.0, "tslow_scale_tumor": 1.0,
}


def biexp_A_tilde(A0, f_fast, lam_fast, lam_slow):
    """Analytical cumulated activity [MBq.h]. lam_* may be arrays."""
    f_slow = 1.0 - f_fast
    return A0 * (f_fast / lam_fast + f_slow / lam_slow)


def _calibrate_S_values():
    """S = target_dose / A_tilde_ref, computed at the reference patient."""
    S = {}
    for organ, p in ORGAN_BIEXP.items():
        if organ not in VIOLET_TARGET_GY:
            continue
        lam_fast = LAMBDA_PHYS + np.log(2) / p["T_bio_fast"]
        lam_slow = LAMBDA_PHYS + np.log(2) / p["T_bio_slow"]
        A_ref = biexp_A_tilde(A0_MBq_REF, p["f_fast"], lam_fast, lam_slow)
        S[organ] = float(VIOLET_TARGET_GY[organ] / A_ref)
    return S


S_CAL = _calibrate_S_values()


def _S_for(organ, tslow_scale=1.0):
    """Reference S-value for one organ, recalibrated to Violet if T_slow is scaled.
    Returns exactly S_CAL[organ] when tslow_scale == 1.0."""
    p = ORGAN_BIEXP[organ]
    lam_fast = LAMBDA_PHYS + np.log(2) / p["T_bio_fast"]
    lam_slow = LAMBDA_PHYS + np.log(2) / (p["T_bio_slow"] * tslow_scale)
    A_ref = biexp_A_tilde(A0_MBq_REF, p["f_fast"], lam_fast, lam_slow)
    return float(VIOLET_TARGET_GY[organ] / A_ref)


def hill_function(tl_psma, k_tl=300.0, gamma=0.8, tl_ref=300.0):
    """Bounded Hill scaling for TL-PSMA, normalised to F_TL=1 at tl_ref."""
    num = tl_psma ** gamma / (tl_psma ** gamma + k_tl ** gamma)
    den = tl_ref  ** gamma / (tl_ref  ** gamma + k_tl ** gamma)
    return num / den


def compute_patient_dose(row, a_administered_gbq, rng,
                         n_mc=1000, return_samples=False, sigmas=None,
                         structural=None):
    """
    Monte Carlo bi-exponential TAC-MIRD dose for one patient.

    Bone marrow returns a physical absorbed dose only (no toxicity model).
    """
    gfr    = float(row["gfr_ml_min"])
    weight = float(row["weight_kg"])
    bsa    = float(row["bsa_m2"])
    tl     = float(row["tl_psma_index"])
    psa    = float(row["psa_ng_ml"])
    a0     = a_administered_gbq * 1000.0   # GBq -> MBq

    # Mean-preserving log-normal amplitude noise (the -0.5*sigma^2 term keeps
    # the median fixed). Tumor sigma is larger to cover PSMA heterogeneity.
    s  = SIGMA_DEFAULTS if sigmas is None else {**SIGMA_DEFAULTS, **sigmas}
    st = STRUCT_DEFAULTS if structural is None else {**STRUCT_DEFAULTS, **structural}
    SIGMA_K, SIGMA_T, SIGMA_S = s["amp_kidney"], s["amp_tumor"], s["amp_saliv"]
    alpha_s = rng.uniform(st["alpha_lo"], st["alpha_hi"], n_mc)
    eta_k   = rng.normal(-0.5 * SIGMA_K ** 2, SIGMA_K, n_mc)
    eta_t   = rng.normal(-0.5 * SIGMA_T ** 2, SIGMA_T, n_mc)
    eta_s   = rng.normal(-0.5 * SIGMA_S ** 2, SIGMA_S, n_mc)

    # Slow-phase half-life sampled log-normally so the intervals also capture
    # TAC shape uncertainty, not just amplitude.
    SIGMA_TAC_K, SIGMA_TAC_T, SIGMA_TAC_S = s["tac_kidney"], s["tac_tumor"], s["tac_saliv"]

    T_slow_k  = (ORGAN_BIEXP["kidney"]["T_bio_slow"] * st["tslow_scale_kidney"]
                 * np.exp(rng.normal(-0.5 * SIGMA_TAC_K**2, SIGMA_TAC_K, n_mc)))
    T_slow_t  = (ORGAN_BIEXP["tumor"]["T_bio_slow"] * st["tslow_scale_tumor"]
                 * np.exp(rng.normal(-0.5 * SIGMA_TAC_T**2, SIGMA_TAC_T, n_mc)))
    T_slow_p  = (ORGAN_BIEXP["parotid"]["T_bio_slow"]
                 * np.exp(rng.normal(-0.5 * SIGMA_TAC_S**2, SIGMA_TAC_S, n_mc)))
    T_slow_sm = (ORGAN_BIEXP["submandibular"]["T_bio_slow"]
                 * np.exp(rng.normal(-0.5 * SIGMA_TAC_S**2, SIGMA_TAC_S, n_mc)))

    # Bone marrow noise: per-patient shift + per-MC jitter, both mean-preserving.
    SIGMA_PATIENT_BM, SIGMA_MC_BM = s["bm_patient"], s["bm_mc"]
    patient_shift_bm = rng.normal(-0.5 * SIGMA_PATIENT_BM ** 2, SIGMA_PATIENT_BM)
    eta_bm           = rng.normal(-0.5 * SIGMA_MC_BM ** 2, SIGMA_MC_BM, n_mc)

    # Organ mass scaling
    w_ratio      = weight / 70.0
    kidney_mass  = REF_MASS_G["kidney"]  * w_ratio ** 0.75
    parotid_mass = REF_MASS_G["parotid"] * w_ratio ** 0.30
    bsa_ratio    = bsa / 1.73  # vs. ICRP reference male BSA

    # Covariate modifiers — anchored so they equal 1 at the reference patient
    f_sink = np.clip(
        1.0 - st["sink_coef"] * (np.log1p(tl / 300.0) - np.log1p(1.0)),
        0.6, 1.1
    )
    f_tl  = hill_function(tl, k_tl=st["k_tl"], gamma=st["gamma"])
    f_psa = np.clip(1.0 + 0.05 * np.log(psa / 80.0), 0.8, 1.2)

    # Kidney
    pk = ORGAN_BIEXP["kidney"]
    lam_fast_k     = LAMBDA_PHYS + np.log(2) / pk["T_bio_fast"]
    lam_bio_slow_k = (np.log(2) / T_slow_k) * (gfr / GFR_REF) ** alpha_s
    lam_slow_k     = LAMBDA_PHYS + lam_bio_slow_k

    A_tilde_k   = biexp_A_tilde(a0, pk["f_fast"], lam_fast_k, lam_slow_k)
    S_kidney    = (_S_for("kidney", st["tslow_scale_kidney"])
                   * (REF_MASS_G["kidney"] / kidney_mass)
                   * f_sink
                   * np.exp(eta_k))
    dose_kidney = A_tilde_k * S_kidney

    # Tumor
    pt = ORGAN_BIEXP["tumor"]
    lam_fast_t = LAMBDA_PHYS + np.log(2) / pt["T_bio_fast"]
    lam_slow_t = LAMBDA_PHYS + np.log(2) / T_slow_t
    A_tilde_t  = biexp_A_tilde(a0, pt["f_fast"], lam_fast_t, lam_slow_t)
    dose_tumor = A_tilde_t * f_tl * f_psa * _S_for("tumor", st["tslow_scale_tumor"]) * np.exp(eta_t)

    # Parotid
    pp = ORGAN_BIEXP["parotid"]
    lam_fast_p = LAMBDA_PHYS + np.log(2) / pp["T_bio_fast"]
    lam_slow_p = LAMBDA_PHYS + np.log(2) / T_slow_p
    A_tilde_p  = biexp_A_tilde(a0, pp["f_fast"], lam_fast_p, lam_slow_p)
    S_parotid  = (S_CAL["parotid"]
                  * (REF_MASS_G["parotid"] / parotid_mass)
                  * f_sink
                  * (1.0 / bsa_ratio) ** 0.3
                  * np.exp(eta_s))
    dose_parotid = A_tilde_p * S_parotid

    # Submandibular
    psm = ORGAN_BIEXP["submandibular"]
    lam_fast_sm = LAMBDA_PHYS + np.log(2) / psm["T_bio_fast"]
    lam_slow_sm = LAMBDA_PHYS + np.log(2) / T_slow_sm
    A_tilde_sm = biexp_A_tilde(a0, psm["f_fast"], lam_fast_sm, lam_slow_sm)
    S_submand  = (S_CAL["submandibular"]
                  * f_sink
                  * (1.0 / bsa_ratio) ** 0.3
                  * np.exp(eta_s))
    dose_submand = A_tilde_sm * S_submand

    # Bone marrow: BSA-based volume modifier only (no haematologic toxicity
    # model). BSA_REF is the cohort median, so f_bsa = 1 there.
    BSA_REF_BM = 1.95
    f_bsa      = (BSA_REF_BM / bsa) ** 0.5

    pbm = ORGAN_BIEXP["bone_marrow"]
    lam_fast_bm = LAMBDA_PHYS + np.log(2) / pbm["T_bio_fast"]
    lam_slow_bm = LAMBDA_PHYS + np.log(2) / pbm["T_bio_slow"]
    A_tilde_bm  = biexp_A_tilde(a0, pbm["f_fast"], lam_fast_bm, lam_slow_bm)
    dose_marrow_physical = (A_tilde_bm
                            * S_CAL["bone_marrow"]
                            * f_bsa
                            * np.exp(patient_shift_bm + eta_bm))

    ratio = dose_tumor / np.where(dose_kidney > 0, dose_kidney, 1e-9)

    def _s(arr):
        return {
            "mean": float(np.mean(arr)),
            "sd":   float(np.std(arr)),
            "p025": float(np.percentile(arr, 2.5)),
            "p975": float(np.percentile(arr, 97.5)),
        }

    result = {
        "kidney":            _s(dose_kidney),
        "tumor":             _s(dose_tumor),
        "parotid":           _s(dose_parotid),
        "submandibular":     _s(dose_submand),
        "bone_marrow":       _s(dose_marrow_physical),
        "therapeutic_ratio": _s(ratio),
    }
    if return_samples:
        result["kidney_samples"] = dose_kidney
        result["tumor_samples"]  = dose_tumor
    return result
