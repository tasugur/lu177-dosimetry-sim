"""
Streamlit front-end for the 177Lu-PSMA-617 dosimetry simulator.

Two modes:
  - Fixed activity: every cycle at the same GBq.
  - Per-cycle:      each cycle can have its own activity (and GFR/PSA/TL-PSMA),
                    cumulative renal dose tracked across cycles.

Research use only.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="177Lu-PSMA-617 Dosimetry Simulator",
    page_icon=None,
    layout="wide",
)


@st.cache_resource
def load_models():
    return joblib.load(
        Path(__file__).parent.parent / "results" / "surrogate_models.joblib"
    )


@st.cache_resource
def load_anchors():
    with open(Path(__file__).parent.parent / "data" / "literature_anchors.json", "r") as f:
        return json.load(f)


models  = load_models()
anchors = load_anchors()

# Input ranges — must match the training distribution
GFR_MIN,    GFR_MAX    =  30.0, 130.0
WEIGHT_MIN, WEIGHT_MAX =  50.0, 140.0
HEIGHT_MIN, HEIGHT_MAX = 155.0, 200.0
PSA_MIN,    PSA_MAX    =   2.0, 5000.0
TL_MIN,     TL_MAX     =   5.0, 5000.0
ACT_MIN,    ACT_MAX    =   5.0,   9.0
RENAL_LIMIT            =  23.0   # Gy


def predict(gfr, weight, bsa, psa, tl_psma, activity_gbq):
    """Run all 15 surrogate models + the deterministic marrow estimate."""
    inp = pd.DataFrame([{
        "gfr_ml_min":   gfr,
        "weight_kg":    weight,
        "bsa_m2":       bsa,
        "log_psa":      np.log1p(psa),
        "log_tl_psma":  np.log1p(tl_psma),
        "activity_gbq": activity_gbq,
    }])

    # Bone marrow: closed-form physical-dose reference (0.11 Gy/GBq x activity x
    # BSA scaling), not a surrogate. The 95% interval uses the engine's combined
    # log-normal sigma (per-patient 0.20 (+) MC 0.10).
    DC_BM_REF = 0.11
    BSA_REF   = 1.95
    f_bsa     = (BSA_REF / bsa) ** 0.5
    m_mean    = DC_BM_REF * activity_gbq * f_bsa
    SIGMA_BM  = (0.20 ** 2 + 0.10 ** 2) ** 0.5   # engine per-patient (+) MC

    return {
        "k_mean": float(models["kidney_dose_mean"].predict(inp)[0]),
        "k_low":  float(models["kidney_dose_p025"].predict(inp)[0]),
        "k_high": float(models["kidney_dose_p975"].predict(inp)[0]),
        "t_mean": float(models["tumor_dose_mean"].predict(inp)[0]),
        "t_low":  float(models["tumor_dose_p025"].predict(inp)[0]),
        "t_high": float(models["tumor_dose_p975"].predict(inp)[0]),
        "r_mean": float(models["therapeutic_ratio_mean"].predict(inp)[0]),
        "r_low":  float(models["therapeutic_ratio_p025"].predict(inp)[0]),
        "r_high": float(models["therapeutic_ratio_p975"].predict(inp)[0]),
        "a_mean": float(models["max_activity_mean"].predict(inp)[0]),
        "a_low":  float(models["max_activity_p025"].predict(inp)[0]),
        "a_high": float(models["max_activity_p975"].predict(inp)[0]),
        "p_mean": float(models["parotid_dose_mean"].predict(inp)[0]),
        "p_low":  float(models["parotid_dose_p025"].predict(inp)[0]),
        "p_high": float(models["parotid_dose_p975"].predict(inp)[0]),
        "m_mean": m_mean,
        # Log-normal 95% interval from the engine's combined sigma (mean-preserving)
        "m_low":  m_mean * float(np.exp(-1.96 * SIGMA_BM)),
        "m_high": m_mean * float(np.exp( 1.96 * SIGMA_BM)),
    }


st.title("[¹⁷⁷Lu]Lu-PSMA-617 Dosimetry Simulation")
st.markdown("""
**Uncertainty-aware absorbed dose simulation · Research use only**

This tool is a hypothesis-generating simulator based on a literature-calibrated
synthetic cohort study (10,000 virtual mCRPC patients). Outputs are simulation
estimates, not clinically validated predictions, and should not inform individual
administered-activity decisions in the absence of post-therapy image-based
dosimetry. Inputs outside the training distribution (GFR < 30 mL/min,
TL-PSMA index > 5,000, activity < 5.0 or > 9.0 GBq) are extrapolations.

*Primary calibration: Violet et al. J Nucl Med 2019;60:517–523.*
""")

st.divider()

st.subheader("Patient Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    gfr = st.number_input(
        "Glomerular Filtration Rate (mL/min)",
        min_value=GFR_MIN, max_value=GFR_MAX,
        value=72.0, step=1.0,
        help="eGFR or measured GFR. Training range: 30-130 mL/min.",
    )
    weight = st.number_input(
        "Body Weight (kg)",
        min_value=WEIGHT_MIN, max_value=WEIGHT_MAX,
        value=80.0, step=0.5,
    )

with col2:
    height = st.number_input(
        "Height (cm)",
        min_value=HEIGHT_MIN, max_value=HEIGHT_MAX,
        value=175.0, step=0.5,
        help="Used to calculate BSA via the DuBois formula.",
    )

with col3:
    psa = st.number_input(
        "Serum PSA (ng/mL)",
        min_value=PSA_MIN, max_value=PSA_MAX,
        value=80.0, step=1.0,
    )
    tl_psma = st.number_input(
        "TL-PSMA index (SUVmean x lesion volume)",
        min_value=TL_MIN, max_value=TL_MAX,
        value=300.0, step=10.0,
        help="Sum of (SUVmean x lesion volume) across all PSMA-avid lesions.",
    )

bsa = 0.007184 * (height ** 0.725) * (weight ** 0.425)
st.caption(f"Calculated BSA (DuBois formula): **{bsa:.3f} m²**")

st.divider()

st.subheader("Dosimetry Mode")

mode = st.radio(
    "Select activity input mode:",
    options=["Fixed activity (all cycles equal)",
             "Per-cycle activity (each cycle individually)"],
    horizontal=True,
)

st.divider()


if mode == "Fixed activity (all cycles equal)":

    activity_gbq = st.slider(
        "Administered activity per cycle (GBq)",
        min_value=ACT_MIN, max_value=ACT_MAX, value=7.4, step=0.1,
        help="Training range: 5.0-9.0 GBq.",
    )
    st.caption(
        f"Selected: **{activity_gbq} GBq/cycle** "
        f"({'standard' if activity_gbq == 7.4 else 'non-standard'} dose)."
    )

    if st.button("Calculate Dose Estimates", type="primary",
                 use_container_width=True):

        p = predict(gfr, weight, bsa, psa, tl_psma, activity_gbq)
        k_mean, k_low, k_high    = p["k_mean"], p["k_low"], p["k_high"]
        t_mean, t_low, t_high    = p["t_mean"], p["t_low"], p["t_high"]
        r_mean, r_low, r_high    = p["r_mean"], p["r_low"], p["r_high"]
        a_mean, a_low, a_high    = p["a_mean"], p["a_low"], p["a_high"]
        pa_mean, pa_low, pa_high = p["p_mean"], p["p_low"], p["p_high"]
        m_mean, m_low, m_high    = p["m_mean"], p["m_low"], p["m_high"]

        cycle_cross  = RENAL_LIMIT / k_mean
        chart_cycles = max(6, int(np.ceil(cycle_cross)))

        st.subheader(f"Dose Estimates per Cycle ({activity_gbq} GBq 177Lu-PSMA-617)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kidney absorbed dose", f"{k_mean:.2f} Gy",
                  help=f"95% MC: {k_low:.2f} - {k_high:.2f} Gy")
        c2.metric("Whole-body tumor dose", f"{t_mean:.1f} Gy",
                  help=f"95% MC: {t_low:.1f} - {t_high:.1f} Gy")
        c3.metric("Tumor-to-kidney ratio", f"{r_mean:.1f}",
                  help=f"95% MC: {r_low:.1f} - {r_high:.1f}")
        c4.metric("Renal-constrained max activity", f"{a_mean:.1f} GBq",
                  help=f"95% MC: {a_low:.1f} - {a_high:.1f} GBq")

        c5, c6, _, _ = st.columns(4)
        c5.metric("Parotid gland dose", f"{pa_mean:.2f} Gy",
                  help=f"95% MC: {pa_low:.2f} - {pa_high:.2f} Gy. "
                       "Salivary toxicity risk increases above 40 Gy cumulative.")
        c6.metric("Bone marrow dose", f"{m_mean:.3f} Gy",
                  help=f"95% interval: {m_low:.3f} – {m_high:.3f} Gy. "
                       "Reference physical-dose estimate (not a surrogate output): "
                       "0.11 Gy/GBq × activity × BSA scaling; interval from the "
                       "engine's per-patient (σ=0.20) ⊕ Monte Carlo (σ=0.10) terms.")

        st.divider()
        st.subheader("Uncertainty Intervals (95% Monte Carlo)")

        violet_k = anchors["dose_coefficients"]["kidney"] * 7.4
        violet_t = (anchors["whole_body_tumor_dose_gy"]["median"]
                    / anchors["mean_administered_activity_gbq"] * 7.4)
        violet_p = anchors["dose_coefficients"]["parotid"] * 7.4
        violet_m = anchors["dose_coefficients"]["bone_marrow"] * 7.4

        st.table(pd.DataFrame({
            "Parameter": [
                "Kidney absorbed dose (Gy/cycle)",
                "Whole-body tumor dose (Gy/cycle)",
                "Tumor-to-kidney ratio",
                "Renal-constrained max activity (GBq)",
                "Parotid gland dose (Gy/cycle)",
                "Bone marrow dose (Gy/cycle) *",
            ],
            "Mean": [f"{k_mean:.2f}", f"{t_mean:.1f}",
                     f"{r_mean:.1f}", f"{a_mean:.1f}",
                     f"{pa_mean:.2f}", f"{m_mean:.3f}"],
            "95% MC lower": [f"{k_low:.2f}", f"{t_low:.1f}",
                             f"{r_low:.1f}", f"{a_low:.1f}",
                             f"{pa_low:.2f}", f"{m_low:.3f}"],
            "95% MC upper": [f"{k_high:.2f}", f"{t_high:.1f}",
                             f"{r_high:.1f}", f"{a_high:.1f}",
                             f"{pa_high:.2f}", f"{m_high:.3f}"],
            "Violet et al. anchor": [
                f"{violet_k:.2f} Gy/cycle",
                f"{violet_t:.2f} Gy/cycle",
                "—", "—",
                f"{violet_p:.2f} Gy/cycle",
                f"{violet_m:.3f} Gy/cycle"],
        }))
        st.caption(
            "\\* Bone marrow dose: reference physical-dose estimate "
            "(0.11 Gy/GBq × activity × BSA scaling), reported separately from "
            "the XGBoost surrogate outputs. Chemotherapy history is not a model "
            "input. The 95% interval is derived from the simulation engine's "
            "per-patient (σ=0.20) and Monte Carlo (σ=0.10) log-normal residual "
            "terms (combined σ≈0.22), not a fixed coefficient of variation."
        )

        st.divider()
        st.subheader("Projected Cumulative Renal Dose")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Kidney dose per cycle (mean) | **{k_mean:.2f} Gy** |
| Kidney dose per cycle (95% MC) | **{k_low:.2f} - {k_high:.2f} Gy** |
| Renal dose threshold | **{RENAL_LIMIT:.0f} Gy** |
| Cycles to reach {RENAL_LIMIT:.0f} Gy (mean) | **{cycle_cross:.1f}** |
| Cycles to reach {RENAL_LIMIT:.0f} Gy (lower bound) | **{RENAL_LIMIT/k_low:.1f}** |
| Cycles to reach {RENAL_LIMIT:.0f} Gy (upper bound) | **{RENAL_LIMIT/k_high:.1f}** |
| Renal-constrained max activity (mean) | **{a_mean:.1f} GBq** |
| Renal-constrained max activity (95% MC) | **{a_low:.1f} - {a_high:.1f} GBq** |
""")

        with col_b:
            fig, ax = plt.subplots(figsize=(6, 4))
            cycles   = list(range(1, chart_cycles + 1))
            cum_mean = [k_mean * c for c in cycles]
            cum_low  = [k_low  * c for c in cycles]
            cum_high = [k_high * c for c in cycles]

            ax.bar(cycles, cum_mean, color="#2166ac", alpha=0.8,
                   label="Mean estimate")
            ax.fill_between(cycles, cum_low, cum_high,
                            alpha=0.25, color="#2166ac",
                            label="95% MC uncertainty")
            ax.axhline(RENAL_LIMIT, color="red", lw=2, linestyle="--",
                       label=f"{RENAL_LIMIT:.0f} Gy renal threshold")
            ax.axvline(cycle_cross, color="orange", lw=1.5, linestyle=":",
                       label=f"Threshold at cycle {cycle_cross:.1f} (mean)")
            ax.set_xlabel("Treatment cycle")
            ax.set_ylabel("Cumulative kidney dose (Gy)")
            ax.set_title("Projected cumulative renal dose")
            ax.set_xticks(cycles)
            ax.legend(fontsize=8)
            sns.despine(ax=ax)
            st.pyplot(fig, use_container_width=True)
            plt.close()

else:
    st.markdown(
        "Enter parameters for each cycle. "
        "**GFR, weight, PSA and TL-PSMA can be updated each cycle** "
        "to reflect changes in renal function and disease burden. "
        "Height and BSA are carried forward from the baseline values above."
    )

    if "cycles" not in st.session_state:
        st.session_state.cycles = [{
            "activity": 7.4,
            "gfr":      gfr,
            "weight":   weight,
            "psa":      psa,
            "tl_psma":  tl_psma,
        }]

    col_add, col_rem, _ = st.columns([1, 1, 4])
    with col_add:
        if st.button("+ Add cycle"):
            last = st.session_state.cycles[-1]
            st.session_state.cycles.append({
                "activity": last["activity"],
                "gfr":      last["gfr"],
                "weight":   last["weight"],
                "psa":      last["psa"],
                "tl_psma":  last["tl_psma"],
            })
            st.rerun()
    with col_rem:
        if st.button("- Remove last") and len(st.session_state.cycles) > 1:
            st.session_state.cycles.pop()
            st.rerun()

    st.divider()

    n_cycles = len(st.session_state.cycles)

    for idx in range(n_cycles):
        cyc = st.session_state.cycles[idx]
        with st.expander(f"Cycle {idx + 1}", expanded=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                cyc["activity"] = st.number_input(
                    "Activity (GBq)",
                    min_value=ACT_MIN, max_value=ACT_MAX,
                    value=float(cyc["activity"]), step=0.1,
                    key=f"act_{idx}",
                )
            with c2:
                cyc["gfr"] = st.number_input(
                    "GFR (mL/min)",
                    min_value=GFR_MIN, max_value=GFR_MAX,
                    value=float(cyc["gfr"]), step=1.0,
                    key=f"gfr_{idx}",
                )
            with c3:
                cyc["weight"] = st.number_input(
                    "Weight (kg)",
                    min_value=WEIGHT_MIN, max_value=WEIGHT_MAX,
                    value=float(cyc["weight"]), step=0.5,
                    key=f"wt_{idx}",
                )
            with c4:
                cyc["psa"] = st.number_input(
                    "PSA (ng/mL)",
                    min_value=PSA_MIN, max_value=PSA_MAX,
                    value=float(cyc["psa"]), step=1.0,
                    key=f"psa_{idx}",
                )
            with c5:
                cyc["tl_psma"] = st.number_input(
                    "TL-PSMA index",
                    min_value=TL_MIN, max_value=TL_MAX,
                    value=float(cyc["tl_psma"]), step=10.0,
                    key=f"tl_{idx}",
                )
        st.session_state.cycles[idx] = cyc

    st.divider()

    if st.button("Calculate Cumulative Dose", type="primary",
                 use_container_width=True):

        cycle_data = []
        cum_k_mean = cum_k_low = cum_k_high = 0.0
        cum_t_mean = 0.0
        cum_p_mean = 0.0
        cum_m_mean = 0.0

        for cyc_idx, cyc in enumerate(st.session_state.cycles, start=1):
            cyc_bsa = 0.007184 * (height ** 0.725) * (cyc["weight"] ** 0.425)

            p = predict(cyc["gfr"], cyc["weight"], cyc_bsa,
                        cyc["psa"], cyc["tl_psma"], cyc["activity"])

            cum_k_mean += p["k_mean"]
            cum_k_low  += p["k_low"]
            cum_k_high += p["k_high"]
            cum_t_mean += p["t_mean"]
            cum_p_mean += p["p_mean"]
            cum_m_mean += p["m_mean"]

            remaining_mean = RENAL_LIMIT - cum_k_mean

            cycle_data.append({
                "Cycle":              cyc_idx,
                "Activity (GBq)":     cyc["activity"],
                "GFR":                cyc["gfr"],
                "Kidney (Gy)":        round(p["k_mean"], 2),
                "Kidney 95% MC":      f"{p['k_low']:.2f} – {p['k_high']:.2f}",
                "Tumor (Gy)":         round(p["t_mean"], 1),
                "Tumor 95% MC":       f"{p['t_low']:.1f} – {p['t_high']:.1f}",
                "Parotid (Gy)":       round(p["p_mean"], 2),
                "Parotid 95% MC":     f"{p['p_low']:.2f} – {p['p_high']:.2f}",
                "Bone marrow (Gy)":   round(p["m_mean"], 3),
                "Cumul. kidney (Gy)": round(cum_k_mean, 2),
                "Cumul. 95% MC":      f"{cum_k_low:.2f} – {cum_k_high:.2f}",
                "Remaining (Gy)":     round(max(0, remaining_mean), 2),
                "Status": (
                    "LIMIT REACHED" if cum_k_mean >= RENAL_LIMIT
                    else "WARNING"   if remaining_mean < p["k_mean"]
                    else "OK"
                ),
            })

        df_cycles = pd.DataFrame(cycle_data)

        st.subheader("Cumulative Dosimetry Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total cycles", f"{n_cycles}")
        m2.metric("Total activity",
                  f"{sum(c['activity'] for c in st.session_state.cycles):.1f} GBq")
        m3.metric("Cumulative kidney dose (mean)",
                  f"{cum_k_mean:.2f} Gy",
                  delta=f"{cum_k_mean - RENAL_LIMIT:.2f} Gy vs limit",
                  delta_color="inverse")
        m4.metric("Remaining renal capacity (mean)",
                  f"{max(0, RENAL_LIMIT - cum_k_mean):.2f} Gy")

        m5, m6, m7, _ = st.columns(4)
        m5.metric("Cumulative tumor dose (mean)", f"{cum_t_mean:.1f} Gy")
        m6.metric("Cumulative parotid dose (mean)", f"{cum_p_mean:.2f} Gy",
                  help="Salivary toxicity risk increases above 40 Gy cumulative.")
        m7.metric("Cumulative bone marrow dose (mean)", f"{cum_m_mean:.3f} Gy",
                  help="Reference physical-dose estimate (0.11 Gy/GBq × activity "
                       "× BSA scaling); not a surrogate output.")

        st.divider()
        st.subheader("Per-Cycle Breakdown")

        def color_status(val):
            if val == "LIMIT REACHED":
                return "background-color: #ffcccc"
            if val == "WARNING":
                return "background-color: #fff3cd"
            return ""

        st.dataframe(
            df_cycles.style.map(color_status, subset=["Status"]),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Cumulative Renal Dose Across Cycles")

        cycle_nums  = df_cycles["Cycle"].tolist()
        cum_k_means = df_cycles["Cumul. kidney (Gy)"].tolist()

        cum_low_list  = []
        cum_high_list = []
        cl = ch = 0.0
        for cyc in st.session_state.cycles:
            cyc_bsa = 0.007184 * (height ** 0.725) * (cyc["weight"] ** 0.425)
            p = predict(cyc["gfr"], cyc["weight"], cyc_bsa,
                        cyc["psa"], cyc["tl_psma"], cyc["activity"])
            cl += p["k_low"]
            ch += p["k_high"]
            cum_low_list.append(cl)
            cum_high_list.append(ch)

        bar_colors = [
            "#d73027" if v >= RENAL_LIMIT
            else "#fdae61" if (RENAL_LIMIT - v) < st.session_state.cycles[i]["activity"] * 0.35
            else "#4393c3"
            for i, v in enumerate(cum_k_means)
        ]

        fig, ax = plt.subplots(
            figsize=(min(10, max(5, n_cycles * 0.9 + 2)), 3.5))
        bar_width = min(0.6, 0.8 - n_cycles * 0.01)

        ax.bar(cycle_nums, cum_k_means, color=bar_colors,
               alpha=0.88, width=bar_width,
               label="Cumulative kidney dose (mean)")
        ax.fill_between(cycle_nums, cum_low_list, cum_high_list,
                        alpha=0.15, color="#2166ac",
                        label="95% MC band")
        ax.axhline(RENAL_LIMIT, color="red", lw=1.8, linestyle="--",
                   label=f"{RENAL_LIMIT:.0f} Gy threshold")

        if n_cycles <= 12:
            for c, v, cyc in zip(cycle_nums, cum_k_means, st.session_state.cycles):
                ax.text(c, v + 0.3, f"{cyc['activity']} GBq",
                        ha="center", fontsize=7, color="#333333")

        ax.set_xlabel("Cycle", fontsize=10)
        ax.set_ylabel("Cumulative kidney dose (Gy)", fontsize=10)
        ax.set_title("Cumulative renal dose — per-cycle plan", fontsize=11)
        ax.set_xticks(cycle_nums)
        ax.set_ylim(0, max(RENAL_LIMIT * 1.1, max(cum_high_list) * 1.15))
        ax.legend(fontsize=8, loc="upper left")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        remaining = RENAL_LIMIT - cum_k_mean
        last_cyc  = st.session_state.cycles[-1]
        last_bsa  = 0.007184 * (height ** 0.725) * (last_cyc["weight"] ** 0.425)
        last_p    = predict(last_cyc["gfr"], last_cyc["weight"], last_bsa,
                            last_cyc["psa"], last_cyc["tl_psma"],
                            last_cyc["activity"])

        if remaining <= 0:
            st.error(
                f"Renal dose threshold of {RENAL_LIMIT:.0f} Gy has been "
                f"reached (cumulative mean: {cum_k_mean:.2f} Gy, "
                f"95% MC: {cum_k_low:.2f}–{cum_k_high:.2f} Gy). "
                "No further cycles are estimated to be feasible."
            )
        else:
            est_more      = remaining / last_p["k_mean"]
            est_more_low  = (RENAL_LIMIT - cum_k_high) / last_p["k_high"] if (RENAL_LIMIT - cum_k_high) > 0 else 0
            est_more_high = (RENAL_LIMIT - cum_k_low)  / last_p["k_low"]
            st.info(
                f"Estimated remaining renal capacity: "
                f"**{remaining:.2f} Gy** (mean) · "
                f"95% MC: **{max(0, RENAL_LIMIT - cum_k_high):.2f}–{RENAL_LIMIT - cum_k_low:.2f} Gy**. "
                f"At last-cycle parameters ({last_cyc['activity']} GBq, "
                f"GFR {last_cyc['gfr']:.0f}, TL-PSMA {last_cyc['tl_psma']:.0f}), "
                f"approximately **{est_more:.1f}** additional cycles (mean) · "
                f"**{est_more_low:.1f}–{est_more_high:.1f}** (95% MC range)."
            )

        st.divider()
        st.caption(
            f"BSA recalculated each cycle from updated weight and baseline height. "
            f"Bone marrow dose: reference physical-dose estimate (0.11 Gy/GBq × activity × BSA scaling); "
            f"reported separately from the surrogate outputs."
        )


st.divider()

col_f1, col_f2 = st.columns([3, 2])

with col_f1:
    st.caption(
        "**Model:** XGBoost surrogate (15 models) trained on 50,000 patient–activity "
        "combinations (10,000 virtual mCRPC patients × 5 activity levels: "
        "5.0, 6.0, 7.4, 8.0, 9.0 GBq/cycle). Balanced 2D stratified design "
        "(GFR × TL-PSMA, 16 cells × 625 patients). "
        "Surrogate inputs: GFR, weight, BSA, PSA, TL-PSMA, and activity. "
        "Monte Carlo uncertainty: 1,000 samples per patient per activity level. "
        "**Primary calibration:** Violet et al. (J Nucl Med 2019;60:517–523). "
        "**Dosimetry literature references:** Fendler et al. (Oncotarget 2017), "
        "Kurth et al. (Cancers 2021), Takano et al. (Ann Nucl Med 2025), "
        "Jackson et al. (Semin Nucl Med 2022)."
    )

with col_f2:
    st.info(
        "**External Validation**\n\n"
        "This tool has not yet been externally validated against independent "
        "patient-level dosimetry data. Researchers interested in external "
        "validation or collaborative model evaluation are welcome to get in touch.\n\n"
        "Contact: ugurtasdr@gmail.com"
    )
