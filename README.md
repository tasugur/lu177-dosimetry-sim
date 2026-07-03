# Simulating [¹⁷⁷Lu]Lu-PSMA-617 Dosimetry from Pre-Therapy Inputs: An Open, Uncertainty-Aware *in silico* Framework for mCRPC

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

> Research use only. Not a clinical tool.

## What this does

This project is a parameter-driven dosimetric simulation framework for [¹⁷⁷Lu]Lu-PSMA-617 radioligand therapy in metastatic castration-resistant prostate cancer (mCRPC). It estimates per-patient absorbed doses from routinely available **pre-therapy inputs** (GFR, body weight, BSA, TL-PSMA index, PSA) without requiring post-therapy imaging-based dosimetry.

The pipeline generates a synthetic cohort of 10,000 virtual mCRPC patients, runs Monte Carlo dosimetry for each one, trains a surrogate model on the results, and serves everything through a Streamlit calculator with per-patient dose estimates and uncertainty intervals.

Unlike image-based dosimetry (MTP/STP), the model does not use patient-specific time-activity measurements. Time-activity curve (TAC) shapes are derived from published population pharmacokinetics (Kurth et al., Takano et al.) and calibrated to reference absorbed doses from Violet et al. Monte Carlo sampling propagates both residual amplitude uncertainty and parametric uncertainty in the assumed TAC structure, so every output carries a quantified uncertainty interval.

## Project structure

```
lu177-dosimetry-sim/
├── data/
│   └── literature_anchors.json
├── src/
│   ├── cohort.py
│   ├── dosimetry.py
│   ├── renal_constraint.py
│   ├── simulate.py
│   ├── weighting.py
│   ├── surrogate.py
│   ├── sensitivity_analysis.py
│   ├── structural_sensitivity.py
│   ├── marrow_surrogate_check.py
│   └── figures.py
├── app/
│   └── calculator.py
├── results/
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

Run these from the project root in order:

```bash
python -m src.cohort
python -m src.simulate
python -m src.weighting
python -m src.surrogate
python -m src.sensitivity_analysis
python -m src.structural_sensitivity
python -m src.marrow_surrogate_check
python -m src.figures
streamlit run app/calculator.py
```

## Outputs

| File | Description |
|------|-------------|
| `results/cohort.csv` | Virtual patient cohort |
| `results/simulation_results.csv` | Per-patient dose estimates |
| `results/weighted_summary.csv` | Weighted cohort summary |
| `results/surrogate_metrics.csv` | Surrogate performance |
| `results/surrogate_models.joblib` | Trained surrogate models |
| `results/sigma_sensitivity.csv` | σ sensitivity sweep (Table S1) |
| `results/structural_sensitivity.csv` | Structural sensitivity sweep (Table S2) |
| `results/marrow_surrogate_check.csv` | Marrow reference diagnostic |
| `figures/fig2_*.png` … `fig7_*.png` | Manuscript figures |
| `figures/figS1_sigma_sensitivity.*` | Supplementary σ figure |
| `figures/figS2_structural_sensitivity.*` | Supplementary structural figure |

## References

- Violet J, et al. J Nucl Med. 2019;60:517–523
- Fendler WP, et al. Oncotarget. 2017;8:3581–3590
- Kurth J, et al. Cancers. 2021;13:3884
- Takano S, et al. Ann Nucl Med. 2025;39:1201–1212
- Jackson P, et al. Semin Nucl Med. 2022;52:243–254
