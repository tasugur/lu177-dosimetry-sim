"""
Renal-constrained maximum cumulative activity from MC kidney dose samples.

Two outputs:
  1. Theoretical max cumulative activity: 23 Gy / kidney_dose_per_GBq
  2. Probability that the standard 6 x 7.4 GBq schedule stays within 23 Gy
"""

import numpy as np

RENAL_DOSE_LIMIT_GY = 23.0
ACTIVITY_PER_CYCLE  = 7.4    # GBq
STANDARD_TOTAL_GBQ  = 44.4   # 6 * 7.4


def max_constrained_activity_from_samples(
        kidney_dose_samples: np.ndarray,
        activity_per_cycle: float = ACTIVITY_PER_CYCLE) -> dict:
    kidney_dose_per_gbq = kidney_dose_samples / activity_per_cycle
    max_act = RENAL_DOSE_LIMIT_GY / kidney_dose_per_gbq

    cumulative_at_standard = kidney_dose_per_gbq * STANDARD_TOTAL_GBQ
    feasibility_pct = float(np.mean(cumulative_at_standard <= RENAL_DOSE_LIMIT_GY) * 100)

    def _s(arr):
        return {
            "mean": float(np.mean(arr)),
            "p025": float(np.percentile(arr, 2.5)),
            "p975": float(np.percentile(arr, 97.5)),
        }

    return {
        "max_cumulative_activity_gbq": _s(max_act),
        "standard_feasibility_pct":    feasibility_pct,
    }
