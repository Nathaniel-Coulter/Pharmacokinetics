# special_case_2_schedule_vs_avg.py
# Goal: compare (i) time-weighted average daily dose vs (ii) schedule-aware Emax-weighted “effective dose”
# using your special case #2 schedule, plus compute observed suppression + suppression velocity.
#
# Outputs (CSV) -> OUTPUT_DIR:
#   1) lgd_special2_summary.csv
#   2) lgd_special2_daybyday_schedule_effect.csv
#
# NOTE:
# - This does NOT “recommend” dosing; it just computes metrics from the provided anecdote.
# - The Emax-weighted effective dose here is defined by:
#     y(d) = Emax * d / (EC50 + d)
#     y_bar = time-weighted average of y(d(t)) over the schedule
#     d_emax = inverse_y(y_bar) such that y(d_emax) = y_bar
#   This is the clean “schedule-aware equivalent constant dose” under the Hill/Emax class (with n=1).

import os
import math
import numpy as np
import pandas as pd

# -----------------------------
# User inputs (special case #2)
# -----------------------------
CASE_ID = "special_2"
dose_schedule = [(7.5, 35), (15.0, 7)]  # (mg/day, days)
TOTAL_DAYS = sum(days for _, days in dose_schedule)

TT_pre  = 689.3
TT_post = 268.2

# Where to write CSV outputs (Windows path you gave)
OUTPUT_DIR = r"C:\Users\hocke\Desktop\Pharmacokinetics\figures"

# -----------------------------
# PD mapping for "schedule-aware Emax-weighted dose"
# -----------------------------
# If your paper’s dose-space mapping uses a particular EC50, set it here.
# (If you used a different mapping like d_eff=d/(EC50+d) with Emax=1, keep Emax=1.)
Emax = 1.0
EC50 = 10.0  # <-- set to your dose-space EC50 if you have one

def y_emax(d_mg: float, Emax: float, EC50: float) -> float:
    """Fractional effect proxy y(d) in [0, Emax)."""
    d = float(d_mg)
    if d <= 0:
        return 0.0
    return float(Emax * d / (EC50 + d))

def invert_emax(y: float, Emax: float, EC50: float) -> float:
    """Return d such that y = Emax*d/(EC50+d). Valid for 0<y<Emax."""
    y = float(y)
    if not (0.0 < y < Emax):
        return float("nan")
    return float(EC50 * y / (Emax - y))

# -----------------------------
# Goal 1A: observed suppression (endpoint-based)
# -----------------------------
y_obs = (TT_pre - TT_post) / TT_pre  # fraction
suppression_pct = 100.0 * y_obs

# -----------------------------
# Goal 1B: suppression velocity (time-normalized)
# -----------------------------
duration_weeks = TOTAL_DAYS / 7.0
suppression_velocity_pct_per_week = suppression_pct / duration_weeks

# -----------------------------
# Avg dose vs schedule-aware Emax-weighted effective dose
# -----------------------------
# (i) time-weighted average daily dose
total_mg = sum(d * days for d, days in dose_schedule)
d_avg = total_mg / TOTAL_DAYS

# (ii) schedule-aware: average the *effect proxy* y(d(t)) across days,
#      then invert to get an equivalent constant dose
# build day-by-day table
rows = []
day_counter = 0
for dose, days in dose_schedule:
    for _ in range(int(days)):
        day_counter += 1
        y = y_emax(dose, Emax, EC50)
        rows.append({
            "case_id": CASE_ID,
            "day": day_counter,
            "dose_mg_per_day": float(dose),
            "y_emax_effect": float(y),
        })

df_days = pd.DataFrame(rows)
y_bar = float(df_days["y_emax_effect"].mean())
d_emax = invert_emax(y_bar, Emax, EC50)

# Optional: “total effect mass” (useful for intuition)
# This is literally sum_y over days; you can compare across schedules of same length.
y_total = float(df_days["y_emax_effect"].sum())

# Also show the ratio: how much schedule-aware differs from avg dose
ratio_d_emax_to_avg = d_emax / d_avg if (d_avg > 0 and np.isfinite(d_emax)) else float("nan")

# -----------------------------
# Summary table
# -----------------------------
summary = pd.DataFrame([{
    "case_id": CASE_ID,
    "TT_pre": float(TT_pre),
    "TT_post": float(TT_post),
    "duration_days": int(TOTAL_DAYS),
    "duration_weeks": float(duration_weeks),

    # Goal 1A/1B outputs
    "observed_suppression_fraction_yobs": float(y_obs),
    "observed_suppression_pct": float(suppression_pct),
    "suppression_velocity_pct_per_week": float(suppression_velocity_pct_per_week),

    # dose comparisons
    "total_mg_over_schedule": float(total_mg),
    "time_weighted_avg_dose_mg_per_day": float(d_avg),

    # Emax-weighted schedule-aware equivalence
    "Emax_used": float(Emax),
    "EC50_used": float(EC50),
    "schedule_avg_effect_ybar": float(y_bar),
    "schedule_total_effect_sum_y": float(y_total),
    "schedule_aware_equivalent_dose_d_emax_mg_per_day": float(d_emax),
    "d_emax_over_d_avg_ratio": float(ratio_d_emax_to_avg),
}])

# -----------------------------
# Write outputs
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

out_summary = os.path.join(OUTPUT_DIR, "lgd_special2_summary.csv")
out_days    = os.path.join(OUTPUT_DIR, "lgd_special2_daybyday_schedule_effect.csv")

summary.to_csv(out_summary, index=False)
df_days.to_csv(out_days, index=False)

print("Wrote:")
print(" -", out_summary)
print(" -", out_days)
print("\nKey results:")
print(f"Observed suppression: {suppression_pct:.2f}% (y_obs={y_obs:.4f}) over {duration_weeks:.2f} weeks")
print(f"Suppression velocity: {suppression_velocity_pct_per_week:.2f}% per week")
print(f"d_avg  = {d_avg:.4f} mg/day")
print(f"d_emax = {d_emax:.4f} mg/day   (Emax={Emax}, EC50={EC50}, y_bar={y_bar:.6f})")
print(f"ratio d_emax/d_avg = {ratio_d_emax_to_avg:.4f}")