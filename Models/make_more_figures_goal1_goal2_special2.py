# make_more_figures_goal1_goal2_special2.py
# Saves 3 additional figures to:
#   C:\Users\hocke\Desktop\Pharmacokinetics\figures
#
# 1) Goal 1: Testosterone loss over time, with dose encoding
#    - x: duration (days)
#    - y: TT loss (ng/dL) = TT_pre - TT_post
#    - marker size: nominal dose (mg/day)
#    - optional label: case_id
#
# 2) Special Case 1: Observed TT trajectory (pre -> mid -> end)
#    - line plot of TT vs day (0, 42, 84)
#
# 3) Special Case 2: Day-by-day schedule effect y(d) + dose schedule
#    - x: day
#    - left y: dose (mg/day)
#    - right y: y_emax_effect (dimensionless)
#    - plus horizontal line for y_bar

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\Users\hocke\Desktop\Pharmacokinetics\figures"

# -----------------------------
# PATHS (you specified these; we will use these exact ones)
# -----------------------------
GOAL1_OBS_PATH = os.path.join(OUTPUT_DIR, "redo_lgd_goal1A1B_observed_suppression_and_velocity.csv")
GOAL1_COMBINED_HILL_PATH = os.path.join(OUTPUT_DIR, "redo_lgd_goal1ABC_combined_results_hill.csv")

SC1_COMBINED_PATH = os.path.join(OUTPUT_DIR, "lgd_goal2_special1_combined_observed_plus_model.csv")
SC1_OBS_SHAPE_PATH = os.path.join(OUTPUT_DIR, "lgd_goal2_special1_observed_shape_metrics.csv")

SC2_DAYS_PATH = os.path.join(OUTPUT_DIR, "lgd_special2_daybyday_schedule_effect.csv")
SC2_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "lgd_special2_summary.csv")

def must_exist(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file:\n{p}")

for p in [GOAL1_OBS_PATH, GOAL1_COMBINED_HILL_PATH, SC1_COMBINED_PATH, SC2_DAYS_PATH, SC2_SUMMARY_PATH]:
    must_exist(p)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1) GOAL 1 — TT loss over time (dose-encoded scatter)
# ============================================================
goal1 = pd.read_csv(GOAL1_OBS_PATH)

required = ["case_id", "dose_mg_per_day", "duration_days", "TT_pre", "TT_post"]
missing = [c for c in required if c not in goal1.columns]
if missing:
    raise ValueError(f"Goal1 obs CSV missing columns: {missing}\nFound: {list(goal1.columns)}")

# numeric coercion
for c in ["case_id", "dose_mg_per_day", "duration_days", "TT_pre", "TT_post"]:
    goal1[c] = pd.to_numeric(goal1[c], errors="coerce")

goal1 = goal1.dropna(subset=["case_id", "dose_mg_per_day", "duration_days", "TT_pre", "TT_post"]).copy()
goal1["tt_loss_ngdl"] = goal1["TT_pre"] - goal1["TT_post"]

# Marker size scaled by dose (gentle scaling)
dose = goal1["dose_mg_per_day"].to_numpy(dtype=float)
size = 40.0 + 18.0 * np.sqrt(np.clip(dose, 0.0, None))

plt.figure()
plt.scatter(goal1["duration_days"], goal1["tt_loss_ngdl"], s=size, alpha=0.85)

# annotate case ids (small)
for _, r in goal1.iterrows():
    plt.text(r["duration_days"], r["tt_loss_ngdl"], f"{int(r['case_id'])}", fontsize=9, ha="left", va="bottom", alpha=0.8)

plt.xlabel("Duration (days)")
plt.ylabel("Total testosterone loss (ng/dL) = TT_pre − TT_post")
plt.title("Anecdotal TT loss over time (marker size ∝ nominal dose)")
plt.grid(True, alpha=0.25)

out1 = os.path.join(OUTPUT_DIR, "fig_goal1_tt_loss_vs_duration_dose_encoded.png")
plt.savefig(out1, dpi=220, bbox_inches="tight")
plt.close()

# ============================================================
# 2) SPECIAL CASE 1 — Observed TT trajectory (pre -> mid -> end)
# ============================================================
# Prefer combined (has TT_pre/mid/post + times). If missing, fall back to observed_shape_metrics.
sc1 = pd.read_csv(SC1_COMBINED_PATH)

if "TT_pre" in sc1.columns and "TT_mid" in sc1.columns and "TT_post" in sc1.columns:
    row = sc1.iloc[0]
    TT_pre = float(row["TT_pre"])
    TT_mid = float(row["TT_mid"])
    TT_post = float(row["TT_post"])

    # Use available time columns if present; else default 0/42/84.
    t_mid = int(row["t_mid_days"]) if "t_mid_days" in sc1.columns else 42
    t_end = int(row["t_end_days"]) if "t_end_days" in sc1.columns else 84
else:
    # fallback
    must_exist(SC1_OBS_SHAPE_PATH)
    sc1o = pd.read_csv(SC1_OBS_SHAPE_PATH).iloc[0]
    TT_pre = float(sc1o["TT_pre"])
    TT_mid = float(sc1o["TT_mid"])
    TT_post = float(sc1o["TT_post"])
    t_mid = int(sc1o["t_mid_days"])
    t_end = int(sc1o["t_end_days"])

days = np.array([0, t_mid, t_end], dtype=float)
tts = np.array([TT_pre, TT_mid, TT_post], dtype=float)

plt.figure()
plt.plot(days, tts, marker="o", linewidth=2.5)
plt.xlabel("Day")
plt.ylabel("Total testosterone (ng/dL)")
plt.title("Special case 1 — Observed TT trajectory (pre → mid → end)")
plt.grid(True, alpha=0.25)

out2 = os.path.join(OUTPUT_DIR, "fig_special1_observed_TT_trajectory.png")
plt.savefig(out2, dpi=220, bbox_inches="tight")
plt.close()

# ============================================================
# 3) SPECIAL CASE 2 — Dose schedule + Emax-effect over time
# ============================================================
sc2_days = pd.read_csv(SC2_DAYS_PATH)
sc2_sum = pd.read_csv(SC2_SUMMARY_PATH).iloc[0]

need_sc2 = ["day", "dose_mg_per_day", "y_emax_effect"]
missing_sc2 = [c for c in need_sc2 if c not in sc2_days.columns]
if missing_sc2:
    raise ValueError(f"Special2 day-by-day CSV missing columns: {missing_sc2}\nFound: {list(sc2_days.columns)}")

for c in need_sc2:
    sc2_days[c] = pd.to_numeric(sc2_days[c], errors="coerce")
sc2_days = sc2_days.dropna(subset=need_sc2).copy()

y_bar = float(sc2_sum["schedule_avg_effect_ybar"]) if "schedule_avg_effect_ybar" in sc2_sum.index else float(np.nan)

fig = plt.figure()
ax1 = plt.gca()

# dose (left axis)
ax1.plot(sc2_days["day"], sc2_days["dose_mg_per_day"], linewidth=2.5)
ax1.set_xlabel("Day")
ax1.set_ylabel("Dose (mg/day)")
ax1.grid(True, alpha=0.25)

# effect (right axis)
ax2 = ax1.twinx()
ax2.plot(sc2_days["day"], sc2_days["y_emax_effect"], linestyle="--", linewidth=2.0)
ax2.set_ylabel("Effect proxy y(d)")

if np.isfinite(y_bar):
    ax2.axhline(y_bar, linestyle=":", linewidth=2.0)

plt.title("Special case 2 — Schedule dose vs Emax-weighted effect proxy")
out3 = os.path.join(OUTPUT_DIR, "fig_special2_schedule_dose_vs_effect.png")
plt.savefig(out3, dpi=220, bbox_inches="tight")
plt.close()

print("Saved:")
print(" -", out1)
print(" -", out2)
print(" -", out3)
