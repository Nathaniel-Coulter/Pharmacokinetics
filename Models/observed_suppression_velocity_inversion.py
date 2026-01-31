# ============================================================
# Goal 1A–C (One-shot) — Observed Suppression + Velocity + Inversion
# Outputs CSVs to: C:\Users\hocke\Desktop\Pharmacokinetics\figures
#
# What this script does:
#   1A) Endpoint suppression fraction + suppression%
#   1B) Suppression velocity (% per week)
#   1C) Inversion (model lens): suppression -> implied exposure (AUC proxy band)
#       using PD uncertainty draws (Emax, EC50) fit to Basaria anchors.
#       Also reports AUC21-equivalent exposure under two PK half-life scenarios.
#
# Run once. No dependencies beyond numpy/pandas.
# ============================================================

import os
import numpy as np
import pandas as pd

# -----------------------------
# USER SETTINGS
# -----------------------------
OUTPUT_DIR = r"C:\Users\hocke\Desktop\Pharmacokinetics\figures"

# If you already have this file from earlier work and want to use it, set this path.
# If it doesn't exist, the script will regenerate PD draws internally.
PD_DRAWS_CSV_OPTIONAL = None
# Example:
# PD_DRAWS_CSV_OPTIONAL = r"C:\Users\hocke\Desktop\Pharmacokinetics\figures\lgd_level2_pd_draws_posteriorish.csv"

# -----------------------------
# INPUT DATA (your table)
# -----------------------------
cases = pd.DataFrame(
    [
        # CaseID, Dose(mg/day), Duration(days), TT_pre, TT_post
        (1, 3.0, 14.0, 432.6, 152.9),
        (2, 10.0, 42.0, 483.0, 190.0),
        (3, 10.0, 70.0, 533.0, 146.0),
        (4, 10.0, 70.0, 1363.0, 557.0),
        (5, 3.0, 4.0, 751.0, 353.0),
        (6, 5.0, 35.0, 370.0, 144.0),
        (7, 10.0, 70.0, 650.0, 156.0),
        (8, 25.0, 175.0, 1453.5, 225.8),
        (9, 7.0, 49.0, 570.0, 50.0),
        (10, 12.5, 87.5, 546.0, 261.0),
    ],
    columns=["case_id", "dose_mg_per_day", "duration_days", "TT_pre", "TT_post"],
)

# -----------------------------
# CLINICAL ANCHORS (Basaria-adapted)
# -----------------------------
TT_baseline = 622.0
auc_day21 = {0.1: 19.0, 0.3: 85.0, 1.0: 238.0}  # ng·24h/mL (AUC0-24 at day 21)
delta_TT_day21 = {0.1: -116.0, 0.3: -186.0, 1.0: -282.0}
# Observed suppression fraction at day-21 anchors:
y_anchor = {d: (-delta_TT_day21[d]) / TT_baseline for d in auc_day21.keys()}

# Two PK half-life scenarios used in prior work (for AUC_t -> AUC_21-equivalent scaling)
PK_SCENARIOS = {
    "24h_half_life_r0p5": 0.5,
    "36h_half_life_r0p63": 0.63,
}

# -----------------------------
# 1A) Observed suppression + 1B) velocity
# -----------------------------
cases["duration_weeks"] = cases["duration_days"] / 7.0
cases["y_obs"] = (cases["TT_pre"] - cases["TT_post"]) / cases["TT_pre"]
cases["suppression_pct"] = 100.0 * cases["y_obs"]
cases["velocity_pct_per_week"] = cases["suppression_pct"] / cases["duration_weeks"]

# -----------------------------
# 1C) Inversion (model lens)
#   We use PD uncertainty draws (Emax, EC50) that are "posterior-ish" by weighting priors
#   to match the 3 Basaria anchor points in suppression-fraction space.
#
# Inversion formula for Emax model:
#   y(x) = Emax * x / (EC50 + x)
#   => x = EC50 * y / (Emax - y), valid for 0<y<Emax
# -----------------------------

def y_emax(x, Emax, EC50):
    return Emax * x / (EC50 + x)

def pd_loglik(Emax, EC50, sigma_y=0.02):
    xs = np.array([auc_day21[0.1], auc_day21[0.3], auc_day21[1.0]], dtype=float)
    ys_obs = np.array([y_anchor[0.1], y_anchor[0.3], y_anchor[1.0]], dtype=float)
    ys_hat = y_emax(xs, Emax, EC50)
    return -0.5 * np.sum(((ys_hat - ys_obs) / sigma_y) ** 2)

def invert_emax(y, Emax, EC50):
    if (y <= 0.0) or (y >= Emax) or (Emax <= 0.0) or (EC50 <= 0.0):
        return np.nan
    return EC50 * y / (Emax - y)

def build_pd_draws(seed=123, n_proposals=120000, n_draws=4000, sigma_y=0.02):
    rng = np.random.default_rng(seed)

    # broad priors
    Emax_prop = rng.uniform(0.25, 0.95, size=n_proposals)
    EC50_prop = rng.lognormal(mean=np.log(60.0), sigma=0.8, size=n_proposals)

    # weight to anchors
    ll = np.array([pd_loglik(Emax_prop[i], EC50_prop[i], sigma_y=sigma_y) for i in range(n_proposals)], dtype=float)
    ll -= ll.max()
    w = np.exp(ll)
    w /= w.sum()

    idx = rng.choice(np.arange(n_proposals), size=n_draws, replace=True, p=w)
    return pd.DataFrame({"Emax": Emax_prop[idx], "EC50": EC50_prop[idx]})

# Load or generate PD draws
if PD_DRAWS_CSV_OPTIONAL and os.path.exists(PD_DRAWS_CSV_OPTIONAL):
    pd_draws = pd.read_csv(PD_DRAWS_CSV_OPTIONAL)
    if not {"Emax", "EC50"}.issubset(pd_draws.columns):
        raise ValueError("PD draws CSV must contain columns: Emax, EC50")
else:
    pd_draws = build_pd_draws()

Emax_s = pd_draws["Emax"].to_numpy(dtype=float)
EC50_s = pd_draws["EC50"].to_numpy(dtype=float)

# Inversion per case: compute implied AUC band for observed suppression fraction
inv_rows = []
for _, row in cases.iterrows():
    y = float(row["y_obs"])
    # implied exposure at measurement time t* (AUC-like scale)
    xs = np.array([invert_emax(y, Emax_s[i], EC50_s[i]) for i in range(len(Emax_s))], dtype=float)
    xs = xs[np.isfinite(xs)]

    out = {
        "case_id": int(row["case_id"]),
        "y_obs": y,
        "suppression_pct": float(row["suppression_pct"]),
        "duration_days": float(row["duration_days"]),
        "dose_mg_per_day": float(row["dose_mg_per_day"]),
        "n_pd_valid": int(xs.size),
    }

    if xs.size == 0:
        out.update({
            "AUC_t_p05": np.nan,
            "AUC_t_p50": np.nan,
            "AUC_t_p95": np.nan,
        })
    else:
        out.update({
            "AUC_t_p05": float(np.percentile(xs, 5)),
            "AUC_t_p50": float(np.percentile(xs, 50)),
            "AUC_t_p95": float(np.percentile(xs, 95)),
        })

    # Convert implied AUC at measurement day t* to "AUC21-equivalent" under toy accumulation
    # using: x_t = AUC21 * (1 - r^t) / (1 - r^21)  =>  AUC21 = x_t * (1 - r^21) / (1 - r^t)
    t = float(row["duration_days"])
    for scen, r in PK_SCENARIOS.items():
        denom = (1.0 - (r ** t))
        numer = (1.0 - (r ** 21.0))
        if (not np.isfinite(denom)) or abs(denom) < 1e-12:
            out[f"AUC21eq_{scen}_p05"] = np.nan
            out[f"AUC21eq_{scen}_p50"] = np.nan
            out[f"AUC21eq_{scen}_p95"] = np.nan
        else:
            scale = numer / denom
            out[f"AUC21eq_{scen}_p05"] = out["AUC_t_p05"] * scale if np.isfinite(out["AUC_t_p05"]) else np.nan
            out[f"AUC21eq_{scen}_p50"] = out["AUC_t_p50"] * scale if np.isfinite(out["AUC_t_p50"]) else np.nan
            out[f"AUC21eq_{scen}_p95"] = out["AUC_t_p95"] * scale if np.isfinite(out["AUC_t_p95"]) else np.nan

    # Simple classification vs clinical AUC anchors using AUC21-equivalent median (24h scenario)
    # (purely descriptive)
    med_ref = out.get("AUC21eq_24h_half_life_r0p5_p50", np.nan)
    if np.isfinite(med_ref):
        if med_ref < auc_day21[0.1]:
            cls = "below_clinical_0.1mg"
        elif med_ref < auc_day21[0.3]:
            cls = "between_0.1_and_0.3mg"
        elif med_ref < auc_day21[1.0]:
            cls = "between_0.3_and_1.0mg"
        else:
            cls = "above_clinical_1.0mg"
    else:
        cls = "no_solution_under_PD_draws"
    out["implied_exposure_class_24h"] = cls

    inv_rows.append(out)

inv_cases = pd.DataFrame(inv_rows).sort_values("case_id")

# -----------------------------
# OUTPUTS
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

out_1A1B = cases[[
    "case_id", "dose_mg_per_day", "duration_days", "duration_weeks",
    "TT_pre", "TT_post", "y_obs", "suppression_pct", "velocity_pct_per_week"
]].copy()

out_1C = inv_cases.copy()

out_combined = out_1A1B.merge(
    out_1C.drop(columns=["dose_mg_per_day", "duration_days", "y_obs", "suppression_pct"], errors="ignore"),
    on="case_id",
    how="left",
)

# Save PD draws actually used (for audit / reproducibility)
pd_draws_out = pd_draws.copy()

csv_1A1B = os.path.join(OUTPUT_DIR, "lgd_goal1A1B_observed_suppression_and_velocity.csv")
csv_1C = os.path.join(OUTPUT_DIR, "lgd_goal1C_inversion_implied_exposure_bands.csv")
csv_all = os.path.join(OUTPUT_DIR, "lgd_goal1ABC_combined_results.csv")
csv_pd = os.path.join(OUTPUT_DIR, "lgd_goal1_pd_draws_used.csv")

out_1A1B.to_csv(csv_1A1B, index=False)
out_1C.to_csv(csv_1C, index=False)
out_combined.to_csv(csv_all, index=False)
pd_draws_out.to_csv(csv_pd, index=False)

print("Saved:")
print(" -", csv_1A1B)
print(" -", csv_1C)
print(" -", csv_all)
print(" -", csv_pd)
