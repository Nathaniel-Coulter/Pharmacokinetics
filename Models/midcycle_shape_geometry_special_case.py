"""
Goal 2 — Analyze the 12-week case with mid-cycle bloodwork (shape / geometry validation)

What this script does (single run):
1) Computes observed endpoint suppressions:
   - pre → mid (day 42)
   - mid → end (day 84)
   - pre → end (day 84)
   plus suppression velocity (per week) for each segment.

2) Runs a person-specific Monte Carlo “shape check” using your Hill-PD draws:
   - Builds a toy exposure series x_t out to day 84 (two PK scenarios: r=0.5 and r=0.63)
   - Samples recovery (alpha) via lognormal H
   - Samples PD parameters (Emax, EC50, n) from your saved Hill PD draws CSV
   - Fits beta *per draw* so TT_end(day 84) matches the observed TT_post exactly
   - Predicts TT_mid(day 42) distribution and checks whether observed TT_mid lies inside bands

Outputs:
- CSV summary (observed + model bands)
- Optional PNG plot(s)
Saves to: C:\Users\hocke\Desktop\Pharmacokinetics\figures
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER CONFIG
# =========================
OUTPUT_DIR = r"C:\Users\you\Desktop\Pharmacokinetics\figures"



PD_DRAWS_HILL_PATH = os.path.join(OUTPUT_DIR, "redo_lgd_goal1_pd_draws_used_hill.csv")


CASE = {
    "case_id": "special_1",
    "dose_mg_per_day": 10.0,
    "t_mid_days": 42,
    "t_end_days": 84,
    "TT_pre": 483.0,
    "TT_mid": 286.0,
    "TT_post": 190.0,
}


TT_BASELINE = 622.0


AUC21_AT_1MG = 238.0


SCENARIOS = {
    "24h_half_life (r=0.5)": 0.5,
    "36h_half_life (r=0.63)": 0.63,
}


N_MC = 15000              # increase if you want tighter quantiles
RNG_SEED = 7

# Alpha prior via recovery half-life H ~ lognormal (same spirit as earlier)
H_MED = 10.0              # days
H_CLIP = (3.0, 40.0)      # days
SPREAD_2X_95 = True       # sigma chosen so ~95% within 2x of median

# Optional: also report results under a "dose-aware" H(dose) sensitivity (you used gamma=0.25 earlier)
DOSE_DEPENDENT_H = False
GAMMA = 0.25              # only used if DOSE_DEPENDENT_H=True

# Plotting
MAKE_PLOTS = True

# =========================
# HELPERS
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def suppression_fraction(TT_pre, TT_post):
    # y_obs = (TT_pre - TT_post)/TT_pre
    if TT_pre <= 0:
        return np.nan
    return (TT_pre - TT_post) / TT_pre

def suppression_pct(TT_pre, TT_post):
    y = suppression_fraction(TT_pre, TT_post)
    return 100.0 * y

def suppression_velocity_per_week(TT_pre, TT_post, duration_days):
    # v = (suppression%)/weeks
    weeks = duration_days / 7.0
    if weeks <= 0:
        return np.nan
    return suppression_pct(TT_pre, TT_post) / weeks

def sample_H_lognormal(rng, n, H_med=10.0, spread_2x_95=True, clip=(3.0, 40.0)):
    if spread_2x_95:
        sigma = np.log(2.0) / 1.96
    else:
        sigma = np.log(1.5) / 1.96
    mu = np.log(H_med)
    H = rng.lognormal(mean=mu, sigma=sigma, size=n)
    H = np.clip(H, clip[0], clip[1])
    return H, sigma

def H_to_alpha(H):
    # alpha = 1 - exp(-ln2/H)
    return 1.0 - np.exp(-np.log(2.0) / H)

def H_dose_dependent(H0, dose, gamma=0.25):
    # H(dose) = H0 * (dose/0.3)^gamma
    return H0 * (dose / 0.3) ** gamma

def exposure_series(auc21, r, days):
    """
    Toy exposure build:
    x_t = auc21 * (1 - r^t)/(1 - r^21) for t=1..days
    This guarantees x_21 = auc21 and continues accumulating (toward asymptote) beyond day 21.
    """
    t = np.arange(1, days + 1, dtype=float)
    denom = (1.0 - r**21)
    if abs(denom) < 1e-12:
        raise ValueError("r too close to 1; cannot normalize.")
    return auc21 * (1.0 - r**t) / denom

def y_hill(x, Emax, EC50, n):
    # y(x) = Emax * x^n/(EC50^n + x^n)
    x = np.asarray(x, dtype=float)
    # guard
    n = max(float(n), 1e-6)
    EC50 = max(float(EC50), 1e-12)
    Emax = float(Emax)
    xn = np.power(np.clip(x, 0.0, None), n)
    denom = np.power(EC50, n) + xn
    return Emax * xn / denom

def simulate_TT(alpha, beta, x_series, Emax, EC50, n, TT0=TT_BASELINE):
    """
    Indirect response:
    TT_{t+1} = TT_t + alpha*(TT_baseline - TT_t) - beta*y(x_t)*TT_baseline
    """
    N = len(x_series)
    TT = np.zeros(N + 1, dtype=float)
    TT[0] = TT0
    for t in range(1, N + 1):
        drug_eff = y_hill(x_series[t - 1], Emax, EC50, n)  # suppression fraction
        TT[t] = TT[t - 1] + alpha * (TT_BASELINE - TT[t - 1]) - beta * drug_eff * TT_BASELINE
    return TT

def beta_to_hit_target(alpha, x_series, TT_target, Emax, EC50, n):
    """
    TT is linear in beta for fixed alpha + PD params; solve with 2 sims.
    """
    TT0 = simulate_TT(alpha, beta=0.0, x_series=x_series, Emax=Emax, EC50=EC50, n=n)[-1]
    TT1 = simulate_TT(alpha, beta=1.0, x_series=x_series, Emax=Emax, EC50=EC50, n=n)[-1]
    denom = (TT1 - TT0)
    if abs(denom) < 1e-12 or not np.isfinite(denom):
        return np.nan
    return (TT_target - TT0) / denom

# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    rng = np.random.default_rng(RNG_SEED)

    # ---- Observed metrics (shape checks)
    t_mid = int(CASE["t_mid_days"])
    t_end = int(CASE["t_end_days"])

    TT_pre  = float(CASE["TT_pre"])
    TT_mid  = float(CASE["TT_mid"])
    TT_post = float(CASE["TT_post"])

    # segment suppressions (endpoint-based)
    y_pre_mid = suppression_fraction(TT_pre, TT_mid)
    y_mid_end = suppression_fraction(TT_mid, TT_post)
    y_pre_end = suppression_fraction(TT_pre, TT_post)

    # velocities per week
    v_pre_mid = suppression_velocity_per_week(TT_pre, TT_mid, t_mid)
    v_mid_end = suppression_velocity_per_week(TT_mid, TT_post, t_end - t_mid)
    v_pre_end = suppression_velocity_per_week(TT_pre, TT_post, t_end)

    # piecewise slope diagnostics in fraction space (as in your screenshot concept)
    # s1 = y(pre->mid)/t_mid, s2 = (y(pre->end)-y(pre->mid))/(t_end - t_mid)
    s1 = y_pre_mid / t_mid if t_mid > 0 else np.nan
    s2 = (y_pre_end - y_pre_mid) / (t_end - t_mid) if (t_end - t_mid) > 0 else np.nan
    accel_flag = "accelerated_late" if (np.isfinite(s1) and np.isfinite(s2) and s2 > s1) else \
                 "slowed_or_plateaued" if (np.isfinite(s1) and np.isfinite(s2) and s2 < s1) else \
                 "undefined"

    observed_summary = pd.DataFrame([{
        "case_id": CASE["case_id"],
        "dose_mg_per_day": CASE["dose_mg_per_day"],
        "t_mid_days": t_mid,
        "t_end_days": t_end,
        "TT_pre": TT_pre,
        "TT_mid": TT_mid,
        "TT_post": TT_post,
        "y_pre_mid": y_pre_mid,
        "y_mid_end": y_mid_end,
        "y_pre_end": y_pre_end,
        "supp_pct_pre_mid": 100*y_pre_mid,
        "supp_pct_mid_end": 100*y_mid_end,
        "supp_pct_pre_end": 100*y_pre_end,
        "vel_pct_per_week_pre_mid": v_pre_mid,
        "vel_pct_per_week_mid_end": v_mid_end,
        "vel_pct_per_week_pre_end": v_pre_end,
        "s1_frac_per_day": s1,
        "s2_frac_per_day": s2,
        "shape_flag": accel_flag
    }])

    # ---- Load Hill PD draws (Emax, EC50, n)
    if not os.path.exists(PD_DRAWS_HILL_PATH):
        raise FileNotFoundError(
            f"Could not find Hill PD draws CSV at:\n{PD_DRAWS_HILL_PATH}\n"
            f"Update PD_DRAWS_HILL_PATH or drop the file into OUTPUT_DIR."
        )

    pd_draws = pd.read_csv(PD_DRAWS_HILL_PATH)

    # accept common column variants
    cols = {c.lower(): c for c in pd_draws.columns}
    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    col_emax = pick("Emax", "emax")
    col_ec50 = pick("EC50", "ec50")
    col_n    = pick("n", "hill_n", "hill", "Hill", "Hill_n")

    if col_emax is None or col_ec50 is None or col_n is None:
        raise ValueError(
            f"PD draws must have columns like Emax, EC50, n (Hill coefficient).\n"
            f"Found columns: {list(pd_draws.columns)}"
        )

    # sample PD draws with replacement for MC
    pd_idx = rng.integers(0, len(pd_draws), size=N_MC)
    Emax_s = pd_draws.loc[pd_idx, col_emax].to_numpy(dtype=float)
    EC50_s = pd_draws.loc[pd_idx, col_ec50].to_numpy(dtype=float)
    n_s    = pd_draws.loc[pd_idx, col_n].to_numpy(dtype=float)

    # ---- sample alpha via H prior
    H0, H_sigma = sample_H_lognormal(rng, N_MC, H_med=H_MED, spread_2x_95=SPREAD_2X_95, clip=H_CLIP)
    if DOSE_DEPENDENT_H:
        H_use = np.clip(H_dose_dependent(H0, CASE["dose_mg_per_day"], gamma=GAMMA), 3.0, 60.0)
    else:
        H_use = H0
    alpha_s = H_to_alpha(H_use)

    # ---- build exposure proxy for this dose
    # Simple, transparent assumption: AUC21 scales linearly with dose (as a proxy; not a claim of PK truth).
    auc21_proxy = AUC21_AT_1MG * (CASE["dose_mg_per_day"] / 1.0)

    # We need x_t through day 84 for each scenario
    results_rows = []

    for scen_name, r in SCENARIOS.items():
        x84 = exposure_series(auc21_proxy, r, days=t_end)   # length 84, indexed day 1..84

        # run MC: fit beta to hit TT_end exactly for each draw
        betas = np.empty(N_MC, dtype=float)
        TT_mid_pred = np.empty(N_MC, dtype=float)
        TT_end_pred = np.empty(N_MC, dtype=float)

        for i in range(N_MC):
            b = beta_to_hit_target(alpha_s[i], x84, TT_target=TT_post, Emax=Emax_s[i], EC50=EC50_s[i], n=n_s[i])
            betas[i] = b
            if not np.isfinite(b):
                TT_mid_pred[i] = np.nan
                TT_end_pred[i] = np.nan
                continue
            TT = simulate_TT(alpha_s[i], b, x84, Emax=Emax_s[i], EC50=EC50_s[i], n=n_s[i])
            TT_mid_pred[i] = TT[t_mid]   # TT[42]
            TT_end_pred[i] = TT[t_end]   # TT[84] (should equal TT_post up to numeric error)

        # filter valid
        m = np.isfinite(TT_mid_pred) & np.isfinite(betas)
        n_valid = int(m.sum())

        if n_valid == 0:
            # still output a row, but mark as failed
            results_rows.append({
                "case_id": CASE["case_id"],
                "scenario": scen_name,
                "r": r,
                "auc21_proxy": auc21_proxy,
                "n_mc": N_MC,
                "n_valid": 0,
                "alpha_med": float(np.nanmedian(alpha_s)),
                "H_med_assumed": H_MED,
                "H_sigma_logn": float(H_sigma),
                "dose_dependent_H": bool(DOSE_DEPENDENT_H),
                "gamma": float(GAMMA if DOSE_DEPENDENT_H else 0.0),
                "beta_p05": np.nan,
                "beta_p50": np.nan,
                "beta_p95": np.nan,
                "TT_mid_p05": np.nan,
                "TT_mid_p50": np.nan,
                "TT_mid_p95": np.nan,
                "supp_mid_p05": np.nan,
                "supp_mid_p50": np.nan,
                "supp_mid_p95": np.nan,
                "obs_TT_mid": TT_mid,
                "obs_supp_mid_pct_vs_baseline622": 100.0 * (TT_BASELINE - TT_mid) / TT_BASELINE,
                "obs_inside_mid_band": False,
                "note": "no_valid_mc_draws"
            })
            continue

        # quantify bands
        def q(a, p): return float(np.nanpercentile(a[m], p))

        TTm_p05, TTm_p50, TTm_p95 = q(TT_mid_pred, 5), q(TT_mid_pred, 50), q(TT_mid_pred, 95)

        # suppression relative to baseline 622 (consistent with your plots)
        supp_mid = 100.0 * (TT_BASELINE - TT_mid_pred[m]) / TT_BASELINE
        supp_mid_p05 = float(np.percentile(supp_mid, 5))
        supp_mid_p50 = float(np.percentile(supp_mid, 50))
        supp_mid_p95 = float(np.percentile(supp_mid, 95))

        # check if observed mid TT lies inside model band
        inside_mid = (TT_mid >= TTm_p05) and (TT_mid <= TTm_p95)

        results_rows.append({
            "case_id": CASE["case_id"],
            "scenario": scen_name,
            "r": r,
            "auc21_proxy": auc21_proxy,
            "n_mc": N_MC,
            "n_valid": n_valid,
            "alpha_med": float(np.median(alpha_s)),
            "H_med_assumed": H_MED,
            "H_sigma_logn": float(H_sigma),
            "dose_dependent_H": bool(DOSE_DEPENDENT_H),
            "gamma": float(GAMMA if DOSE_DEPENDENT_H else 0.0),
            "beta_p05": q(betas, 5),
            "beta_p50": q(betas, 50),
            "beta_p95": q(betas, 95),
            "TT_mid_p05": TTm_p05,
            "TT_mid_p50": TTm_p50,
            "TT_mid_p95": TTm_p95,
            "supp_mid_p05": supp_mid_p05,
            "supp_mid_p50": supp_mid_p50,
            "supp_mid_p95": supp_mid_p95,
            "obs_TT_mid": TT_mid,
            "obs_supp_mid_pct_vs_baseline622": 100.0 * (TT_BASELINE - TT_mid) / TT_BASELINE,
            "obs_inside_mid_band": bool(inside_mid),
            "note": ""
        })

        # ---- optional plot: one scenario, show TT curve bands? (lightweight: just show mid band + observed)
        if MAKE_PLOTS:
            # We’ll only do a compact visualization: predicted mid TT distribution vs observed
            plt.figure()
            plt.hist(TT_mid_pred[m], bins=60)
            plt.axvline(TT_mid, linewidth=2, label="Observed TT_mid")
            plt.axvline(TTm_p05, linestyle="--", linewidth=1, label="p05/p95 band")
            plt.axvline(TTm_p95, linestyle="--", linewidth=1)
            plt.xlabel("Predicted TT at day 42")
            plt.ylabel("Count")
            plt.title(f"{CASE['case_id']} — {scen_name}\nMC predicts TT_mid; beta re-fit to hit TT_end exactly")
            plt.legend()
            out_png = os.path.join(OUTPUT_DIR, f"{CASE['case_id']}_goal2_mid_TT_hist_{scen_name.replace(' ','_').replace('(','').replace(')','')}.png")
            plt.savefig(out_png, dpi=170, bbox_inches="tight")
            plt.close()

    model_summary = pd.DataFrame(results_rows)

    # Save outputs
    out_obs = os.path.join(OUTPUT_DIR, "lgd_goal2_special1_observed_shape_metrics.csv")
    out_mod = os.path.join(OUTPUT_DIR, "lgd_goal2_special1_model_midpoint_check.csv")

    observed_summary.to_csv(out_obs, index=False)
    model_summary.to_csv(out_mod, index=False)

    # Also produce a single combined one-row-per-scenario table that’s paper-friendly
    combined = observed_summary.merge(model_summary, on="case_id", how="cross")
    out_comb = os.path.join(OUTPUT_DIR, "lgd_goal2_special1_combined_observed_plus_model.csv")
    combined.to_csv(out_comb, index=False)

    print("Saved:")
    print(" -", out_obs)
    print(" -", out_mod)
    print(" -", out_comb)
    print("\nQuick read:")
    print(observed_summary.to_string(index=False))
    print("\nModel midpoint check (per scenario):")
    print(model_summary[[
        "scenario","n_valid","TT_mid_p05","TT_mid_p50","TT_mid_p95","obs_TT_mid","obs_inside_mid_band"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()