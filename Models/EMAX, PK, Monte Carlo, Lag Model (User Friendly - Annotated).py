import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user

# -----------------------------
# Inputs / clinical anchors
# -----------------------------
TT_baseline = 622.0
auc_day21 = {0.1: 19.0, 0.3: 85.0, 1.0: 238.0}  # AUC0-24 at day 21 (ng·24h/mL)
delta_TT_day21 = {0.1: -116.0, 0.3: -186.0, 1.0: -282.0}  # mean change at day 21

# Anchor y at day 21 (suppression fraction)
y_anchor = {d: (-delta_TT_day21[d]) / TT_baseline for d in auc_day21.keys()}

# PK scenarios (toy exposure build)
scenarios = {"24h_half_life (r=0.5)": 0.5, "36h_half_life (r=0.63)": 0.63}

def exposure_series(auc21, r, days=21):
    """Toy build: daily AUC rises toward measured day-21 AUC in a geometric way, normalized so x_21=auc21."""
    t = np.arange(1, days + 1)
    return auc21 * (1 - r**t) / (1 - r**days)  # length=days

# -----------------------------
# PD mapping y(x) = Emax * x / (EC50 + x)
# We will add uncertainty by sampling (Emax, EC50) from a broad prior,
# then weighting samples by how well they match the 3 anchor points.
# -----------------------------
def y_emax(x, Emax, EC50):
    return Emax * x / (EC50 + x)

def pd_loglik(Emax, EC50, sigma_y=0.02):
    """Gaussian log-likelihood on the 3 anchor points in suppression fraction space."""
    xs = np.array([auc_day21[0.1], auc_day21[0.3], auc_day21[1.0]])
    ys_obs = np.array([y_anchor[0.1], y_anchor[0.3], y_anchor[1.0]])
    ys_hat = y_emax(xs, Emax, EC50)
    return -0.5 * np.sum(((ys_hat - ys_obs) / sigma_y) ** 2)

# -----------------------------
# Lag model (indirect response) with alpha/beta
# TT_{t+1} = TT_t + alpha*(TT_baseline - TT_t) - beta * y(x_t) * TT_baseline
# For each draw, we re-fit beta so TT_21 matches the clinical day-21 drop per dose.
# -----------------------------
def simulate_TT(alpha, beta, x_series, Emax, EC50):
    N = len(x_series)
    TT = np.zeros(N + 1, dtype=float)
    TT[0] = TT_baseline
    for t in range(1, N + 1):
        drug_effect = y_emax(x_series[t - 1], Emax, EC50)  # suppression fraction
        TT[t] = TT[t - 1] + alpha * (TT_baseline - TT[t - 1]) - beta * drug_effect * TT_baseline
    return TT

def beta_to_hit_target(alpha, x_series, TT_target, Emax, EC50):
    TT0 = simulate_TT(alpha, beta=0.0, x_series=x_series, Emax=Emax, EC50=EC50)[-1]
    TT1 = simulate_TT(alpha, beta=1.0, x_series=x_series, Emax=Emax, EC50=EC50)[-1]
    # protect against degeneracy
    denom = (TT1 - TT0)
    if abs(denom) < 1e-9:
        return np.nan
    return (TT_target - TT0) / denom

# -----------------------------
# Alpha priors (H prior) and optional dose-dependence
# -----------------------------
def sample_H_lognormal(rng, n, H_med=10.0, spread_2x_95=True, clip=(3.0, 40.0)):
    # Choose sigma so that ~95% of mass within /2x of median (roughly) if spread_2x_95
    if spread_2x_95:
        sigma = np.log(2.0) / 1.96
    else:
        sigma = np.log(1.5) / 1.96
    mu = np.log(H_med)
    H = rng.lognormal(mean=mu, sigma=sigma, size=n)
    H = np.clip(H, clip[0], clip[1])
    return H, sigma

def H_dose_dependent(H0, dose, gamma=0.25):
    """
    Optional, assumption-heavy sensitivity: H increases with dose mildly.
    H(dose) = H0 * (dose / 0.3)^gamma
    - gamma=0 => dose-independent
    - positive gamma => higher dose => slower recovery (larger H => smaller alpha)
    """
    return H0 * (dose / 0.3) ** gamma

# -----------------------------
# PD uncertainty sampling with likelihood weighting
# -----------------------------
rng = np.random.default_rng(123)

N_pd_proposals = 120000
# Broad priors (intentionally generous):
Emax_prop = rng.uniform(0.25, 0.95, size=N_pd_proposals)   # max suppression fraction
EC50_prop = rng.lognormal(mean=np.log(60.0), sigma=0.8, size=N_pd_proposals)  # exposure scale

# Weight by likelihood to anchors
sigma_y = 0.02  # 2 percentage points in fraction space ~ reasonable anchor tolerance
ll = np.array([pd_loglik(Emax_prop[i], EC50_prop[i], sigma_y=sigma_y) for i in range(N_pd_proposals)])
# stabilize
ll = ll - ll.max()
w = np.exp(ll)
w = w / w.sum()

# Resample posterior-ish PD params
N_pd = 4000
idx = rng.choice(np.arange(N_pd_proposals), size=N_pd, replace=True, p=w)
Emax_s = Emax_prop[idx]
EC50_s = EC50_prop[idx]

pd_draws = pd.DataFrame({"Emax": Emax_s, "EC50": EC50_s})
display_dataframe_to_user("PD draws (Emax/EC50) weighted to match clinical anchors)", pd_draws.describe(percentiles=[0.05,0.5,0.95]))

# -----------------------------
# Monte Carlo: PK + lag + PD uncertainty
# We produce two variants:
#   A) Dose-independent H prior
#   B) Dose-dependent H prior (sensitivity)
# -----------------------------
N_mc = 8000  # per dose per scenario (kept moderate for speed)
H0, H_sigma = sample_H_lognormal(rng, N_mc, H_med=10.0, spread_2x_95=True, clip=(3.0, 40.0))

# Map H->alpha
def H_to_alpha(H):
    return 1 - np.exp(-np.log(2) / H)

days = np.arange(0, 22, dtype=int)
pct_list = [1,5,25,50,75,95,99]

def run_mc(dose_dependent=False, gamma=0.25, label=""):
    band_rows = []
    beta_rows = []
    for scen_name, r in scenarios.items():
        for dose, auc21 in auc_day21.items():
            x = exposure_series(auc21, r, days=21)
            TT_target = TT_baseline + delta_TT_day21[dose]

            # Sample alphas (via H)
            H_use = H0.copy()
            if dose_dependent:
                H_use = H_dose_dependent(H_use, dose, gamma=gamma)
                H_use = np.clip(H_use, 3.0, 60.0)
            alpha = H_to_alpha(H_use)

            # Pair each alpha sample with a PD draw (cycle through PD draws)
            j = np.arange(N_mc) % N_pd
            Emax = Emax_s[j]
            EC50 = EC50_s[j]

            sup = np.empty((N_mc, 22), dtype=float)
            betas = np.empty(N_mc, dtype=float)

            for i in range(N_mc):
                b = beta_to_hit_target(alpha[i], x, TT_target, Emax[i], EC50[i])
                betas[i] = b
                TT = simulate_TT(alpha[i], b, x, Emax[i], EC50[i])
                sup[i, :] = 100.0 * (TT_baseline - TT) / TT_baseline

            percs = np.percentile(sup, pct_list, axis=0)
            for d_idx, d in enumerate(days):
                row = {"Engine": label, "Scenario": scen_name, "Dose_mg_per_day": float(dose), "Day": int(d), "n_mc": int(N_mc)}
                for p_i, p in enumerate(pct_list):
                    row[f"suppression_pct_p{p:02d}"] = float(percs[p_i, d_idx])
                band_rows.append(row)

            beta_rows.append({
                "Engine": label,
                "Scenario": scen_name,
                "Dose_mg_per_day": float(dose),
                "n_mc": int(N_mc),
                "H_sigma_logn": float(H_sigma),
                "dose_dependent_H": bool(dose_dependent),
                "gamma": float(gamma if dose_dependent else 0.0),
                "beta_mean": float(np.nanmean(betas)),
                "beta_sd": float(np.nanstd(betas, ddof=1)),
                "beta_p05": float(np.nanpercentile(betas, 5)),
                "beta_p50": float(np.nanpercentile(betas, 50)),
                "beta_p95": float(np.nanpercentile(betas, 95)),
            })

    return pd.DataFrame(band_rows).sort_values(["Engine","Scenario","Dose_mg_per_day","Day"]), pd.DataFrame(beta_rows).sort_values(["Engine","Scenario","Dose_mg_per_day"])

bands_A, betas_A = run_mc(dose_dependent=False, label="Level2: PK+lag+PD (H dose-independent)")
bands_B, betas_B = run_mc(dose_dependent=True, gamma=0.25, label="Level2+: PK+lag+PD (H dose-dependent, gamma=0.25)")

bands = pd.concat([bands_A, bands_B], ignore_index=True).sort_values(["Engine","Scenario","Dose_mg_per_day","Day"])
beta_stats = pd.concat([betas_A, betas_B], ignore_index=True).sort_values(["Engine","Scenario","Dose_mg_per_day"])

display_dataframe_to_user("Beta summary (Level 2 with PD uncertainty; two H variants)", beta_stats)

# Save CSVs
bands_path = "/mnt/data/lgd_level2_suppression_bands_mc_alpha_pd.csv"
beta_stats_path = "/mnt/data/lgd_level2_beta_stats_mc_alpha_pd.csv"
pd_draws_path = "/mnt/data/lgd_level2_pd_draws_posteriorish.csv"

bands.to_csv(bands_path, index=False)
beta_stats.to_csv(beta_stats_path, index=False)
pd_draws.to_csv(pd_draws_path, index=False)

# -----------------------------
# Inversion: observed suppression at day 21 -> implied exposure distribution (AUC units)
# Using PD uncertainty only (lag cancels if we stick to the Basaria-style day-21 mapping y(x))
# For each PD draw, invert x = EC50*y / (Emax - y) for y < Emax
# -----------------------------
def invert_emax(y, Emax, EC50):
    if y <= 0 or y >= Emax:
        return np.nan
    return EC50 * y / (Emax - y)

# Build a grid of observed day-21 suppression fractions (5% to 80%)
y_grid = np.linspace(0.05, 0.80, 76)  # fraction
inv_rows = []
for y in y_grid:
    xs = np.array([invert_emax(y, Emax_s[i], EC50_s[i]) for i in range(N_pd)])
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        continue
    inv_rows.append({
        "suppression_fraction_y": float(y),
        "suppression_pct": float(100*y),
        "x_AUC_p05": float(np.percentile(xs, 5)),
        "x_AUC_p50": float(np.percentile(xs, 50)),
        "x_AUC_p95": float(np.percentile(xs, 95)),
        "n_pd_used": int(len(xs))
    })
inv = pd.DataFrame(inv_rows)

inv_path = "/mnt/data/lgd_inversion_suppression_to_implied_AUC_pd_uncertainty.csv"
inv.to_csv(inv_path, index=False)

# -----------------------------
# Quick figures: one per engine, scenario, dose (median + 25-75 + 5-95)
# -----------------------------
def plot_engine(engine_label):
    for scen_name in scenarios.keys():
        for dose in sorted(auc_day21.keys()):
            g = bands[(bands["Engine"] == engine_label) & (bands["Scenario"] == scen_name) & (bands["Dose_mg_per_day"] == float(dose))].sort_values("Day")
            x_day = g["Day"].to_numpy(dtype=float)
            p05 = g["suppression_pct_p05"].to_numpy(dtype=float)
            p25 = g["suppression_pct_p25"].to_numpy(dtype=float)
            p50 = g["suppression_pct_p50"].to_numpy(dtype=float)
            p75 = g["suppression_pct_p75"].to_numpy(dtype=float)
            p95 = g["suppression_pct_p95"].to_numpy(dtype=float)

            plt.figure()
            plt.fill_between(x_day, p05, p95, alpha=0.12, label="5–95%")
            plt.fill_between(x_day, p25, p75, alpha=0.22, label="25–75%")
            plt.plot(x_day, p50, label="Median")
            plt.xlabel("Day")
            plt.ylabel("Implied TT suppression (%)")
            plt.title(f"{engine_label}\n{dose} mg/day — {scen_name}")
            plt.legend()
            plt.show()

plot_engine("Level2: PK+lag+PD (H dose-independent)")
plot_engine("Level2+: PK+lag+PD (H dose-dependent, gamma=0.25)")

bands_path, beta_stats_path, pd_draws_path, inv_path
