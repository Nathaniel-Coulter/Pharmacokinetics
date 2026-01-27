import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user

# -----------------------------
# Clinical anchors
# -----------------------------
TT_baseline = 622.0
auc_day21 = {0.1: 19.0, 0.3: 85.0, 1.0: 238.0}     # AUC0-24 at day 21 (ng·24h/mL)
delta_TT_day21 = {0.1: -116.0, 0.3: -186.0, 1.0: -282.0}  # mean change at day 21
y_anchor = {d: (-delta_TT_day21[d]) / TT_baseline for d in auc_day21.keys()}  # suppression fraction at day 21

scenarios = {"24h_half_life (r=0.5)": 0.5, "36h_half_life (r=0.63)": 0.63}

def exposure_series(auc21, r, days=21):
    t = np.arange(1, days + 1)
    return auc21 * (1 - r**t) / (1 - r**days)

# -----------------------------
# PD: y(x) = Emax * x / (EC50 + x)
# Add PD uncertainty by sampling and weighting to match the 3 anchors.
# -----------------------------
def y_emax(x, Emax, EC50):
    return Emax * x / (EC50 + x)

def pd_loglik(Emax, EC50, sigma_y=0.02):
    xs = np.array([auc_day21[0.1], auc_day21[0.3], auc_day21[1.0]])
    ys_obs = np.array([y_anchor[0.1], y_anchor[0.3], y_anchor[1.0]])
    ys_hat = y_emax(xs, Emax, EC50)
    return -0.5 * np.sum(((ys_hat - ys_obs) / sigma_y) ** 2)

rng = np.random.default_rng(123)

N_pd_proposals = 20000
Emax_prop = rng.uniform(0.25, 0.95, size=N_pd_proposals)
EC50_prop = rng.lognormal(mean=np.log(60.0), sigma=0.8, size=N_pd_proposals)

ll = np.empty(N_pd_proposals, dtype=float)
for i in range(N_pd_proposals):
    ll[i] = pd_loglik(Emax_prop[i], EC50_prop[i], sigma_y=0.02)

ll -= ll.max()
w = np.exp(ll)
w /= w.sum()

N_pd = 2000
idx = rng.choice(np.arange(N_pd_proposals), size=N_pd, replace=True, p=w)
Emax_s = Emax_prop[idx]
EC50_s = EC50_prop[idx]

pd_draws = pd.DataFrame({"Emax": Emax_s, "EC50": EC50_s})
display_dataframe_to_user("PD draws (posterior-ish) summary", pd_draws.describe(percentiles=[0.05, 0.5, 0.95]))

# -----------------------------
# Lag model (indirect response) and beta refit
# -----------------------------
def simulate_TT(alpha, beta, x_series, Emax, EC50):
    N = len(x_series)
    TT = np.zeros(N + 1, dtype=float)
    TT[0] = TT_baseline
    for t in range(1, N + 1):
        drug_effect = y_emax(x_series[t - 1], Emax, EC50)
        TT[t] = TT[t - 1] + alpha * (TT_baseline - TT[t - 1]) - beta * drug_effect * TT_baseline
    return TT

def beta_to_hit_target(alpha, x_series, TT_target, Emax, EC50):
    TT0 = simulate_TT(alpha, beta=0.0, x_series=x_series, Emax=Emax, EC50=EC50)[-1]
    TT1 = simulate_TT(alpha, beta=1.0, x_series=x_series, Emax=Emax, EC50=EC50)[-1]
    denom = TT1 - TT0
    if abs(denom) < 1e-12:
        return np.nan
    return (TT_target - TT0) / denom

# -----------------------------
# Alpha prior via recovery half-life H ~ lognormal
# -----------------------------
def sample_H_lognormal(rng, n, H_med=10.0, clip=(3.0, 40.0)):
    sigma = np.log(2.0) / 1.96  # 95% ~ within 2x of median
    mu = np.log(H_med)
    H = rng.lognormal(mean=mu, sigma=sigma, size=n)
    H = np.clip(H, clip[0], clip[1])
    return H, sigma

def H_to_alpha(H):
    return 1 - np.exp(-np.log(2) / H)

def H_dose_dependent(H0, dose, gamma=0.25):
    return H0 * (dose / 0.3) ** gamma

days = np.arange(0, 22, dtype=int)
pct_list = [1, 5, 25, 50, 75, 95, 99]

N_mc = 3000
H0, H_sigma = sample_H_lognormal(rng, N_mc, H_med=10.0, clip=(3.0, 40.0))

def run_mc(engine_label, dose_dependent=False, gamma=0.25):
    band_rows = []
    beta_rows = []
    for scen_name, r in scenarios.items():
        for dose, auc21 in auc_day21.items():
            x = exposure_series(auc21, r, days=21)
            TT_target = TT_baseline + delta_TT_day21[dose]

            H_use = H0.copy()
            if dose_dependent:
                H_use = np.clip(H_dose_dependent(H_use, dose, gamma=gamma), 3.0, 60.0)
            alpha = H_to_alpha(H_use)

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
                row = {"Engine": engine_label, "Scenario": scen_name, "Dose_mg_per_day": float(dose), "Day": int(d), "n_mc": int(N_mc)}
                for p_i, p in enumerate(pct_list):
                    row[f"suppression_pct_p{p:02d}"] = float(percs[p_i, d_idx])
                band_rows.append(row)

            beta_rows.append({
                "Engine": engine_label,
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

    return pd.DataFrame(band_rows).sort_values(["Engine", "Scenario", "Dose_mg_per_day", "Day"]), pd.DataFrame(beta_rows).sort_values(["Engine", "Scenario", "Dose_mg_per_day"])

bands_A, betas_A = run_mc("Level2: PK+lag+PD (H dose-independent)", dose_dependent=False)
bands_B, betas_B = run_mc("Level2+: PK+lag+PD (H dose-dependent, gamma=0.25)", dose_dependent=True, gamma=0.25)

bands = pd.concat([bands_A, bands_B], ignore_index=True).sort_values(["Engine","Scenario","Dose_mg_per_day","Day"])
beta_stats = pd.concat([betas_A, betas_B], ignore_index=True).sort_values(["Engine","Scenario","Dose_mg_per_day"])
display_dataframe_to_user("Beta summary (two engines)", beta_stats)

# Save CSVs
bands_path = "/mnt/data/lgd_level2_suppression_bands_mc_alpha_pd.csv"
beta_stats_path = "/mnt/data/lgd_level2_beta_stats_mc_alpha_pd.csv"
pd_draws_path = "/mnt/data/lgd_level2_pd_draws_posteriorish.csv"
bands.to_csv(bands_path, index=False)
beta_stats.to_csv(beta_stats_path, index=False)
pd_draws.to_csv(pd_draws_path, index=False)

# -----------------------------
# Inversion: suppression y -> implied exposure x (AUC) under PD uncertainty
# x = EC50*y/(Emax - y) for y<Emax
# -----------------------------
def invert_emax(y, Emax, EC50):
    if y <= 0 or y >= Emax:
        return np.nan
    return EC50 * y / (Emax - y)

y_grid = np.linspace(0.05, 0.80, 76)  # fraction
inv_rows = []
for y in y_grid:
    xs = np.array([invert_emax(y, Emax_s[i], EC50_s[i]) for i in range(N_pd)])
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        continue
    inv_rows.append({
        "suppression_fraction_y": float(y),
        "suppression_pct": float(100*y),
        "x_AUC_p05": float(np.percentile(xs, 5)),
        "x_AUC_p50": float(np.percentile(xs, 50)),
        "x_AUC_p95": float(np.percentile(xs, 95)),
        "n_pd_used": int(xs.size),
    })
inv = pd.DataFrame(inv_rows)
inv_path = "/mnt/data/lgd_inversion_suppression_to_implied_AUC_pd_uncertainty.csv"
inv.to_csv(inv_path, index=False)

# -----------------------------
# Save a small set of figures (median + bands) as PNGs
# -----------------------------
png_paths = []
for engine_label in bands["Engine"].unique():
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

            safe_engine = engine_label.replace(":", "").replace("+", "plus").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            safe_scen = scen_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            path = f"/mnt/data/fig_{safe_engine}_{safe_scen}_dose{dose}.png"
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            png_paths.append(path)

bands_path, beta_stats_path, pd_draws_path, inv_path, png_paths[:3], len(png_paths)

