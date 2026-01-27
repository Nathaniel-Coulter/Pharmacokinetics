import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user

TT_baseline = 622.0
auc_day21 = {0.1: 19.0, 0.3: 85.0, 1.0: 238.0}
delta_TT_day21 = {0.1: -116.0, 0.3: -186.0, 1.0: -282.0}

Emax = 0.5099542401801108
EC50 = 42.704725734240566

def y_emax(x):
    return Emax * x / (EC50 + x)

def exposure_series(auc21, r, days=21):
    t = np.arange(1, days + 1)
    return auc21 * (1 - r**t) / (1 - r**days)

def simulate_TT(alpha, beta, x_series):
    N = len(x_series)
    TT = np.zeros(N + 1, dtype=float)
    TT[0] = TT_baseline
    for t in range(1, N + 1):
        drug_effect = y_emax(x_series[t - 1])
        TT[t] = TT[t - 1] + alpha * (TT_baseline - TT[t - 1]) - beta * drug_effect * TT_baseline
    return TT

def beta_to_hit_target(alpha, x_series, TT_target):

    TT0 = simulate_TT(alpha, beta=0.0, x_series=x_series)[-1]
    TT1 = simulate_TT(alpha, beta=1.0, x_series=x_series)[-1]
    return (TT_target - TT0) / (TT1 - TT0)


scenarios = {"24h_half_life (r=0.5)": 0.5, "36h_half_life (r=0.63)": 0.63}


H_med = 10.0
sigma = np.log(2.0) / 1.96  
mu = np.log(H_med)          

N = 20000  
rng = np.random.default_rng(7)
H_samples = rng.lognormal(mean=mu, sigma=sigma, size=N)

H_min, H_max = 3.0, 40.0
H_samples = np.clip(H_samples, H_min, H_max)

alpha_samples = 1 - np.exp(-np.log(2) / H_samples)

days = np.arange(0, 22, dtype=int)

band_rows = []
beta_stats_rows = []

pct_list = [1, 5, 25, 50, 75, 95, 99]

for scen_name, r in scenarios.items():
    for dose, auc21 in auc_day21.items():
        x = exposure_series(auc21, r, days=21)
        TT_target = TT_baseline + delta_TT_day21[dose]

        sup = np.empty((N, 22), dtype=float)
        betas = np.empty(N, dtype=float)

        for i, alpha in enumerate(alpha_samples):
            beta = beta_to_hit_target(alpha, x, TT_target)
            betas[i] = beta
            TT = simulate_TT(alpha, beta, x)
            sup[i, :] = 100.0 * (TT_baseline - TT) / TT_baseline

        percs = np.percentile(sup, pct_list, axis=0)  

        for d_idx, d in enumerate(days):
            row = {"Scenario": scen_name, "Dose_mg_per_day": float(dose), "Day": int(d), "n_mc": int(N)}
            for p_i, p in enumerate(pct_list):
                row[f"suppression_pct_p{p:02d}"] = float(percs[p_i, d_idx])
            band_rows.append(row)

        beta_stats_rows.append({
            "Scenario": scen_name,
            "Dose_mg_per_day": float(dose),
            "n_mc": int(N),
            "H_median_assumed": H_med,
            "H_sigma_logn": float(sigma),
            "H_clip_min": H_min,
            "H_clip_max": H_max,
            "beta_mean": float(betas.mean()),
            "beta_sd": float(betas.std(ddof=1)),
            "beta_p05": float(np.percentile(betas, 5)),
            "beta_p50": float(np.percentile(betas, 50)),
            "beta_p95": float(np.percentile(betas, 95)),
        })

bands_mc = pd.DataFrame(band_rows).sort_values(["Scenario", "Dose_mg_per_day", "Day"])
beta_stats = pd.DataFrame(beta_stats_rows).sort_values(["Scenario", "Dose_mg_per_day"])

display_dataframe_to_user("Monte Carlo beta distribution summary (beta re-fit, day-21 anchored)", beta_stats)

bands_path = "/mnt/data/lgd_toy_pkpd_suppression_bands_mc_alpha_lognormal.csv"
beta_stats_path = "/mnt/data/lgd_toy_pkpd_beta_stats_mc_alpha_lognormal.csv"
bands_mc.to_csv(bands_path, index=False)
beta_stats.to_csv(beta_stats_path, index=False)

for scen_name in scenarios.keys():
    for dose in sorted(auc_day21.keys()):
        g = bands_mc[(bands_mc["Scenario"] == scen_name) & (bands_mc["Dose_mg_per_day"] == float(dose))].sort_values("Day")
        x_day = g["Day"].to_numpy(dtype=float)
        p05 = g["suppression_pct_p05"].to_numpy(dtype=float)
        p50 = g["suppression_pct_p50"].to_numpy(dtype=float)
        p95 = g["suppression_pct_p95"].to_numpy(dtype=float)
        p25 = g["suppression_pct_p25"].to_numpy(dtype=float)
        p75 = g["suppression_pct_p75"].to_numpy(dtype=float)

        plt.figure()
        plt.fill_between(x_day, p05, p95, alpha=0.15, label="5–95% band")
        plt.fill_between(x_day, p25, p75, alpha=0.25, label="25–75% band")
        plt.plot(x_day, p50, label="Median")
        plt.xlabel("Day")
        plt.ylabel("Implied TT suppression (%)")
        plt.title(
            f"Monte Carlo suppression bands (alpha~lognormal(H)) — {dose} mg/day — {scen_name}\n"
            f"H median={H_med}d, spread ~[~5d,~20d] (clipped {H_min}-{H_max}d), beta re-fit to match TT21"
        )
        plt.legend()
        plt.show()

(bands_path, beta_stats_path)

