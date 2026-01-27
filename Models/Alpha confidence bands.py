Always show details
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
    x = auc21 * (1 - r**t) / (1 - r**days)
    return x  

def simulate_TT(alpha, beta, x_series, TT0=TT_baseline):
    N = len(x_series)
    TT = np.zeros(N + 1)
    TT[0] = TT0
    for t in range(1, N + 1):
        drug_effect = y_emax(x_series[t - 1])  
        TT[t] = TT[t - 1] + alpha * (TT_baseline - TT[t - 1]) - beta * drug_effect * TT_baseline
    return TT

def beta_to_hit_target(alpha, x_series, TT_target):
    TT_b0 = simulate_TT(alpha, beta=0.0, x_series=x_series)[-1]
    TT_b1 = simulate_TT(alpha, beta=1.0, x_series=x_series)[-1]
    return (TT_target - TT_b0) / (TT_b1 - TT_b0)

scenarios = {
    "24h_half_life (r=0.5)": 0.5,
    "36h_half_life (r=0.63)": 0.63
}

# alpha uncertainty grid (via recovery half-life H days)
H_grid = np.array([5, 7, 10, 14, 20], dtype=float)
alpha_grid = 1 - np.exp(-np.log(2) / H_grid)

days = np.arange(0, 22)
band_rows = []
fit_rows = []

for scen_name, r in scenarios.items():
    for dose, auc21 in auc_day21.items():
        x = exposure_series(auc21, r, days=21)
        TT_target = TT_baseline + delta_TT_day21[dose]

        sup_curves = []
        betas = []
        for alpha in alpha_grid:
            beta = beta_to_hit_target(alpha, x, TT_target)
            betas.append(beta)
            TT = simulate_TT(alpha, beta, x)
            suppression_pct = 100.0 * (TT_baseline - TT) / TT_baseline
            sup_curves.append(suppression_pct)

        sup = np.vstack(sup_curves)  # (n_alpha, 22)
        p05 = np.percentile(sup, 5, axis=0)
        p50 = np.percentile(sup, 50, axis=0)
        p95 = np.percentile(sup, 95, axis=0)
        smin = sup.min(axis=0)
        smax = sup.max(axis=0)

        for d, lo, mid, hi, mn, mx in zip(days, p05, p50, p95, smin, smax):
            band_rows.append({
                "Scenario": scen_name,
                "Dose_mg_per_day": dose,
                "Day": int(d),
                "suppression_pct_p05": float(lo),
                "suppression_pct_p50": float(mid),
                "suppression_pct_p95": float(hi),
                "suppression_pct_min": float(mn),
                "suppression_pct_max": float(mx),
                "n_alpha": int(len(alpha_grid))
            })

        for H, alpha, beta in zip(H_grid, alpha_grid, betas):
            fit_rows.append({
                "Scenario": scen_name,
                "Dose_mg_per_day": dose,
                "recovery_half_life_days_H": float(H),
                "alpha_per_day": float(alpha),
                "beta_fitted": float(beta),
                "TT21_target": float(TT_target)
            })

bands = pd.DataFrame(band_rows).sort_values(["Scenario", "Dose_mg_per_day", "Day"])
fits = pd.DataFrame(fit_rows).sort_values(["Scenario", "Dose_mg_per_day", "recovery_half_life_days_H"])

display_dataframe_to_user("Alpha-sensitivity: beta re-fit (hits TT21 for each alpha)", fits)

bands_path = "/mnt/data/lgd_toy_pkpd_suppression_bands_by_alpha.csv"
fits_path = "/mnt/data/lgd_toy_pkpd_beta_fits_by_alpha.csv"
bands.to_csv(bands_path, index=False)
fits.to_csv(fits_path, index=False)

for scen_name in scenarios.keys():
    for dose in sorted(auc_day21.keys()):
        g = bands[(bands["Scenario"] == scen_name) & (bands["Dose_mg_per_day"] == dose)].sort_values("Day")
        plt.figure()
        plt.fill_between(g["Day"], g["suppression_pct_p05"], g["suppression_pct_p95"], alpha=0.2, label="5–95% band (alpha grid)")
        plt.plot(g["Day"], g["suppression_pct_p50"], label="Median")
        plt.xlabel("Day")
        plt.ylabel("Implied TT suppression (%)")
        plt.title(
            f"Suppression band — {dose} mg/day — {scen_name}\n"
            f"alpha from recovery half-life H ∈ {list(H_grid.astype(int))} days (beta re-fit to match TT21)"
        )
        plt.legend()
        plt.show()

(bands_path, fits_path)
