import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user

# Basaria paper as anchors
TT_baseline = 622.0
auc_day21 = {0.1: 19.0, 0.3: 85.0, 1.0: 238.0}  # ng·24h/mL
delta_TT_day21 = {0.1: -116.0, 0.3: -186.0, 1.0: -282.0}

# Emax model fit
Emax = 0.5099542401801108
EC50 = 42.704725734240566

def y_emax(x):
    return Emax * x / (EC50 + x)

def exposure_series(auc21, r, days=21):
    # Note: x_t is scaled so that x_21 == auc21 using a geometric accumulation shape
    # Note: x_t = auc21 * (1 - r^t) / (1 - r^21), with t starting at 1..days
    t = np.arange(1, days+1)
    x = auc21 * (1 - r**t) / (1 - r**days)
    return t, x

def simulate_TT(alpha, beta, x_series, TT0=TT_baseline):
    # Note: x_series length N for days 1..N, returns TT for day 0..N
    N = len(x_series)
    TT = np.zeros(N+1)
    TT[0] = TT0
    for t in range(1, N+1):
        drug_effect = y_emax(x_series[t-1])  # <--- the fraction
        TT[t] = TT[t-1] + alpha*(TT_baseline - TT[t-1]) - beta*drug_effect*TT_baseline
    return TT

def beta_to_hit_target(alpha, x_series, TT_target):
    # Math: TT is linear in beta; solve via two sims
    TT_b0 = simulate_TT(alpha, beta=0.0, x_series=x_series)[-1]
    TT_b1 = simulate_TT(alpha, beta=1.0, x_series=x_series)[-1]
    # Math: TT(beta) = TT_b0 + beta*(TT_b1 - TT_b0)
    return (TT_target - TT_b0) / (TT_b1 - TT_b0)

# select a "reasonable" daily recovery rate where Half-life ~10 days => alpha = 1 - exp(-ln2/10)
alpha = 1 - np.exp(-np.log(2)/10.0)

scenarios = {
    "24h_half_life (r=0.5)": 0.5,
    "36h_half_life (r=0.63)": 0.63
}

all_rows = []
traj_tables = {}

for scen_name, r in scenarios.items():
    for dose, auc21 in auc_day21.items():
        days, x = exposure_series(auc21, r, days=21)
        TT_target = TT_baseline + delta_TT_day21[dose]
        beta = beta_to_hit_target(alpha, x, TT_target)
        TT = simulate_TT(alpha, beta, x)
        df = pd.DataFrame({
            "Day": np.arange(0, 22),
            "AUC_0_24_exposure_x": np.concatenate([[0.0], x]),
            "y_emax(x)": np.concatenate([[0.0], y_emax(x)]),
            "TT_simulated_ng_dL": TT
        })
        df["Dose_mg_per_day"] = dose
        df["Scenario"] = scen_name
        traj_tables[(scen_name, dose)] = (df, beta)
        all_rows.append({
            "Scenario": scen_name,
            "Dose_mg_per_day": dose,
            "AUC_day21": auc21,
            "Observed_TT_day21": TT_target,
            "Simulated_TT_day21": TT[-1],
            "alpha_per_day": alpha,
            "beta_fitted": beta
        })

summary = pd.DataFrame(all_rows).sort_values(["Scenario", "Dose_mg_per_day"])
display_dataframe_to_user("Toy lag model fit summary (matches day-21 TT)", summary)

# combine the full trajectories into a single table 
full = pd.concat([v[0] for v in traj_tables.values()], ignore_index=True)
csv_path = "/mnt/data/lgd_toy_pkpd_daybyday.csv"
full.to_csv(csv_path, index=False)

#exposure trajectories per each scenario (multiple figs)
for scen_name, r in scenarios.items():
    plt.figure()
    for dose in sorted(auc_day21.keys()):
        df, beta = traj_tables[(scen_name, dose)]
        plt.plot(df["Day"], df["AUC_0_24_exposure_x"], label=f"{dose} mg/d (beta={beta:.3f})")
    plt.xlabel("Day")
    plt.ylabel("AUC_0–24 exposure proxy x (ng·24h/mL)")
    plt.title(f"Exposure accumulation x_t (scaled to Basaria day-21 AUC) — {scen_name}")
    plt.legend()
    plt.show()

# test trajectories per each scenario
for scen_name, r in scenarios.items():
    plt.figure()
    for dose in sorted(auc_day21.keys()):
        df, beta = traj_tables[(scen_name, dose)]
        plt.plot(df["Day"], df["TT_simulated_ng_dL"], label=f"{dose} mg/d (target day21={TT_baseline+delta_TT_day21[dose]:.0f})")
    plt.axhline(TT_baseline, linestyle="--", linewidth=1, label="Baseline 622")
    plt.xlabel("Day")
    plt.ylabel("Total Testosterone TT (ng/dL)")
    plt.title(f"Toy TT trajectory with lag/recovery — {scen_name}")
    plt.legend()
    plt.show()

# add your CSV path 

