import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user

#Since my notes are kind of sloppy I had chatgpt remake this neater and add instructions for your convenience :) 
# - Nate

# ---- Inputs ----
csv_in = "/mnt/data/lgd_toy_pkpd_daybyday.csv"
TT_baseline = 622.0

# ---- Load trajectories ----
df = pd.read_csv(csv_in)

# ---- Compute implied suppression ----
df["suppression_frac"] = (TT_baseline - df["TT_simulated_ng_dL"]) / TT_baseline
df["suppression_pct"] = 100.0 * df["suppression_frac"]

# ---- Save augmented file ----
csv_out = "/mnt/data/lgd_toy_pkpd_daybyday_with_suppression.csv"
df.to_csv(csv_out, index=False)

# ---- Quick sanity summary at day 21 ----
day21 = df[df["Day"] == 21].copy()
day21_summary = day21[
    ["Scenario", "Dose_mg_per_day", "TT_simulated_ng_dL", "suppression_pct", "AUC_0_24_exposure_x"]
].sort_values(["Scenario", "Dose_mg_per_day"])
display_dataframe_to_user("Day-21 implied suppression (toy trajectories)", day21_summary)

# ---- Plots: suppression % vs day (one fig per scenario) ----
for scen, g in df.groupby("Scenario"):
    plt.figure()
    for dose in sorted(g["Dose_mg_per_day"].unique()):
        gd = g[g["Dose_mg_per_day"] == dose].sort_values("Day")
        plt.plot(gd["Day"], gd["suppression_pct"], label=f"{dose} mg/d")
    plt.xlabel("Day")
    plt.ylabel("Implied TT suppression (%)")
    plt.title(f"Implied day-by-day TT suppression — {scen}")
    plt.legend()
    plt.show()

# ---- Plots: suppression % vs exposure x (diagnostic trajectory) ----
for scen, g in df.groupby("Scenario"):
    plt.figure()
    for dose in sorted(g["Dose_mg_per_day"].unique()):
        gd = g[g["Dose_mg_per_day"] == dose].sort_values("Day")
        plt.plot(gd["AUC_0_24_exposure_x"], gd["suppression_pct"], label=f"{dose} mg/d")
    plt.xlabel("Exposure proxy x (AUC_0–24)")
    plt.ylabel("Implied TT suppression (%)")
    plt.title(f"Suppression vs exposure trajectory — {scen}")
    plt.legend()
    plt.show()

print("Wrote:", csv_out)

