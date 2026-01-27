import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user

csv_path = "/mnt/data/lgd_toy_pkpd_daybyday.csv"
df = pd.read_csv(csv_path)

TT_baseline = 622.0

df["Suppression_fraction"] = (TT_baseline - df["TT_simulated_ng_dL"]) / TT_baseline
df["Suppression_percent"] = df["Suppression_fraction"] * 100.0

day21 = df[df["Day"] == 21].copy()
day21_summary = day21[["Scenario", "Dose_mg_per_day", "TT_simulated_ng_dL", "Suppression_fraction", "Suppression_percent"]].sort_values(
    ["Scenario", "Dose_mg_per_day"]
)

display_dataframe_to_user("Implied suppression at Day 21 (toy trajectories)", day21_summary)

for scen in sorted(df["Scenario"].unique()):
    plt.figure()
    sub = df[df["Scenario"] == scen]
    for dose in sorted(sub["Dose_mg_per_day"].unique()):
        s2 = sub[sub["Dose_mg_per_day"] == dose].sort_values("Day")
        plt.plot(s2["Day"], s2["Suppression_percent"], label=f"{dose} mg/day")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Day")
    plt.ylabel("Implied TT suppression (%) vs baseline 622")
    plt.title(f"Implied day-by-day % TT suppression (toy model) — {scen}")
    plt.legend()
    plt.show()

out_path = "/mnt/data/lgd_toy_daybyday_suppression.csv"
df.to_csv(out_path, index=False)

# (exact code run here)
script = f"""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load trajectories
df = pd.read_csv(r"{csv_path}")
TT_baseline = 622.0

# Implied suppression curves
df["Suppression_fraction"] = (TT_baseline - df["TT_simulated_ng_dL"]) / TT_baseline
df["Suppression_percent"] = df["Suppression_fraction"] * 100.0

# Day-21 check table
day21 = df[df["Day"] == 21].copy()
day21_summary = day21[["Scenario","Dose_mg_per_day","TT_simulated_ng_dL","Suppression_fraction","Suppression_percent"]]\
    .sort_values(["Scenario","Dose_mg_per_day"])
print(day21_summary.to_string(index=False))

# Plot suppression curves
for scen in sorted(df["Scenario"].unique()):
    plt.figure()
    sub = df[df["Scenario"] == scen]
    for dose in sorted(sub["Dose_mg_per_day"].unique()):
        s2 = sub[sub["Dose_mg_per_day"] == dose].sort_values("Day")
        plt.plot(s2["Day"], s2["Suppression_percent"], label=f"{{dose}} mg/day")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Day")
    plt.ylabel("Implied TT suppression (%) vs baseline 622")
    plt.title(f"Implied day-by-day % TT suppression (toy model) — {{scen}}")
    plt.legend()
    plt.show()

# Optional export
df.to_csv(r"{out_path}", index=False)
print("Wrote:", r"{out_path}")
"""
print(script)

out_path

