#3D appendix — time (x), dose (y), suppression (z)

AUC21 = np.array([19.0, 85.0, 238.0])
y21 = np.array([116/622, 186/622, 282/622], dtype=float)

E_grid = np.linspace(max(y21), 0.99, 400)  
EC_grid = np.linspace(1e-3, 800.0, 800)    
best = None
best_params = None
for E in E_grid:

    EC = EC_grid
    pred = (E * AUC21[None,:]) / (EC[:,None] + AUC21[None,:])
    sse = ((pred - y21[None,:])**2).sum(axis=1)
    idx = int(np.argmin(sse))
    sse_min = float(sse[idx])
    if (best is None) or (sse_min < best):
        best = sse_min
        best_params = (float(E), float(EC[idx]))

Emax_hat, EC50_hat = best_params

def y_of_x(x):
    x = np.asarray(x, dtype=float)
    return (Emax_hat * x) / (EC50_hat + x)

def AUC21_of_dose(dose_mg_day):
    return 238.0 * np.asarray(dose_mg_day, dtype=float)

k_lag = 0.12  
def f_time(t):
    t = np.asarray(t, dtype=float)
    num = 1 - np.exp(-k_lag*np.minimum(t,21))
    den = 1 - np.exp(-k_lag*21)
    return num/den

t_grid = np.linspace(0, 56, 57)          
dose_grid = np.linspace(0.05, 10.0, 80)  
T, D = np.meshgrid(t_grid, dose_grid)
X = AUC21_of_dose(D) * f_time(T)        
Z = 100.0 * y_of_x(X)                    

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(T, D, Z, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.35)

clinical_doses = np.array([0.1,0.3,1.0])
clinical_supp = 100.0 * y21
ax.scatter(np.full_like(clinical_doses,21.0), clinical_doses, clinical_supp, s=50)
for d, z in zip(clinical_doses, clinical_supp):
    ax.text(21.5, float(d), float(z)+1.0, f"{d:g}mg @21d", fontsize=8)

for _, r in endpoints2.iterrows():
    if not str(r["Series"]).startswith("Anecdote"):
        continue
    dose = float(r["Dose_mg_day"])
    t_end = float(r["Duration_days"])
    z_end = float(r["Suppression_end_pct"])
    ax.plot([0, t_end], [dose, dose], [0, z_end], linewidth=2)
    ax.scatter([t_end],[dose],[z_end], s=35)
    ax.text(t_end+1.0, dose, z_end+1.0, str(r["Series"]).split(":")[0].replace("Anecdote ","A"), fontsize=8)

ax.set_title("LGD-4033: Dose–Time–Suppression 3D (clinical-constrained illustrative surface + anecdote paths)")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Dose (mg/day)")
ax.set_zlabel("Total testosterone suppression (%)")

out2 = "/mnt/data/lgd_dose_time_suppression_3d_appendix.png"
fig.tight_layout()
fig.savefig(out2, dpi=200)
plt.close(fig)

out2_csv = "/mnt/data/lgd_dose_time_suppression_surface_grid.csv"
pd.DataFrame({
    "day": T.ravel(),
    "dose_mg_day": D.ravel(),
    "suppression_pct": Z.ravel()
}).to_csv(out2_csv, index=False)

out2, out2_csv, (Emax_hat, EC50_hat)

