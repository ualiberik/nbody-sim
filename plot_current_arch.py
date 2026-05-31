"""Plot current orbital architecture of the running simulation."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path(r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv")
OUT_DIR = Path(r"C:\Projects\nbody-sim\analysis_output_v3")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV)
df["mass_Mearth"] = df["n_particles"].astype(float)
df["r_cyl"] = np.sqrt(df["cx_AU"]**2 + df["cz_AU"]**2)

last_complete = df["frame"].max() - 1
snap = df[df["frame"] == last_complete].copy()
snap = snap.sort_values("mass_Mearth", ascending=False).reset_index(drop=True)
t_yr = snap["sim_time_years"].iloc[0]
t_T0 = snap["sim_time_T0"].iloc[0]

# Also load the previous run for comparison
DATA_V1 = Path(r"C:\Projects\nbody-sim\data\20260525_185147")
df1 = pd.read_csv(DATA_V1 / "stats.csv")
df1["mass_Mearth"] = df1["n_particles"].astype(float)
df1["r_cyl"] = np.sqrt(df1["cx_AU"]**2 + df1["cz_AU"]**2)
last_v1 = df1[df1["frame"] == df1["frame"].max()].copy()

# ----------------------------------------------------------------------
# FIG 1: SIDE-BY-SIDE top-down view  (v1 final vs current snapshot)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# v1 — 159 yr
sc1 = axes[0].scatter(last_v1["cx_AU"], last_v1["cz_AU"],
                      s=np.clip(last_v1["mass_Mearth"]*0.3, 1, 400),
                      c=np.log10(last_v1["mass_Mearth"].clip(lower=0.5)),
                      cmap="plasma", alpha=0.75, edgecolors="none")
axes[0].plot(0, 0, marker="*", color="gold", markersize=24, markeredgecolor="black", zorder=10)
axes[0].set_aspect("equal"); axes[0].set_xlim(-8, 8); axes[0].set_ylim(-8, 8)
axes[0].set_title(f"v1: t = 159 yr  ({len(last_v1)} aggregates)")
axes[0].set_xlabel("x (AU)"); axes[0].set_ylabel("z (AU)")
plt.colorbar(sc1, ax=axes[0], label="log₁₀ M⊕")
axes[0].grid(alpha=0.2)

# Current snapshot
sc2 = axes[1].scatter(snap["cx_AU"], snap["cz_AU"],
                      s=np.clip(snap["mass_Mearth"]*0.3, 1, 400),
                      c=np.log10(snap["mass_Mearth"].clip(lower=0.5)),
                      cmap="plasma", alpha=0.75, edgecolors="none")
axes[1].plot(0, 0, marker="*", color="gold", markersize=24, markeredgecolor="black", zorder=10)
axes[1].set_aspect("equal"); axes[1].set_xlim(-15, 15); axes[1].set_ylim(-15, 15)
axes[1].set_title(f"v3: t = {t_yr:.0f} yr  ({len(snap)} aggregates)")
axes[1].set_xlabel("x (AU)"); axes[1].set_ylabel("z (AU)")
plt.colorbar(sc2, ax=axes[1], label="log₁₀ M⊕")
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_topdown_compare.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# FIG 2: Zoom-out current snapshot showing Sedna-analog at 57 AU
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 11))
sc = ax.scatter(snap["cx_AU"], snap["cz_AU"],
                s=np.clip(snap["mass_Mearth"]*0.6, 2, 500),
                c=np.log10(snap["mass_Mearth"].clip(lower=0.5)),
                cmap="plasma", alpha=0.75, edgecolors="none")
ax.plot(0, 0, marker="*", color="gold", markersize=24, markeredgecolor="black", zorder=10)
ax.set_aspect("equal"); ax.set_xlim(-65, 65); ax.set_ylim(-65, 65)
ax.set_title(f"Wide view at t = {t_yr:.0f} yr — note Sedna-analog at r ≈ 57 AU!")
ax.set_xlabel("x (AU)"); ax.set_ylabel("z (AU)")
plt.colorbar(sc, ax=ax, label="log₁₀ M⊕")
ax.grid(alpha=0.2)

# Highlight the ejected body
ejected = snap[snap["r_cyl"] > 30]
for _, row in ejected.iterrows():
    ax.annotate(f"  {row['mass_Mearth']:.0f} M⊕", (row["cx_AU"], row["cz_AU"]),
                fontsize=11, color="red", fontweight="bold")
    ax.plot(row["cx_AU"], row["cz_AU"], marker="o", mfc="none", mec="red", ms=20, mew=2)

plt.tight_layout()
plt.savefig(OUT_DIR / "02_wide_view_sedna.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# FIG 3: Radial distribution histogram  (planet class only ≥1 M⊕)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# Bodies ≥1 M⊕
planets = snap[snap["mass_Mearth"] >= 1].copy()
gas_giants = snap[snap["mass_Mearth"] >= 30].copy()

axes[0].scatter(planets["r_cyl"], planets["mass_Mearth"],
                s=np.clip(planets["mass_Mearth"]*0.5, 4, 400),
                c=np.log10(planets["mass_Mearth"]),
                cmap="plasma", alpha=0.75, edgecolors="none")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlim(0.5, 100); axes[0].set_ylim(0.5, 1000)
axes[0].set_xlabel("Orbital radius (AU)"); axes[0].set_ylabel("Mass (M⊕)")
axes[0].set_title(f"Mass vs orbital radius — {len(planets)} bodies ≥ 1 M⊕  (t = {t_yr:.0f} yr)")
axes[0].grid(alpha=0.3, which="both")

# Reference: Solar system planets
ss = [("Mercury",0.39,0.055),("Venus",0.72,0.82),("Earth",1.0,1.0),("Mars",1.52,0.107),
      ("Jupiter",5.2,318),("Saturn",9.5,95),("Uranus",19.2,14.5),("Neptune",30.1,17)]
for name, r, m in ss:
    axes[0].plot(r, m, marker="*", color="black", ms=14, mfc="none", mew=1.5)
    axes[0].annotate(f" {name}", (r, m), fontsize=8, color="black")

# Histogram of radii for >5 M⊕ bodies, log bins
bins = np.logspace(np.log10(0.3), np.log10(80), 40)
big = snap[snap["mass_Mearth"] >= 5]
medium = snap[(snap["mass_Mearth"] >= 1) & (snap["mass_Mearth"] < 5)]
axes[1].hist(medium["r_cyl"], bins=bins, color="lightblue", alpha=0.6, label=f"1-5 M⊕ ({len(medium)})", edgecolor="navy")
axes[1].hist(big["r_cyl"],    bins=bins, color="orange",    alpha=0.7, label=f"≥5 M⊕ ({len(big)})", edgecolor="darkred")
axes[1].set_xscale("log")
axes[1].set_xlim(0.3, 80)
axes[1].set_xlabel("Orbital radius (AU)")
axes[1].set_ylabel("Count")
axes[1].set_title("Radial distribution of bodies")
axes[1].legend()
axes[1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(OUT_DIR / "03_mass_radius.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# FIG 4: Period-ratio chain for gas giants (≥30 M⊕)
# ----------------------------------------------------------------------
heavy = snap[snap["mass_Mearth"] >= 30].copy()
heavy = heavy.sort_values("r_cyl").reset_index(drop=True)
heavy["T"] = heavy["r_cyl"]**1.5

print(f"\n=== Gas giants (≥30 M⊕) at t={t_yr:.0f} yr ===")
print(heavy[["agg_id","mass_Mearth","r_cyl","T"]].to_string(index=False))

if len(heavy) > 1:
    ratios = heavy["T"].values[1:] / heavy["T"].values[:-1]
    print(f"\nConsecutive period ratios:")
    for i, r in enumerate(ratios):
        print(f"  pair {i}: r_in={heavy['r_cyl'].iloc[i]:.3f}, r_out={heavy['r_cyl'].iloc[i+1]:.3f}, T_out/T_in={r:.3f}")

# Architecture plot: position of each gas giant
fig, ax = plt.subplots(figsize=(13, 5))
for i, row in heavy.iterrows():
    size = np.clip(row["mass_Mearth"] * 2, 30, 800)
    ax.scatter(row["r_cyl"], 0, s=size,
               c=[np.log10(row["mass_Mearth"])],
               cmap="plasma", vmin=1, vmax=2.7,
               alpha=0.8, edgecolors="black", linewidth=1)
    ax.annotate(f"{row['mass_Mearth']:.0f}", (row["r_cyl"], 0),
                fontsize=10, ha="center", va="center", fontweight="bold")
ax.set_xlim(2.5, 9)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_xlabel("Orbital radius (AU)")
ax.set_title(f"Gas giants (≥30 M⊕) architecture at t = {t_yr:.0f} yr — {len(heavy)} bodies")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_gas_giants_chain.png", dpi=110)
plt.close()

print(f"\nPlots saved to {OUT_DIR}")
