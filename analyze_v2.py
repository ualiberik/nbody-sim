"""
Combined analysis + detective for the LIGHT-DISK simulation (run #2).
Generates side-by-side comparison with the heavy-disk run.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_V1 = Path(r"C:\Projects\nbody-sim\data\20260525_185147")
DATA_V2 = Path(r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_195133")
OUT_DIR = Path(r"C:\Projects\nbody-sim\analysis_output_v2")
OUT_DIR.mkdir(exist_ok=True)

# Mass scale for v2: 1 particle = 0.05 M_earth
# (Note: stats.csv mass_Mearth column uses the OLD scale = 1 M_earth/particle.
#  We need to rescale: real mass = mass_Mearth × 0.05)
MASS_PER_PARTICLE_v2 = 0.05    # M_earth
MASS_PER_PARTICLE_v1 = 1.0     # M_earth (original heavy disk)

print("Loading both stats.csv files ...")
df1 = pd.read_csv(DATA_V1 / "stats.csv")
df2 = pd.read_csv(DATA_V2 / "stats.csv")

# Rescale mass column for true M_earth in each run
df1["true_mass_Mearth"] = df1["n_particles"] * MASS_PER_PARTICLE_v1
df2["true_mass_Mearth"] = df2["n_particles"] * MASS_PER_PARTICLE_v2

print(f"  v1 (heavy disk): {len(df1):,} rows, {df1['frame'].nunique()} frames")
print(f"  v2 (light disk): {len(df2):,} rows, {df2['frame'].nunique()} frames")

groups2 = {fr: g for fr, g in df2.groupby("frame")}
frames2 = sorted(groups2.keys())
last_frame_v2 = frames2[-1]

# ----------------------------------------------------------------------
# 1. Side-by-side aggregate count + champion mass curves
# ----------------------------------------------------------------------
per1 = df1.groupby("frame").agg(yrs=("sim_time_years","first"),
                                  n_agg=("agg_id","count"),
                                  max_mass=("true_mass_Mearth","max")).reset_index()
per2 = df2.groupby("frame").agg(yrs=("sim_time_years","first"),
                                  n_agg=("agg_id","count"),
                                  max_mass=("true_mass_Mearth","max")).reset_index()

fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
ax[0].plot(per1["yrs"], per1["n_agg"], color="crimson", lw=0.7, label="v1: heavy disk (10000 M⊕)")
ax[0].plot(per2["yrs"], per2["n_agg"], color="navy",    lw=0.7, label="v2: light disk (500 M⊕)")
ax[0].set_ylabel("Aggregate count")
ax[0].set_title("Aggregate count: heavy vs light disk")
ax[0].grid(alpha=0.3); ax[0].legend()

ax[1].plot(per1["yrs"], per1["max_mass"], color="crimson", lw=0.7, label="v1")
ax[1].plot(per2["yrs"], per2["max_mass"], color="navy",    lw=0.7, label="v2")
ax[1].set_ylabel("Heaviest body (M⊕, true scale)")
ax[1].set_xlabel("Time (years)")
ax[1].set_yscale("log")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "01_compare_growth.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# 2. Final state of v2: top 20 by true mass
# ----------------------------------------------------------------------
final = df2[df2["frame"] == last_frame_v2].copy()
final = final.sort_values("true_mass_Mearth", ascending=False).reset_index(drop=True)
final["r_cyl"]   = np.sqrt(final["cx_AU"]**2 + final["cz_AU"]**2)
final["period_yr"] = final["r_cyl"]**1.5

print(f"\n=== v2 FINAL STATE (t={final['sim_time_years'].iloc[0]:.2f} yr) ===")
print(f"  Total aggregates: {len(final)}")
print(f"  Total mass in aggregates: {final['true_mass_Mearth'].sum():.1f} M⊕")
print(f"  Top 15 bodies:")
top = final.head(15)[["n_particles", "true_mass_Mearth", "r_cyl", "period_yr"]]
top.columns = ["n_part", "mass_Mearth", "r_AU", "T_yr"]
print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

final.head(50).to_csv(OUT_DIR / "v2_final_top50.csv", index=False)

# ----------------------------------------------------------------------
# 3. Architecture comparison: planet-class bodies (>= 1 M_earth true mass)
# ----------------------------------------------------------------------
planets_v2 = final[final["true_mass_Mearth"] >= 1.0].copy()
planets_v2 = planets_v2.sort_values("r_cyl").reset_index(drop=True)

# Same for v1 final state
last_frame_v1 = df1["frame"].max()
fv1 = df1[df1["frame"] == last_frame_v1].copy()
fv1["r_cyl"] = np.sqrt(fv1["cx_AU"]**2 + fv1["cz_AU"]**2)
planets_v1 = fv1[fv1["true_mass_Mearth"] >= 1.0].sort_values("r_cyl").reset_index(drop=True)

print(f"\n=== Planet-class bodies (>= 1 M⊕) ===")
print(f"  v1 (heavy): {len(planets_v1)} planets")
print(f"  v2 (light): {len(planets_v2)} planets")

# Period-ratio chain for v2
fig, ax = plt.subplots(2, 1, figsize=(12, 9))
# Top: positions on log radius axis
for i, p in planets_v1.iterrows():
    size = np.clip(p["true_mass_Mearth"] * 1.5, 8, 400)
    ax[0].scatter(p["r_cyl"], 1, s=size, color="crimson", alpha=0.6, edgecolors="black")
for i, p in planets_v2.iterrows():
    size = np.clip(p["true_mass_Mearth"] * 1.5, 8, 400)
    ax[0].scatter(p["r_cyl"], 0, s=size, color="navy", alpha=0.6, edgecolors="black")
ax[0].set_yticks([0, 1])
ax[0].set_yticklabels(["v2 light", "v1 heavy"])
ax[0].set_xlabel("Orbital radius (AU)")
ax[0].set_title(f"Final architecture (size ∝ mass): {len(planets_v1)} vs {len(planets_v2)} planets ≥1 M⊕")
ax[0].set_xscale("log")
ax[0].set_xlim(0.3, 10)
ax[0].grid(alpha=0.3, which="both")

# Bottom: period ratio of consecutive planet pairs in v2
if len(planets_v2) > 1:
    ratios = planets_v2["period_yr"].values[1:] / planets_v2["period_yr"].values[:-1]
    ax[1].scatter(np.arange(len(ratios)), ratios, c="navy", s=50)
    for ref, lbl in [(2.0,"2:1"),(1.5,"3:2"),(1.333,"4:3"),(1.25,"5:4"),
                     (1.4,"7:5"),(1.667,"5:3"),(1.2,"6:5"),(1.6,"8:5")]:
        ax[1].axhline(ref, ls="--", alpha=0.4, color="red")
        ax[1].text(len(ratios)-0.3, ref, f" {lbl}", va="center", color="red", fontsize=9)
    ax[1].set_xlabel("planet-pair index (inner -> outer)")
    ax[1].set_ylabel("T_outer / T_inner")
    ax[1].set_title("v2 period-ratio chain")
    ax[1].set_ylim(1.0, max(3.0, ratios.max()*1.05))
    ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "02_architecture_compare.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 4. Mass concentration: what fraction is in the largest body?
# ----------------------------------------------------------------------
total_v1 = fv1["true_mass_Mearth"].sum()
total_v2 = final["true_mass_Mearth"].sum()
biggest_v1 = fv1["true_mass_Mearth"].max()
biggest_v2 = final["true_mass_Mearth"].max()
print(f"\n=== Mass concentration ===")
print(f"  v1: biggest = {biggest_v1:.1f} M⊕  /  total = {total_v1:.1f} M⊕  → {100*biggest_v1/total_v1:.1f} %")
print(f"  v2: biggest = {biggest_v2:.1f} M⊕  /  total = {total_v2:.1f} M⊕  → {100*biggest_v2/total_v2:.1f} %")
print(f"  Solar System reference: Jupiter / total ≈ 71 %")

# ----------------------------------------------------------------------
# 5. Final disk top-down view for v2
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# v1 reference
sc1 = axes[0].scatter(fv1["cx_AU"], fv1["cz_AU"],
                      s=np.clip(fv1["n_particles"]*0.3, 1, 200),
                      c=np.log10(fv1["true_mass_Mearth"].clip(lower=0.01)),
                      cmap="plasma", alpha=0.7)
axes[0].plot(0, 0, marker="*", color="gold", markersize=18, markeredgecolor="black")
axes[0].set_aspect("equal"); axes[0].set_xlim(-8, 8); axes[0].set_ylim(-8, 8)
axes[0].set_title(f"v1 heavy disk — {len(planets_v1)} planets ≥1 M⊕")
axes[0].set_xlabel("x (AU)"); axes[0].set_ylabel("z (AU)")
plt.colorbar(sc1, ax=axes[0], label="log₁₀ M⊕")

# v2 light
sc2 = axes[1].scatter(final["cx_AU"], final["cz_AU"],
                      s=np.clip(final["n_particles"]*2.0, 1, 200),
                      c=np.log10(final["true_mass_Mearth"].clip(lower=0.01)),
                      cmap="plasma", alpha=0.7)
axes[1].plot(0, 0, marker="*", color="gold", markersize=18, markeredgecolor="black")
axes[1].set_aspect("equal"); axes[1].set_xlim(-8, 8); axes[1].set_ylim(-8, 8)
axes[1].set_title(f"v2 light disk — {len(planets_v2)} planets ≥1 M⊕")
axes[1].set_xlabel("x (AU)"); axes[1].set_ylabel("z (AU)")
plt.colorbar(sc2, ax=axes[1], label="log₁₀ M⊕")

plt.tight_layout()
plt.savefig(OUT_DIR / "03_topdown_compare.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 6. Disk heating comparison
# ----------------------------------------------------------------------
def disk_h(df, frame_step=20):
    h_list = []
    fr_list = sorted(df["frame"].unique())
    for fr in fr_list[::frame_step]:
        g = df[df["frame"] == fr]
        w = g["n_particles"].values.astype(float)
        h = np.sqrt(np.average(g["cy_AU"].values**2, weights=w))
        h_list.append((g["sim_time_years"].iloc[0], h))
    return np.array(h_list).T

t1, h1 = disk_h(df1)
t2, h2 = disk_h(df2)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(t1, h1, color="crimson", lw=1.2, label=f"v1 heavy: 0.010 → {h1[-1]:.3f} AU (×{h1[-1]/h1[0]:.0f})")
ax.plot(t2, h2, color="navy",    lw=1.2, label=f"v2 light: {h2[0]:.4f} → {h2[-1]:.3f} AU (×{h2[-1]/h2[0]:.0f})")
ax.set_xlabel("Time (yr)"); ax.set_ylabel("RMS vertical thickness (AU)")
ax.set_title("Disk vertical heating: heavy vs light")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "04_disk_heating_compare.png", dpi=120)
plt.close()
print(f"\n=== Disk heating ===")
print(f"  v1: 0.0100 → {h1[-1]:.4f} AU  (× {h1[-1]/h1[0]:.0f})")
print(f"  v2: {h2[0]:.4f} → {h2[-1]:.4f} AU  (× {h2[-1]/h2[0]:.0f})")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print(f"\n=== SUMMARY ===")
print(f"  v1 final:  {len(planets_v1):>3} planets ≥1 M⊕,  biggest = {biggest_v1:>7.1f} M⊕,  concentration = {100*biggest_v1/total_v1:.1f}%")
print(f"  v2 final:  {len(planets_v2):>3} planets ≥1 M⊕,  biggest = {biggest_v2:>7.1f} M⊕,  concentration = {100*biggest_v2/total_v2:.1f}%")

print(f"\nPlots written to: {OUT_DIR}")
