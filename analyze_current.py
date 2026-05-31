"""Quick snapshot of body-mass distribution from the LIVE-running simulation."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

CSV = Path(r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv")

# Read the file — last frame may be incomplete, so we keep only complete frames
df = pd.read_csv(CSV)
# 1 particle = 1 M_earth (heavy disk, super-Jupiter config)
df["mass_Mearth"] = df["n_particles"].astype(float)

last_complete = df["frame"].max() - 1   # drop possibly-partial last frame
snap = df[df["frame"] == last_complete].copy()
snap = snap.sort_values("mass_Mearth", ascending=False).reset_index(drop=True)

t_yr = snap["sim_time_years"].iloc[0]
t_T0 = snap["sim_time_T0"].iloc[0]

print(f"=== Snapshot at frame {last_complete} (t = {t_T0:.1f} T0 = {t_yr:.2f} yr) ===")
print(f"Total aggregates: {len(snap)}")
print(f"Total mass: {snap['mass_Mearth'].sum():.1f} M_earth")
print()

# Mass classification
classes = [
    ("Star-incinerating (>1000 M⊕, > 3 M_jup)",  1000, 1e10),
    ("Super-Jupiter (300-1000 M⊕)",               300,  1000),
    ("Jupiter-like (95-300 M⊕)",                   95,  300),
    ("Saturn-like (30-95 M⊕)",                     30,  95),
    ("Neptune/Uranus-like (10-30 M⊕)",             10,  30),
    ("Super-Earth (3-10 M⊕)",                       3,  10),
    ("Earth-Venus-Mars (1-3 M⊕)",                   1,  3),
    ("Mercury-like / large asteroid (0.1-1 M⊕)",  0.1,  1),
    ("Small asteroid / dust (< 0.1 M⊕)",          -1,  0.1),
]

print(f"{'Class':<45} {'count':>6}  {'total mass':>11}  {'biggest':>9}")
print("-" * 80)
for name, lo, hi in classes:
    sub = snap[(snap["mass_Mearth"] >= lo) & (snap["mass_Mearth"] < hi)]
    if len(sub) == 0:
        print(f"{name:<45} {0:>6}  {'-':>11}  {'-':>9}")
    else:
        print(f"{name:<45} {len(sub):>6}  {sub['mass_Mearth'].sum():>9.1f} M⊕  {sub['mass_Mearth'].max():>7.1f} M⊕")
print("-" * 80)

print()
print("=== Top 15 bodies by mass ===")
top = snap.head(15).copy()
top["r_cyl"] = np.sqrt(top["cx_AU"]**2 + top["cz_AU"]**2)
top["v"]     = np.sqrt(top["vx"]**2 + top["vy"]**2 + top["vz"]**2)
out = top[["agg_id","n_particles","mass_Mearth","r_cyl","v"]].copy()
out.columns = ["agg_id","particles","mass_M⊕","r_AU","|v|"]
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Spatial distribution by mass class
print()
print("=== Radial distribution of >5 M⊕ bodies ===")
heavy = snap[snap["mass_Mearth"] >= 5].copy()
heavy["r_cyl"] = np.sqrt(heavy["cx_AU"]**2 + heavy["cz_AU"]**2)
bins = [0, 1, 2, 3, 4, 5, 6, 8, 12, 20]
heavy["r_bin"] = pd.cut(heavy["r_cyl"], bins)
agg = heavy.groupby("r_bin", observed=True).agg(count=("mass_Mearth","size"),
                                                 total_mass=("mass_Mearth","sum"),
                                                 biggest=("mass_Mearth","max"))
print(agg.to_string())

# Comparison hint vs original 1000 T0 run
print()
print("=== Comparison with previous 1000 T0 run (super-Jupiter run) ===")
print("  Previous final (t=1000 T0):")
print("    - 733 aggregates total")
print("    - 32 planet-class bodies (≥5 M⊕)")
print("    - biggest = 237 M⊕ at r=5.45 AU")
print("    - 3 'planets' >100 M⊕")
