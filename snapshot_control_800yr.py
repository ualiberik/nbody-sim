"""Extract control-sim snapshot near t=800 yr for fair comparison with exp 1."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import pandas as pd
from pathlib import Path

STATS = Path(r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv")
OUT = Path(r"C:\Projects\nbody-sim\analysis_control")

# Target: t = 800 yr = 5026.5 T0
TARGET_T0 = 5026.5
TOL = 0.1  # accept frames within this T0 window

print(f"Streaming for frames near t = {TARGET_T0} T0 (~800 yr) ...")
dtypes = {
    "frame": np.int32, "sim_time_T0": np.float32, "n_aggregates": np.int32,
    "agg_id": np.int32, "n_particles": np.int32,
    "mass_Mearth": np.float32,
    "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32,
    "vx": np.float32, "vy": np.float32, "vz": np.float32,
    "dist_center_AU": np.float32,
}
collected = []
t0 = time.time()
last_log = time.time()
n_rows = 0
done = False
for chunk in pd.read_csv(STATS, dtype=dtypes, chunksize=1_000_000, usecols=list(dtypes.keys())):
    n_rows += len(chunk)
    in_window = chunk[(chunk["sim_time_T0"] >= TARGET_T0 - TOL) & (chunk["sim_time_T0"] <= TARGET_T0 + TOL)]
    if len(in_window):
        collected.append(in_window)
        # If we've passed the window, we can stop
        if chunk["sim_time_T0"].max() > TARGET_T0 + TOL:
            done = True
            break
    if time.time() - last_log > 5:
        last_log = time.time()
        print(f"  ... {n_rows:,} rows, current t={chunk['sim_time_T0'].iloc[-1]:.1f} T0, elapsed {time.time()-t0:.0f}s")

snap = pd.concat(collected, ignore_index=True)
# Find single closest frame
target_frame = snap.loc[(snap["sim_time_T0"] - TARGET_T0).abs().idxmin(), "frame"]
snap = snap[snap["frame"] == target_frame].copy()
snap["mass_Mearth"] = snap["n_particles"].astype(float)
snap["r_cyl"] = np.sqrt(snap["cx_AU"]**2 + snap["cz_AU"]**2)
snap = snap.sort_values("mass_Mearth", ascending=False).reset_index(drop=True)

t_yr = snap["sim_time_T0"].iloc[0] * 0.15915494
print(f"\n=== CONTROL SNAPSHOT at frame {target_frame} (t={snap['sim_time_T0'].iloc[0]:.1f} T0 = {t_yr:.1f} yr) ===")
print(f"Total aggregates: {len(snap)}")
print(f"Total mass: {snap['mass_Mearth'].sum():.0f} M_earth")
print()

classes = [
    ("Super-Jupiter (300-1000 M⊕)", 300, 1000),
    ("Jupiter-like (95-300 M⊕)",    95,  300),
    ("Saturn-like (30-95 M⊕)",      30,   95),
    ("Neptune/Uranus (10-30 M⊕)",   10,   30),
    ("Super-Earth (3-10 M⊕)",        3,   10),
    ("Earth-like (1-3 M⊕)",          1,    3),
]
print(f"{'Class':<35} {'count':>6}  {'total':>10}  {'biggest':>9}")
for name, lo, hi in classes:
    sub = snap[(snap["mass_Mearth"] >= lo) & (snap["mass_Mearth"] < hi)]
    print(f"{name:<35} {len(sub):>6}  {sub['mass_Mearth'].sum():>8.1f} M⊕  "
          f"{sub['mass_Mearth'].max() if len(sub) else 0:>7.1f} M⊕")

print()
print("Top 15 bodies:")
top = snap.head(15)[["agg_id","n_particles","mass_Mearth","r_cyl"]].copy()
top.columns = ["agg_id","particles","mass_M⊕","r_AU"]
print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print()
print(f"Gas giants ≥30 M⊕: {(snap['mass_Mearth'] >= 30).sum()}")
giants = snap[snap["mass_Mearth"] >= 30].sort_values("r_cyl")
for _, g in giants.iterrows():
    print(f"  {g['mass_Mearth']:>5.0f} M⊕  at r={g['r_cyl']:6.3f} AU")

# Save
snap.head(50).to_csv(OUT / "snapshot_800yr_top50.csv", index=False)
print(f"\nSaved top-50 to {OUT}/snapshot_800yr_top50.csv")

# Ejected
print(f"\nBodies on r > 10 AU: {(snap['r_cyl'] > 10).sum()}")
print(f"Bodies on r > 30 AU: {(snap['r_cyl'] > 30).sum()}")
print(f"Most distant: r = {snap['r_cyl'].max():.1f} AU")
