"""
H1 test: Does the inner-planet count drop non-linearly with perturber mass?
We define "inner planets" using several thresholds to be robust.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

RUNS = [
    ("control_800",   0.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv", 5026.5),
    ("exp1_03Mjup",  0.3,   r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_000833\stats.csv", 5000.0),
    ("exp2_10Mjup",  1.0,   r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_015429\stats.csv", 5000.0),
    ("exp3_30Mjup",  3.0,   r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_034408\stats.csv", 5000.0),
    ("exp4_50Mjup",  5.0,   r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_053325\stats.csv", 5000.0),
    ("exp5_100Mjup", 10.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_073428\stats.csv", 5000.0),
]

def extract_frame_state(stats_path, target_T0, tol=0.5):
    dtypes = {
        "sim_time_T0": np.float32, "agg_id": np.int32, "n_particles": np.int32,
        "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32,
    }
    cols = list(dtypes.keys())
    rows = []
    for chunk in pd.read_csv(stats_path, dtype=dtypes, chunksize=2_000_000, usecols=cols):
        in_window = chunk[(chunk["sim_time_T0"] >= target_T0 - tol) & (chunk["sim_time_T0"] <= target_T0 + tol)]
        if len(in_window):
            rows.append(in_window)
        if chunk["sim_time_T0"].iloc[-1] > target_T0 + tol and rows:
            break
    snap = pd.concat(rows, ignore_index=True)
    closest_t = snap.loc[(snap["sim_time_T0"] - target_T0).abs().idxmin(), "sim_time_T0"]
    snap = snap[snap["sim_time_T0"] == closest_t].copy()
    snap = snap[snap["agg_id"] != -1]
    snap["mass"] = snap["n_particles"].astype(float)
    snap["r"]    = np.sqrt(snap["cx_AU"]**2 + snap["cz_AU"]**2)
    return snap

print(f"\n{'='*120}")
print("H1: Зависимость числа внутренних планет от массы возмутителя")
print(f"{'='*120}\n")

# Define multiple "inner planet" criteria
inner_zones = [
    ("r < 2 AU",  2.0),
    ("r < 5 AU",  5.0),
    ("r < 10 AU", 10.0),
    ("r < 20 AU (внутри орбиты возмут)", 20.0),
]
mass_thresholds = [
    ("≥1 M⊕ (Earth+)", 1.0),
    ("≥5 M⊕ (super-Earth+)", 5.0),
    ("≥10 M⊕ (Neptune+)", 10.0),
    ("≥30 M⊕ (Saturn+)", 30.0),
    ("≥95 M⊕ (Jupiter+)", 95.0),
]

# For each run, compute snapshot and all metrics
all_snaps = {}
for label, mj, path, t_T in RUNS:
    print(f"Loading {label} ({mj} M_Jup)...", end=" ", flush=True)
    t0 = time.time()
    snap = extract_frame_state(path, t_T)
    print(f"{time.time()-t0:.0f}s  ({len(snap)} aggregates)")
    all_snaps[label] = (mj, snap)

# Print master table: inner-planet count for each (zone × mass threshold)
print()
for zone_label, r_max in inner_zones:
    print(f"\n--- Зона {zone_label} ---")
    header = f"{'M_Jup':>7}  " + "  ".join(f"{ml[:18]:>18}" for ml, _ in mass_thresholds)
    print(header)
    for label, mj, _, _ in RUNS:
        snap = all_snaps[label][1]
        in_zone = snap[snap["r"] < r_max]
        counts = []
        for ml, m_min in mass_thresholds:
            n = (in_zone["mass"] >= m_min).sum()
            counts.append(f"{n:>18}")
        print(f"  {mj:>5.1f}  " + "  ".join(counts))

# Compute mass distributed across zones
print(f"\n\n{'='*120}")
print("МАССА в каждой зоне (в M⊕ и в % от диска):")
print(f"{'='*120}\n")

zone_bounds = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 100), (100, 300)]
header = f"{'M_Jup':>7}  " + "  ".join(f"{f'{lo}-{hi}':>10}" for lo, hi in zone_bounds)
print(header + "      total")
for label, mj, _, _ in RUNS:
    snap = all_snaps[label][1]
    cells = []
    total = 0
    for lo, hi in zone_bounds:
        m = snap[(snap["r"] >= lo) & (snap["r"] < hi)]["mass"].sum()
        total += m
        pct = 100 * m / 10000
        cells.append(f"{m:>6.0f} ({pct:>2.0f}%)")
    print(f"  {mj:>5.1f}  " + "  ".join(cells) + f"    {total:>5.0f}")

# Total inner-planet metric — supreme definition (≥5 M_earth, r<20 AU)
print(f"\n\n{'='*120}")
print("СВОДНАЯ ТАБЛИЦА (по разным определениям 'внутренние планеты'):")
print(f"{'='*120}\n")

print(f"{'M_Jup':>7}  {'r<2,≥1':>8}  {'r<5,≥1':>8}  {'r<5,≥5':>8}  {'r<5,≥30':>8}  "
      f"{'r<10,≥5':>9}  {'r<20,≥5':>9}  {'r<20,≥30':>10}")
for label, mj, _, _ in RUNS:
    snap = all_snaps[label][1]
    n_a = ((snap["r"] < 2) & (snap["mass"] >= 1)).sum()
    n_b = ((snap["r"] < 5) & (snap["mass"] >= 1)).sum()
    n_c = ((snap["r"] < 5) & (snap["mass"] >= 5)).sum()
    n_d = ((snap["r"] < 5) & (snap["mass"] >= 30)).sum()
    n_e = ((snap["r"] < 10) & (snap["mass"] >= 5)).sum()
    n_f = ((snap["r"] < 20) & (snap["mass"] >= 5)).sum()
    n_g = ((snap["r"] < 20) & (snap["mass"] >= 30)).sum()
    print(f"  {mj:>5.1f}  {n_a:>8}  {n_b:>8}  {n_c:>8}  {n_d:>8}  {n_e:>9}  {n_f:>9}  {n_g:>10}")
