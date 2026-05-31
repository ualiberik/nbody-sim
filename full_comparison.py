"""
Comprehensive cross-experiment comparison.
For each run computes ~20 metrics and prints a unified table for analysis.
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

# Each entry: label, M_jup, stats_dir, final_top30_csv, per_frame_summary, total_disk_mearth
RUNS = [
    ("control_800",   0.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv",
                            r"C:\Projects\nbody-sim\analysis_control\snapshot_800yr_top50.csv",
                            r"C:\Projects\nbody-sim\analysis_control\per_frame_summary.csv", 5026.5),
    ("control_1592",  0.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv",
                            r"C:\Projects\nbody-sim\analysis_control\final_top30.csv",
                            r"C:\Projects\nbody-sim\analysis_control\per_frame_summary.csv", 10000.0),
    ("exp1_03Mjup",  0.3,   r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_000833\stats.csv",
                            r"C:\Projects\nbody-sim\analysis_exp1_03Mjup\final_top30.csv",
                            r"C:\Projects\nbody-sim\analysis_exp1_03Mjup\per_frame_summary.csv", 5000.0),
    ("exp2_10Mjup",  1.0,   r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_015429\stats.csv",
                            r"C:\Projects\nbody-sim\analysis_exp2_10Mjup\final_top30.csv",
                            r"C:\Projects\nbody-sim\analysis_exp2_10Mjup\per_frame_summary.csv", 5000.0),
    ("exp3_30Mjup",  3.0,   r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_034408\stats.csv",
                            r"C:\Projects\nbody-sim\analysis_exp3_30Mjup\final_top30.csv",
                            r"C:\Projects\nbody-sim\analysis_exp3_30Mjup\per_frame_summary.csv", 5000.0),
    ("exp4_50Mjup",  5.0,   r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_053325\stats.csv",
                            r"C:\Projects\nbody-sim\analysis_exp4_50Mjup\final_top30.csv",
                            r"C:\Projects\nbody-sim\analysis_exp4_50Mjup\per_frame_summary.csv", 5000.0),
    ("exp5_100Mjup", 10.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_073428\stats.csv",
                            r"C:\Projects\nbody-sim\analysis_exp5_100Mjup\final_top30.csv",
                            r"C:\Projects\nbody-sim\analysis_exp5_100Mjup\per_frame_summary.csv", 5000.0),
]

# Total disk mass = 10000 particles × 1 M_earth = 10000 M_earth in all runs
M_DISK_TOTAL = 10000.0

# Stream the FULL final-frame state from stats.csv for each run.
# We re-read because the final_top30.csv only has top 30.
def extract_frame_state(stats_path, target_T0, tol=0.5):
    dtypes = {
        "sim_time_T0": np.float32, "agg_id": np.int32, "n_particles": np.int32,
        "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32,
    }
    cols = list(dtypes.keys())
    rows = []
    last_t = 0.0
    for chunk in pd.read_csv(stats_path, dtype=dtypes, chunksize=2_000_000, usecols=cols):
        last_t = chunk["sim_time_T0"].iloc[-1]
        in_window = chunk[(chunk["sim_time_T0"] >= target_T0 - tol) & (chunk["sim_time_T0"] <= target_T0 + tol)]
        if len(in_window):
            rows.append(in_window)
        if last_t > target_T0 + tol and rows:
            break
    if not rows:
        return None
    snap = pd.concat(rows, ignore_index=True)
    # Pick the single frame closest to target
    closest_t = snap["sim_time_T0"].iloc[(snap["sim_time_T0"] - target_T0).abs().idxmin()]
    snap = snap[snap["sim_time_T0"] == closest_t].copy()
    snap = snap[snap["agg_id"] != -1]  # exclude perturber row
    snap["mass"] = snap["n_particles"].astype(float)
    snap["r"]    = np.sqrt(snap["cx_AU"]**2 + snap["cz_AU"]**2)
    return snap

def power_law_slope(masses, m_min=5):
    """Estimate cumulative mass-function power-law slope above m_min.
       Uses Maximum Likelihood Estimator (Clauset+2009).
       N(>m) ~ m^-alpha implies pdf p(m) ~ m^-(alpha+1).
       MLE slope alpha = N / sum(ln(m_i/m_min))"""
    m = masses[masses >= m_min]
    if len(m) < 3:
        return float('nan'), 0
    a = len(m) / np.sum(np.log(m / m_min))
    return a, len(m)

def vel_dispersion(group):
    """Velocity dispersion of bodies (proxy for dynamical heating)."""
    # Not available in current stats.csv columns; placeholder.
    return float('nan')

print(f"{'Run':<14} {'Mj':>5} {'N_tot':>6} {'N_giants':>9} {'N_plnts':>8} {'M_max':>7} {'M_giant':>8} "
      f"{'%disk':>6} {'M_eject':>8} {'%disk':>6} {'r_med':>7} {'r_max':>7} "
      f"{'M_pk':>7} {'t_pk':>7} {'slope':>6}")
print("-" * 145)

results = []
for label, mj, stats_path, top_csv, pf_csv, t_target in RUNS:
    print(f"  Loading {label} ...", end=" ", flush=True)

    # Per-frame summary — fast
    pf = pd.read_csv(pf_csv)
    peak_idx = pf["max_mass"].idxmax()
    peak_mass = pf["max_mass"].iloc[peak_idx]
    peak_yrs  = pf["yrs"].iloc[peak_idx]
    final_idx = pf["yrs"].idxmax()
    final_yrs = pf["yrs"].iloc[final_idx]

    # Lifetime over 500 M_earth (super-Jupiter regime)
    over500 = (pf["max_mass"] > 500).sum() * (final_yrs / len(pf))   # rough

    # Full final-frame state from stats.csv (only for runs without pre-cached snapshot)
    print("[stats stream...", end=" ", flush=True)
    t0 = time.time()
    snap = extract_frame_state(stats_path, t_target)
    print(f"{time.time()-t0:.0f}s]", end=" ", flush=True)

    if snap is None:
        print("FAILED")
        continue

    masses = snap["mass"].values
    radii  = snap["r"].values

    n_tot = len(masses)
    n_giants = int((masses >= 30).sum())
    n_planets = int((masses >= 5).sum())
    M_max = float(masses.max())
    M_giant_total = float(masses[masses >= 30].sum())
    M_planet_total = float(masses[masses >= 5].sum())
    pct_disk_giants = 100.0 * M_giant_total / M_DISK_TOTAL

    # Ejected: r > 30 AU
    n_eject = int((radii > 30).sum())
    M_eject = float(masses[radii > 30].sum())
    pct_disk_eject = 100.0 * M_eject / M_DISK_TOTAL

    # Median / max radius of GIANTS
    g_radii = radii[masses >= 30]
    r_med = float(np.median(g_radii)) if len(g_radii) else 0.0
    r_max = float(radii.max())

    # Mass function slope (power law for m ≥ 5)
    alpha, n_fit = power_law_slope(masses, m_min=5)

    # Inner-disk mass retention: < 5 AU
    M_inner = float(masses[radii < 5].sum())
    pct_inner = 100.0 * M_inner / M_DISK_TOTAL

    # Most massive in 30 AU < r < ejection
    M_outerring = float(masses[(radii > 30) & (radii < 80)].sum())

    results.append({
        "label": label, "Mj": mj,
        "n_tot": n_tot, "n_giants": n_giants, "n_planets": n_planets,
        "M_max": M_max, "M_giant_total": M_giant_total,
        "pct_disk_giants": pct_disk_giants,
        "M_eject": M_eject, "pct_disk_eject": pct_disk_eject,
        "r_med_giants": r_med, "r_max": r_max,
        "peak_mass": peak_mass, "peak_yrs": peak_yrs,
        "slope_alpha": alpha,
        "M_inner_5AU": M_inner, "pct_inner": pct_inner,
        "M_outerring": M_outerring,
        "final_yrs": final_yrs,
    })

    print(f"{label:<14} {mj:>5.1f} {n_tot:>6} {n_giants:>9} {n_planets:>8} "
          f"{M_max:>6.0f}  {M_giant_total:>7.0f}  {pct_disk_giants:>5.1f}  "
          f"{M_eject:>7.0f}  {pct_disk_eject:>5.1f}  "
          f"{r_med:>6.1f}  {r_max:>6.1f}  "
          f"{peak_mass:>6.0f}  {peak_yrs:>6.0f}  {alpha:>5.2f}")

print()
print("=" * 145)
print("Metric definitions:")
print("  Mj             — perturber mass in Jupiter masses")
print("  N_tot          — total aggregates in final frame")
print("  N_giants       — bodies ≥ 30 M⊕ (gas-giant class)")
print("  N_plnts        — bodies ≥  5 M⊕ (planet class)")
print("  M_max          — heaviest body (M⊕)")
print("  M_giant        — sum of mass of giants ≥30 M⊕ (M⊕)")
print("  %disk (giant)  — that mass / 10000 M⊕ initial disk")
print("  M_eject        — sum mass of bodies at r > 30 AU")
print("  %disk (eject)  — ejected fraction of disk mass")
print("  r_med          — median r of giants")
print("  r_max          — most distant body's r")
print("  M_pk           — peak max_mass over entire run (M⊕)")
print("  t_pk           — when peak occurred (yr)")
print("  slope          — power-law exponent of mass function N(>m) ~ m^-alpha")

# Save results to JSON for later use
import json
out_json = Path(r"C:\Projects\nbody-sim\analysis_final_comparison\metrics_table.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrics saved to {out_json}")
