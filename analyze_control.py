"""
Detailed analysis of the LONG control simulation (10000 T0 = ~1590 yr).
The stats.csv is ~10 GB — we stream-read it in chunks to keep memory bounded.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643")
OUT  = Path(r"C:\Projects\nbody-sim\analysis_control")
OUT.mkdir(exist_ok=True)
STATS = DATA / "stats.csv"

print(f"Streaming {STATS.name} ({STATS.stat().st_size/1e9:.2f} GB) ...")
t0 = time.time()

# Per-frame summary streamed from CSV — keep one row per frame
# Columns: frame, sim_time_T0, sim_time_years, n_aggregates, agg_id, n_particles,
#          mass_Msun, mass_Mearth, cx, cy, cz, vx, vy, vz, dist_center_AU
dtypes = {
    "frame":  np.int32,
    "sim_time_T0": np.float32,
    "n_aggregates": np.int32,
    "agg_id": np.int32,
    "n_particles": np.int32,
    "mass_Mearth": np.float32,
    "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32,
    "vx": np.float32,    "vy": np.float32,    "vz": np.float32,
    "dist_center_AU": np.float32,
}

# Two-pass strategy:
#   pass 1: stream the CSV, compute per-frame summary (n_agg, max_mass, etc.)
#           AND keep ONLY the last frame's rows for detailed final-state analysis
#   pass 2: with that summary in memory, generate plots
last_frame_rows = []
per_frame = []  # (frame, t, yrs, n_agg, max_mass, sec_mass, third_mass)

# Reasonable chunk size — ~1M rows ~ 100MB in memory
CHUNK = 2_000_000
last_seen_frame = -1
cumulative = []  # current frame accumulator

# Also track top-3 max masses per frame
def flush(buf):
    if not buf:
        return None
    fr = buf[0]['frame']
    t  = buf[0]['sim_time_T0']
    yrs = t * 0.15915494
    n_agg = len(buf)
    masses = np.array([r['n_particles'] for r in buf])  # = M_earth (heavy disk)
    top3 = np.sort(masses)[::-1][:3]
    return {
        "frame": fr, "t": t, "yrs": yrs, "n_agg": n_agg,
        "max_mass": top3[0] if len(top3) > 0 else 0,
        "sec_mass": top3[1] if len(top3) > 1 else 0,
        "third_mass": top3[2] if len(top3) > 2 else 0,
    }

n_rows = 0
last_log_t = time.time()
for chunk in pd.read_csv(STATS, dtype=dtypes, chunksize=CHUNK,
                          usecols=list(dtypes.keys())):
    n_rows += len(chunk)
    for fr, sub in chunk.groupby("frame"):
        if fr != last_seen_frame and last_seen_frame >= 0:
            # New frame — flush previous
            summary = flush(cumulative)
            if summary: per_frame.append(summary)
            cumulative = []
        last_seen_frame = fr
        cumulative.extend(sub.to_dict("records"))
    if time.time() - last_log_t > 5:
        print(f"  ... processed {n_rows:,} rows, frame {last_seen_frame}, "
              f"elapsed {time.time()-t0:.0f}s")
        last_log_t = time.time()

# Flush last frame
if cumulative:
    summary = flush(cumulative)
    if summary: per_frame.append(summary)
    last_frame_rows = cumulative  # keep for final analysis

pf = pd.DataFrame(per_frame)
print(f"Done streaming in {time.time()-t0:.0f}s — {len(pf)} frames captured.")
print(f"Time range: t = {pf['t'].min():.1f} ... {pf['t'].max():.1f} T0")
print(f"Final frame has {len(last_frame_rows)} aggregates")

# Save per-frame summary
pf.to_csv(OUT / "per_frame_summary.csv", index=False)

# ----------------------------------------------------------------------
# PLOT 1: aggregate count + top-3 mass over time
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
axes[0].plot(pf["yrs"], pf["n_agg"], color="steelblue", lw=0.5)
axes[0].set_ylabel("Aggregate count")
axes[0].set_title(f"Control simulation: {len(pf)} frames over 0 → {pf['yrs'].max():.0f} yr")
axes[0].grid(alpha=0.3)

axes[1].plot(pf["yrs"], pf["max_mass"], color="crimson",  lw=0.5, label="1st heaviest (M⊕)")
axes[1].plot(pf["yrs"], pf["sec_mass"], color="steelblue", lw=0.5, label="2nd heaviest")
axes[1].plot(pf["yrs"], pf["third_mass"], color="darkgreen", lw=0.5, label="3rd heaviest")
axes[1].set_yscale("log")
axes[1].set_ylabel("Mass (M⊕)")
axes[1].set_xlabel("Time (years)")
axes[1].grid(alpha=0.3, which="both")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT / "01_growth_curves.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# Final-state analysis
# ----------------------------------------------------------------------
final = pd.DataFrame(last_frame_rows)
final["mass_Mearth"] = final["n_particles"].astype(float)
final["r_cyl"] = np.sqrt(final["cx_AU"]**2 + final["cz_AU"]**2)
final = final.sort_values("mass_Mearth", ascending=False).reset_index(drop=True)

t_final_yr = final["sim_time_T0"].iloc[0] * 0.15915494
print(f"\n=== FINAL STATE (t = {final['sim_time_T0'].iloc[0]:.1f} T0 = {t_final_yr:.1f} yr) ===")
print(f"Total aggregates: {len(final)}")
print(f"Total mass: {final['mass_Mearth'].sum():.1f} M⊕")
print()

classes = [
    ("Super-Jupiter (300-1000 M⊕)",  300, 1000),
    ("Jupiter-like (95-300 M⊕)",      95,  300),
    ("Saturn-like (30-95 M⊕)",        30,   95),
    ("Neptune/Uranus (10-30 M⊕)",     10,   30),
    ("Super-Earth (3-10 M⊕)",          3,   10),
    ("Earth-like (1-3 M⊕)",            1,    3),
]
print(f"{'Class':<35} {'count':>6}  {'total':>10}  {'biggest':>9}")
for name, lo, hi in classes:
    sub = final[(final["mass_Mearth"] >= lo) & (final["mass_Mearth"] < hi)]
    print(f"{name:<35} {len(sub):>6}  {sub['mass_Mearth'].sum():>8.1f} M⊕  "
          f"{sub['mass_Mearth'].max() if len(sub) else 0:>7.1f} M⊕")

# Save top 30
final.head(30).to_csv(OUT / "final_top30.csv", index=False)

# ----------------------------------------------------------------------
# PLOT 2: Top-down view of final disk
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))
sc = ax.scatter(final["cx_AU"], final["cz_AU"],
                s=np.clip(final["mass_Mearth"]*0.4, 1, 500),
                c=np.log10(final["mass_Mearth"].clip(lower=0.5)),
                cmap="plasma", alpha=0.75, edgecolors="none")
ax.plot(0, 0, marker="*", color="gold", markersize=30, markeredgecolor="black", zorder=10)
ax.set_aspect("equal")
xmax = max(80, abs(final["cx_AU"]).max() * 1.1)
ax.set_xlim(-xmax, xmax); ax.set_ylim(-xmax, xmax)
ax.set_title(f"Control sim final state: t = {t_final_yr:.0f} yr, {len(final)} aggregates")
ax.set_xlabel("x (AU)"); ax.set_ylabel("z (AU)")
plt.colorbar(sc, ax=ax, label="log₁₀ M⊕")
ax.grid(alpha=0.2)

# Highlight any ejected bodies (r > 30)
ejected = final[final["r_cyl"] > 30]
for _, row in ejected.iterrows():
    ax.annotate(f" {row['mass_Mearth']:.0f}", (row["cx_AU"], row["cz_AU"]),
                fontsize=11, color="red", fontweight="bold")
    ax.plot(row["cx_AU"], row["cz_AU"], marker="o", mfc="none", mec="red", ms=18, mew=1.5)

plt.tight_layout()
plt.savefig(OUT / "02_final_topdown.png", dpi=120)
plt.close()
print(f"\nEjected bodies (r > 30 AU): {len(ejected)}")
for _, row in ejected.iterrows():
    print(f"  {row['mass_Mearth']:.0f} M⊕ at r = {row['r_cyl']:.1f} AU")

# ----------------------------------------------------------------------
# PLOT 3: Mass vs radius (with Solar System overlay)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(final["r_cyl"], final["mass_Mearth"],
           s=np.clip(final["mass_Mearth"]*0.5, 4, 400),
           c=np.log10(final["mass_Mearth"]),
           cmap="plasma", alpha=0.7, edgecolors="none")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.3, 100); ax.set_ylim(0.5, max(1500, final["mass_Mearth"].max()*1.2))
ax.set_xlabel("Orbital radius (AU)"); ax.set_ylabel("Mass (M⊕)")
ax.set_title(f"Mass vs orbital radius at t = {t_final_yr:.0f} yr — {len(final)} bodies")
ax.grid(alpha=0.3, which="both")

for name, r, m in [("Mercury",0.39,0.055),("Venus",0.72,0.82),("Earth",1.0,1.0),("Mars",1.52,0.107),
                    ("Jupiter",5.2,318),("Saturn",9.5,95),("Uranus",19.2,14.5),("Neptune",30.1,17)]:
    ax.plot(r, m, marker="*", color="black", ms=12, mfc="none", mew=1.5)
    ax.annotate(f" {name}", (r, m), fontsize=9, color="black")
plt.tight_layout()
plt.savefig(OUT / "03_mass_radius.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# Gas-giant architecture
# ----------------------------------------------------------------------
giants = final[final["mass_Mearth"] >= 30].sort_values("r_cyl").reset_index(drop=True)
giants["T"] = giants["r_cyl"]**1.5
print(f"\n=== Gas giants (≥30 M⊕): {len(giants)} ===")
for _, g in giants.iterrows():
    print(f"  {g['mass_Mearth']:>5.0f} M⊕  at r={g['r_cyl']:6.3f} AU,  T={g['T']:6.2f} yr")
if len(giants) > 1:
    print(f"\nConsecutive period ratios:")
    for i in range(len(giants)-1):
        ratio = giants['T'].iloc[i+1] / giants['T'].iloc[i]
        print(f"  pair {i}: r={giants['r_cyl'].iloc[i]:.2f}→{giants['r_cyl'].iloc[i+1]:.2f}  "
              f"ratio={ratio:.3f}")

print(f"\nAll outputs in: {OUT}")
