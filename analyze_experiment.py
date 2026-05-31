"""
Generic per-experiment analysis. Pass data_dir, label, perturber_mass_mjup.
Outputs go to analysis_<label>/.
"""
import sys, os, argparse, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="data dir with frames.bin + stats.csv")
ap.add_argument("--label", required=True, help="output label (e.g. 'exp1_03Mjup')")
ap.add_argument("--mass_mjup", type=float, required=True, help="perturber mass in M_Jup")
args = ap.parse_args()

DATA = Path(args.data)
LABEL = args.label
PMASS = args.mass_mjup
OUT = Path(rf"C:\Projects\nbody-sim\analysis_{LABEL}")
OUT.mkdir(exist_ok=True)
STATS = DATA / "stats.csv"

print(f"=== Analyzing {LABEL}  (perturber = {PMASS} M_Jup) ===")
print(f"Stats: {STATS}  ({STATS.stat().st_size/1e9:.2f} GB)")

dtypes = {
    "frame": np.int32, "sim_time_T0": np.float32, "n_aggregates": np.int32,
    "agg_id": np.int32, "n_particles": np.int32,
    "mass_Mearth": np.float32,
    "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32,
    "vx": np.float32, "vy": np.float32, "vz": np.float32,
    "dist_center_AU": np.float32,
}

# Stream + collect last-frame rows + perturber track + per-frame summary
per_frame = []
last_rows = []
pert_track = []   # perturber positions over time (agg_id == -1)
last_seen_frame = -1
cumulative = []

def flush(buf):
    if not buf:
        return None
    # separate perturber and aggregates
    pert = [r for r in buf if r["agg_id"] == -1]
    aggs = [r for r in buf if r["agg_id"] != -1]
    if not aggs and not pert:
        return None
    fr = buf[0]["frame"]
    t = buf[0]["sim_time_T0"]
    yrs = t * 0.15915494
    masses = np.array([r["n_particles"] for r in aggs])
    top3 = np.sort(masses)[::-1][:3] if len(masses) else np.array([])
    return {
        "frame": fr, "t": t, "yrs": yrs,
        "n_agg": len(aggs),
        "max_mass": top3[0] if len(top3) > 0 else 0,
        "sec_mass": top3[1] if len(top3) > 1 else 0,
        "third_mass": top3[2] if len(top3) > 2 else 0,
        "pert": pert[0] if pert else None,
    }

CHUNK = 2_000_000
n_rows = 0
t0 = time.time()
last_log = time.time()
for chunk in pd.read_csv(STATS, dtype=dtypes, chunksize=CHUNK, usecols=list(dtypes.keys())):
    n_rows += len(chunk)
    for fr, sub in chunk.groupby("frame"):
        if fr != last_seen_frame and last_seen_frame >= 0:
            summary = flush(cumulative)
            if summary:
                per_frame.append(summary)
                if summary["pert"]: pert_track.append(summary["pert"])
            cumulative = []
        last_seen_frame = fr
        cumulative.extend(sub.to_dict("records"))
    if time.time() - last_log > 5:
        print(f"  ... {n_rows:,} rows, frame {last_seen_frame}, elapsed {time.time()-t0:.0f}s")
        last_log = time.time()

if cumulative:
    summary = flush(cumulative)
    if summary:
        per_frame.append(summary)
        if summary["pert"]: pert_track.append(summary["pert"])
    last_rows = cumulative

pf = pd.DataFrame(per_frame)
print(f"Done streaming in {time.time()-t0:.0f}s — {len(pf)} frames captured.")

# Save per-frame summary
pf.drop(columns=["pert"]).to_csv(OUT / "per_frame_summary.csv", index=False)

# ----------------------------------------------------------------------
# Final-state analysis (exclude perturber row)
# ----------------------------------------------------------------------
final = pd.DataFrame(last_rows)
pert_final = final[final["agg_id"] == -1]
final = final[final["agg_id"] != -1].copy()
final["mass_Mearth"] = final["n_particles"].astype(float)
final["r_cyl"] = np.sqrt(final["cx_AU"]**2 + final["cz_AU"]**2)
final = final.sort_values("mass_Mearth", ascending=False).reset_index(drop=True)

t_fin_yr = pf["yrs"].iloc[-1]
print(f"\n=== FINAL STATE (t = {t_fin_yr:.1f} yr) ===")
print(f"Total aggregates: {len(final)}")
print(f"Total mass in aggregates: {final['mass_Mearth'].sum():.1f} M⊕")
if len(pert_final):
    p = pert_final.iloc[0]
    print(f"Perturber at: ({p['cx_AU']:.2f}, {p['cz_AU']:.2f}), r = {p['dist_center_AU']:.2f} AU, mass = {p['mass_Mearth']:.1f} M⊕")

classes = [
    ("Super-Jupiter (300-1000 M⊕)",  300, 1000),
    ("Jupiter-like (95-300 M⊕)",      95,  300),
    ("Saturn-like (30-95 M⊕)",        30,   95),
    ("Neptune/Uranus (10-30 M⊕)",     10,   30),
    ("Super-Earth (3-10 M⊕)",          3,   10),
    ("Earth-like (1-3 M⊕)",            1,    3),
]
print()
print(f"{'Class':<35} {'count':>6}  {'total':>10}  {'biggest':>9}")
for name, lo, hi in classes:
    sub = final[(final["mass_Mearth"] >= lo) & (final["mass_Mearth"] < hi)]
    print(f"{name:<35} {len(sub):>6}  {sub['mass_Mearth'].sum():>8.1f} M⊕  "
          f"{sub['mass_Mearth'].max() if len(sub) else 0:>7.1f} M⊕")

# Save final state
final.head(30).to_csv(OUT / "final_top30.csv", index=False)

# ----------------------------------------------------------------------
# Plot: growth curves
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
axes[0].plot(pf["yrs"], pf["n_agg"], color="steelblue", lw=0.5)
axes[0].set_ylabel("Aggregate count")
axes[0].set_title(f"{LABEL}: {PMASS} M_Jup perturber at r=20 AU")
axes[0].grid(alpha=0.3)

axes[1].plot(pf["yrs"], pf["max_mass"], color="crimson",  lw=0.5, label="1st heaviest")
axes[1].plot(pf["yrs"], pf["sec_mass"], color="steelblue", lw=0.5, label="2nd heaviest")
axes[1].plot(pf["yrs"], pf["third_mass"], color="darkgreen", lw=0.5, label="3rd heaviest")
axes[1].set_yscale("log")
axes[1].set_ylabel("Mass (M⊕)")
axes[1].set_xlabel("Time (yr)")
axes[1].grid(alpha=0.3, which="both")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT / "01_growth_curves.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# Plot: top-down view of final disk + perturber position
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))
sc = ax.scatter(final["cx_AU"], final["cz_AU"],
                s=np.clip(final["mass_Mearth"]*0.4, 1, 500),
                c=np.log10(final["mass_Mearth"].clip(lower=0.5)),
                cmap="plasma", alpha=0.75, edgecolors="none")
ax.plot(0, 0, marker="*", color="gold", markersize=30, markeredgecolor="black", zorder=10)

# Perturber: green diamond
if len(pert_final):
    p = pert_final.iloc[0]
    pmass_size = max(40, np.log10(PMASS+0.1) * 100 + 100)
    ax.plot(p["cx_AU"], p["cz_AU"], marker="D", color="limegreen",
            markersize=20, markeredgecolor="black", mew=2, zorder=11)
    ax.annotate(f"  {PMASS} M_Jup\n  r={p['dist_center_AU']:.1f}",
                (p["cx_AU"], p["cz_AU"]),
                fontsize=11, color="darkgreen", fontweight="bold")
    # Draw perturber orbit
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(20*np.cos(theta), 20*np.sin(theta), '--', color="limegreen", alpha=0.4, lw=1)

ax.set_aspect("equal")
xmax = max(80, abs(final["cx_AU"]).max() * 1.1)
ax.set_xlim(-xmax, xmax); ax.set_ylim(-xmax, xmax)
ax.set_title(f"{LABEL} final state at t = {t_fin_yr:.0f} yr — {len(final)} aggregates")
ax.set_xlabel("x (AU)"); ax.set_ylabel("z (AU)")
plt.colorbar(sc, ax=ax, label="log₁₀ M⊕")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / "02_final_topdown.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# Plot: mass vs radius
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(final["r_cyl"], final["mass_Mearth"],
           s=np.clip(final["mass_Mearth"]*0.5, 4, 400),
           c=np.log10(final["mass_Mearth"]),
           cmap="plasma", alpha=0.7)
# Perturber as a horizontal reference line at its r
if len(pert_final):
    p = pert_final.iloc[0]
    ax.axvline(p["dist_center_AU"], color="limegreen", alpha=0.6, ls="--",
               label=f"perturber at r=20 (M={PMASS} M_Jup)")
    ax.scatter([p["dist_center_AU"]], [p["mass_Mearth"]],
               s=300, c="limegreen", marker="D", edgecolors="black",
               label=f"perturber {PMASS} M_Jup", zorder=10)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.3, 100); ax.set_ylim(0.5, max(2000, final["mass_Mearth"].max()*1.2,
                                              pert_final["mass_Mearth"].max()*1.2 if len(pert_final) else 0))
ax.set_xlabel("Orbital radius (AU)"); ax.set_ylabel("Mass (M⊕)")
ax.set_title(f"{LABEL}: mass vs r")
ax.grid(alpha=0.3, which="both")
for name, r, m in [("Mercury",0.39,0.055),("Venus",0.72,0.82),("Earth",1.0,1.0),("Mars",1.52,0.107),
                    ("Jupiter",5.2,318),("Saturn",9.5,95),("Uranus",19.2,14.5),("Neptune",30.1,17)]:
    ax.plot(r, m, marker="*", color="black", ms=10, mfc="none", mew=1.5)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "03_mass_radius.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# Print gas-giant list
# ----------------------------------------------------------------------
giants = final[final["mass_Mearth"] >= 30].sort_values("r_cyl").reset_index(drop=True)
print(f"\nGas giants (≥30 M⊕): {len(giants)}")
for _, g in giants.iterrows():
    print(f"  {g['mass_Mearth']:>5.0f} M⊕  at r={g['r_cyl']:6.3f} AU")

# Ejected bodies
ejected = final[final["r_cyl"] > 30]
print(f"\nEjected bodies (r > 30 AU): {len(ejected)}")
print(f"  Most distant: r = {ejected['r_cyl'].max() if len(ejected) else 0:.1f} AU")
print(f"  Total ejected mass: {ejected['mass_Mearth'].sum():.1f} M⊕")

print(f"\nOutputs in: {OUT}")
