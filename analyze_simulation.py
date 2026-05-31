"""
Deep analysis of the N-body planet formation simulation.

Looks for:
  - Largest-body growth history (planet formation timescale)
  - Mass distribution evolution (size spectrum vs. time)
  - Ring / gap structure in the surface-density profile
  - Migration tracks of the most massive bodies
  - Mean-motion resonances among the surviving planets
  - Collision / merger events (sudden mass jumps in a track)
  - Ejection events (particles flung inward to the star or outward)
"""

import os
import sys
import time
# Force UTF-8 on Windows so we can print unicode arrows / sub-/super-scripts
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r"C:\Projects\nbody-sim\data\20260525_185147")
OUT_DIR  = Path(r"C:\Projects\nbody-sim\analysis_output")
OUT_DIR.mkdir(exist_ok=True)

STATS = DATA_DIR / "stats.csv"

# ----------------------------------------------------------------------
# 1. Load stats.csv — large file (~522 MB). Use efficient dtypes.
# ----------------------------------------------------------------------
print(f"[1/8] Loading {STATS.name} ...")
t0 = time.time()

dtypes = {
    "frame": np.int32,
    "sim_time_T0": np.float32,
    "sim_time_years": np.float32,
    "n_aggregates": np.int32,
    "agg_id": np.int32,
    "n_particles": np.int32,
    "mass_Msun": np.float64,
    "mass_Mearth": np.float32,
    "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32,
    "vx": np.float32, "vy": np.float32, "vz": np.float32,
    "dist_center_AU": np.float32,
}
df = pd.read_csv(STATS, dtype=dtypes)
print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s")
print(f"  frames: {df['frame'].min()} … {df['frame'].max()}")
print(f"  unique frames: {df['frame'].nunique()}")

# Per-frame summary view
per_frame = df.groupby("frame").agg(
    t=("sim_time_T0", "first"),
    yrs=("sim_time_years", "first"),
    n_agg=("agg_id", "count"),
    n_particles_total=("n_particles", "sum"),
    max_size=("n_particles", "max"),
    max_mass_Mearth=("mass_Mearth", "max"),
).reset_index()

# ----------------------------------------------------------------------
# 2. Aggregate-count and largest-body growth over time
# ----------------------------------------------------------------------
print("[2/8] Plotting growth curves ...")
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax[0].plot(per_frame["yrs"], per_frame["n_agg"], color="steelblue", lw=0.7)
ax[0].set_ylabel("Number of aggregates")
ax[0].set_title("Aggregate count over time")
ax[0].grid(alpha=0.3)

ax[1].plot(per_frame["yrs"], per_frame["max_size"], color="crimson", lw=0.7, label="Largest (# particles)")
ax[1].set_ylabel("Largest aggregate size")
ax[1].set_xlabel("Time (years)")
ax[1].set_yscale("log")
ax[1].grid(alpha=0.3)
ax[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "01_growth_curves.png", dpi=110)
plt.close()

print(f"  initial #agg = {per_frame['n_agg'].iloc[0]}")
print(f"  final   #agg = {per_frame['n_agg'].iloc[-1]}")
print(f"  max ever largest size = {per_frame['max_size'].max()} particles "
      f"(= {per_frame['max_mass_Mearth'].max():.2f} M_earth)")

# ----------------------------------------------------------------------
# 3. Final-frame statistics — what does the system look like at t=1000?
# ----------------------------------------------------------------------
print("[3/8] Final-frame analysis ...")
last_frame = df["frame"].max()
final = df[df["frame"] == last_frame].copy()
final = final.sort_values("mass_Mearth", ascending=False).reset_index(drop=True)

print(f"  final frame: {last_frame}, t = {final['sim_time_years'].iloc[0]:.2f} yr")
print(f"  total aggregates: {len(final)}")
print(f"  Top 10 most massive bodies:")
top = final.head(10)[["agg_id", "n_particles", "mass_Mearth", "dist_center_AU"]]
print(top.to_string(index=False))

# Save top-50 final list
final.head(50).to_csv(OUT_DIR / "final_top50.csv", index=False)

# ----------------------------------------------------------------------
# 4. Mass distribution evolution (size spectrum)
# ----------------------------------------------------------------------
print("[4/8] Mass spectrum evolution ...")
snapshot_times = [10, 50, 100, 250, 500, 1000]  # T0 units
frames_to_plot = []
for t_target in snapshot_times:
    closest_frame = (per_frame["t"] - t_target).abs().idxmin()
    frames_to_plot.append((t_target, per_frame.loc[closest_frame, "frame"]))

fig, ax = plt.subplots(figsize=(10, 7))
for t_target, fr in frames_to_plot:
    sizes = df[df["frame"] == fr]["n_particles"].values
    if len(sizes) < 2:
        continue
    # Cumulative N(>m): on log-log this exposes a power law if present
    s_sorted = np.sort(sizes)[::-1]
    cum_n    = np.arange(1, len(s_sorted) + 1)
    ax.plot(s_sorted, cum_n, label=f"t={t_target} T₀ ({t_target/(2*np.pi):.1f} yr)", lw=1.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Aggregate size (particles)")
ax.set_ylabel("N (> size)")
ax.set_title("Cumulative size distribution evolution")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "02_mass_spectrum.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# 5. Radial surface-density profile evolution — look for rings/gaps
# ----------------------------------------------------------------------
print("[5/8] Radial density profile ...")
r_bins = np.linspace(0, 8, 81)         # 0.1-AU bins out to 8 AU
r_mid  = 0.5 * (r_bins[1:] + r_bins[:-1])

fig, ax = plt.subplots(figsize=(11, 7))
for t_target, fr in frames_to_plot:
    frame_df = df[df["frame"] == fr]
    # Surface density weighted by particle count (mass proxy)
    weights = frame_df["n_particles"].values
    hist, _ = np.histogram(frame_df["dist_center_AU"], bins=r_bins, weights=weights)
    # Sigma = mass / (2 pi r dr)
    area   = 2 * np.pi * r_mid * np.diff(r_bins)
    sigma  = hist / area
    ax.plot(r_mid, sigma, label=f"t={t_target} T₀", lw=1.4)

ax.set_xlabel("Radius (AU)")
ax.set_ylabel("Σ (particles / AU²)  [mass-weighted]")
ax.set_yscale("log")
ax.set_title("Surface-density profile — rings & gaps")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "03_radial_profile.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# 6. Migration tracks: follow the top-N most massive final bodies backwards
#    (agg_id is NOT stable across frames, so we follow by spatial nearest
#    neighbour: pick a body in the final frame, find the most-massive body
#    within a search radius in the previous frame, repeat.)
# ----------------------------------------------------------------------
print("[6/8] Migration tracks of largest bodies ...")
N_TRACK = 8
final_top = final.head(N_TRACK).reset_index(drop=True)

# Build per-frame groups once for speed
frames = sorted(df["frame"].unique())
groups = {fr: g for fr, g in df.groupby("frame")}

tracks = []  # list of dicts: {label, t[], r[], mass[]}
for i, row in final_top.iterrows():
    t_arr, r_arr, m_arr = [], [], []
    cx, cy, cz = row["cx_AU"], row["cy_AU"], row["cz_AU"]
    mass = row["mass_Mearth"]
    for fr in reversed(frames):
        g = groups[fr]
        # candidate: heaviest body within 0.5 AU of current position
        dx = g["cx_AU"].values - cx
        dy = g["cy_AU"].values - cy
        dz = g["cz_AU"].values - cz
        d  = np.sqrt(dx*dx + dy*dy + dz*dz)
        # Allow more drift if the body is moving fast (large step)
        mask = d < 0.6
        if not mask.any():
            break
        idx_sub = np.where(mask)[0]
        # Pick the candidate with the most similar mass (prevents jumping)
        sub = g.iloc[idx_sub]
        # Score: distance + |log(mass ratio)|
        log_mratio = np.abs(np.log(sub["mass_Mearth"].values / max(mass, 1e-9)))
        score = d[idx_sub] + 0.3 * log_mratio
        best = idx_sub[np.argmin(score)]
        cx, cy, cz = g["cx_AU"].iloc[best], g["cy_AU"].iloc[best], g["cz_AU"].iloc[best]
        mass = g["mass_Mearth"].iloc[best]
        t_arr.append(g["sim_time_years"].iloc[best])
        r_arr.append(np.sqrt(cx*cx + cz*cz))   # cylindrical radius in disk plane
        m_arr.append(mass)
    tracks.append({
        "label": f"#{i+1}  final {row['mass_Mearth']:.2f} M⊕",
        "t": np.array(t_arr[::-1]),
        "r": np.array(r_arr[::-1]),
        "m": np.array(m_arr[::-1]),
    })

fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 1, N_TRACK))
for tr, c in zip(tracks, colors):
    ax[0].plot(tr["t"], tr["r"], color=c, lw=0.9, label=tr["label"])
    ax[1].plot(tr["t"], tr["m"], color=c, lw=0.9)

ax[0].set_ylabel("Orbital radius (AU)")
ax[0].set_title("Migration tracks of top-8 final bodies")
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=8, ncol=2)
ax[1].set_ylabel("Mass (M⊕)")
ax[1].set_xlabel("Time (years)")
ax[1].set_yscale("log")
ax[1].grid(alpha=0.3, which="both")
ax[1].set_title("Mass growth tracks")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_migration_tracks.png", dpi=110)
plt.close()

# ----------------------------------------------------------------------
# 7. Mean-motion resonance search among the largest survivors
# ----------------------------------------------------------------------
print("[7/8] Resonance search ...")
# In Kepler units (G=M=1), a = r if circular. We use r as a proxy for semi-major axis.
top20 = final.head(20).copy()
# Cylindrical orbital radius (disk plane = XZ)
top20["r_cyl"] = np.sqrt(top20["cx_AU"]**2 + top20["cz_AU"]**2)
top20["period_yr"] = top20["r_cyl"]**1.5     # Kepler: T = a^{3/2} in years (M=1 M_sun)
top20 = top20.sort_values("r_cyl").reset_index(drop=True)

print("  Top-20 final bodies (sorted by radius):")
print(top20[["agg_id", "n_particles", "mass_Mearth", "r_cyl", "period_yr"]].to_string(index=False))

# Check period ratios for low-integer commensurabilities
print("\n  Period-ratio commensurabilities (within 2 %):")
common = [(2,1), (3,2), (4,3), (5,3), (5,4), (7,5), (3,1), (5,2), (7,3)]
hits = 0
for i in range(len(top20)):
    for j in range(i+1, len(top20)):
        ratio = top20["period_yr"].iloc[j] / top20["period_yr"].iloc[i]
        for p, q in common:
            target = p / q
            if abs(ratio/target - 1) < 0.02:
                print(f"    body#{i+1} (r={top20['r_cyl'].iloc[i]:.3f} AU)"
                      f" ↔ body#{j+1} (r={top20['r_cyl'].iloc[j]:.3f} AU)"
                      f"  ratio={ratio:.4f}  ≈ {p}:{q}")
                hits += 1
                break
if hits == 0:
    print("    (no clean MMRs found within 2 % tolerance)")

# ----------------------------------------------------------------------
# 8. Collisions / mergers and ejections
# ----------------------------------------------------------------------
print("[8/8] Collision & ejection events ...")
# Track the global maximum mass over time.  Sharp jumps = mergers involving
# the heaviest body.  Drops can happen if the heaviest body gets ejected
# (drops outside the FOF radius and a smaller body becomes new champion).
mass_over_t = per_frame["max_mass_Mearth"].values
yrs         = per_frame["yrs"].values
dmass = np.diff(mass_over_t)

# A "merger jump" = >5 % mass increase in one frame
merger_idx = np.where(dmass / mass_over_t[:-1] > 0.05)[0]
print(f"  major merger events affecting champion body: {len(merger_idx)}")
if len(merger_idx) > 0:
    print(f"  first merger: t = {yrs[merger_idx[0]+1]:.2f} yr "
          f"({mass_over_t[merger_idx[0]]:.3f} → {mass_over_t[merger_idx[0]+1]:.3f} M⊕)")
    biggest_jump = merger_idx[np.argmax(dmass[merger_idx])]
    print(f"  biggest jump:  t = {yrs[biggest_jump+1]:.2f} yr "
          f"({mass_over_t[biggest_jump]:.3f} → {mass_over_t[biggest_jump+1]:.3f} M⊕)")

# Hot-inner-region accumulation: how much mass ends up inside r<1 AU vs r>3 AU?
inner_initial = df[(df["frame"]==frames[0]) & (df["dist_center_AU"]<1.0)]["n_particles"].sum()
outer_initial = df[(df["frame"]==frames[0]) & (df["dist_center_AU"]>3.0)]["n_particles"].sum()
inner_final   = df[(df["frame"]==last_frame) & (df["dist_center_AU"]<1.0)]["n_particles"].sum()
outer_final   = df[(df["frame"]==last_frame) & (df["dist_center_AU"]>3.0)]["n_particles"].sum()
print(f"\n  Mass redistribution (particle count):")
print(f"    r < 1 AU : {inner_initial:>6}  →  {inner_final:>6}  ({inner_final-inner_initial:+d})")
print(f"    r > 3 AU : {outer_initial:>6}  →  {outer_final:>6}  ({outer_final-outer_initial:+d})")

# ----------------------------------------------------------------------
# Final summary plot: combine everything onto a dashboard
# ----------------------------------------------------------------------
print("\nWriting dashboard ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) aggregate count
axes[0,0].plot(per_frame["yrs"], per_frame["n_agg"], color="steelblue", lw=0.7)
axes[0,0].set_title("Aggregate count")
axes[0,0].set_xlabel("years"); axes[0,0].set_ylabel("count"); axes[0,0].grid(alpha=0.3)

# (b) champion mass
axes[0,1].plot(per_frame["yrs"], per_frame["max_mass_Mearth"], color="crimson", lw=0.7)
axes[0,1].set_title("Heaviest body mass")
axes[0,1].set_xlabel("years"); axes[0,1].set_ylabel("M⊕"); axes[0,1].set_yscale("log"); axes[0,1].grid(alpha=0.3, which="both")

# (c) final disk top-down view
sc = axes[1,0].scatter(final["cx_AU"], final["cz_AU"],
                       s=np.clip(final["n_particles"]*0.3, 1, 200),
                       c=np.log10(final["mass_Mearth"].clip(lower=0.001)),
                       cmap="plasma", alpha=0.7)
axes[1,0].set_aspect("equal")
axes[1,0].set_xlim(-7, 7); axes[1,0].set_ylim(-7, 7)
axes[1,0].set_title(f"Final disk top-down (t={final['sim_time_years'].iloc[0]:.1f} yr)")
axes[1,0].set_xlabel("x (AU)"); axes[1,0].set_ylabel("z (AU)")
plt.colorbar(sc, ax=axes[1,0], label="log₁₀ M⊕")
axes[1,0].plot(0, 0, marker="*", color="yellow", markersize=20, markeredgecolor="black")

# (d) final radial profile
hist_final, _ = np.histogram(final["dist_center_AU"], bins=r_bins,
                             weights=final["n_particles"])
area = 2 * np.pi * r_mid * np.diff(r_bins)
axes[1,1].plot(r_mid, hist_final / area, color="darkgreen", lw=1.5)
axes[1,1].set_title("Final surface density")
axes[1,1].set_xlabel("r (AU)"); axes[1,1].set_ylabel("Σ"); axes[1,1].set_yscale("log")
axes[1,1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(OUT_DIR / "00_dashboard.png", dpi=120)
plt.close()

print(f"\nAll outputs written to: {OUT_DIR}")
print("  00_dashboard.png        — overview")
print("  01_growth_curves.png    — aggregate count + largest size vs time")
print("  02_mass_spectrum.png    — size distribution evolution")
print("  03_radial_profile.png   — surface density / rings & gaps")
print("  04_migration_tracks.png — top-8 body migration & mass tracks")
print("  final_top50.csv         — final-state top-50 list")
