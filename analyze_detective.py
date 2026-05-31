"""
Detective story: what happened to the 919 M_earth super-planet?
Also: hunt for inner rings near the star, plot the resonant chain,
visualize the most dramatic single trajectory.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r"C:\Projects\nbody-sim\data\20260525_185147")
OUT_DIR  = Path(r"C:\Projects\nbody-sim\analysis_output")
STATS    = DATA_DIR / "stats.csv"

print("Loading stats.csv ...")
df = pd.read_csv(STATS)
groups = {fr: g for fr, g in df.groupby("frame")}
frames = sorted(groups.keys())
last_frame = frames[-1]

per_frame = df.groupby("frame").agg(
    t=("sim_time_T0","first"), yrs=("sim_time_years","first"),
    n_agg=("agg_id","count"),
    max_mass=("mass_Mearth","max"),
    sec_mass=("mass_Mearth", lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0),
    third_mass=("mass_Mearth", lambda s: s.nlargest(3).iloc[-1] if len(s) > 2 else 0),
).reset_index()

# ----------------------------------------------------------------------
# 1.  THE DETECTIVE: trace the heaviest body backwards from its peak.
# ----------------------------------------------------------------------
peak_frame = per_frame.loc[per_frame["max_mass"].idxmax(), "frame"]
peak_mass  = per_frame["max_mass"].max()
print(f"\n=== The super-planet ===")
print(f"  Peak mass {peak_mass:.1f} M⊕ reached at frame {peak_frame} "
      f"(t = {per_frame.loc[per_frame['frame']==peak_frame,'yrs'].iloc[0]:.2f} yr)")

# Find the row at peak
peak_row = df[df["frame"] == peak_frame].nlargest(1, "mass_Mearth").iloc[0]
print(f"  Position at peak: r = {peak_row['dist_center_AU']:.3f} AU")
print(f"  Velocity         : v = {np.sqrt(peak_row['vx']**2+peak_row['vy']**2+peak_row['vz']**2):.3f}")

# Forward-track from the peak: follow it until it vanishes or shrinks.
def track_body(start_frame, start_row, direction=+1):
    """Walk frame-by-frame, picking the nearest similar-mass body each step.
       Returns lists (t, r, mass, x, y, z, n_particles)."""
    t_arr, r_arr, m_arr, x_arr, y_arr, z_arr, n_arr = [], [], [], [], [], [], []
    cx, cy, cz = start_row["cx_AU"], start_row["cy_AU"], start_row["cz_AU"]
    mass = start_row["mass_Mearth"]
    fr = start_frame
    while True:
        fr += direction
        if fr < 0 or fr > last_frame: break
        g = groups[fr]
        dx = g["cx_AU"].values - cx
        dy = g["cy_AU"].values - cy
        dz = g["cz_AU"].values - cz
        d  = np.sqrt(dx*dx + dy*dy + dz*dz)
        # Allow large displacement for fast bodies; ignore extreme jumps
        mask = d < 1.0
        if not mask.any(): break
        idx_sub = np.where(mask)[0]
        sub = g.iloc[idx_sub]
        # Score: prefer close + similar mass
        log_mr = np.abs(np.log(sub["mass_Mearth"].values / max(mass, 1e-9)))
        score  = d[idx_sub] + 0.3 * log_mr
        best   = idx_sub[np.argmin(score)]
        row = g.iloc[best]
        cx, cy, cz = row["cx_AU"], row["cy_AU"], row["cz_AU"]
        mass = row["mass_Mearth"]
        t_arr.append(row["sim_time_years"])
        r_arr.append(np.sqrt(cx*cx + cz*cz))
        m_arr.append(mass)
        x_arr.append(cx); y_arr.append(cy); z_arr.append(cz)
        n_arr.append(row["n_particles"])
    return map(np.array, (t_arr, r_arr, m_arr, x_arr, y_arr, z_arr, n_arr))

# Backwards: from peak to t=0
tb,rb,mb,xb,yb,zb,nb = track_body(peak_frame, peak_row, direction=-1)
# Forwards: from peak to t=end
tf,rf,mf,xf,yf,zf,nf = track_body(peak_frame, peak_row, direction=+1)

# Combine
t_all = np.concatenate([tb[::-1], [df[df["frame"]==peak_frame]["sim_time_years"].iloc[0]], tf])
r_all = np.concatenate([rb[::-1], [peak_row['dist_center_AU']], rf])
m_all = np.concatenate([mb[::-1], [peak_mass], mf])
x_all = np.concatenate([xb[::-1], [peak_row['cx_AU']], xf])
z_all = np.concatenate([zb[::-1], [peak_row['cz_AU']], zf])
n_all = np.concatenate([nb[::-1], [peak_row['n_particles']], nf])

# Find the moment of catastrophe — when mass drops by > 50 % in one frame
dm  = np.diff(m_all)
dmr = dm / m_all[:-1]
catastrophe = np.where(dmr < -0.5)[0]
if len(catastrophe):
    k = catastrophe[0]
    print(f"\n  CATASTROPHE detected at t = {t_all[k+1]:.2f} yr")
    print(f"    mass before/after: {m_all[k]:.1f} M⊕ -> {m_all[k+1]:.1f} M⊕")
    print(f"    radius before/after: r = {r_all[k]:.3f} AU -> {r_all[k+1]:.3f} AU")
else:
    print(f"\n  No catastrophic loss — body stayed coherent.")
    print(f"  Final state: mass = {m_all[-1]:.1f} M⊕, r = {r_all[-1]:.3f} AU")

# Plot the life-story
fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
ax[0].plot(t_all, m_all, color="crimson", lw=1.2)
ax[0].axvline(t_all[len(tb)], color="black", ls="--", alpha=0.4, label=f"peak ({peak_mass:.0f} M⊕)")
ax[0].set_ylabel("Mass (M⊕)")
ax[0].set_title("Life story of the heaviest body that ever existed")
ax[0].grid(alpha=0.3)
ax[0].legend()
ax[0].set_yscale("log")

ax[1].plot(t_all, r_all, color="navy", lw=1.2)
ax[1].axvline(t_all[len(tb)], color="black", ls="--", alpha=0.4)
ax[1].set_ylabel("Orbital radius (AU)")
ax[1].grid(alpha=0.3)

# Number of constituent particles - shows accretion vs disruption events
ax[2].plot(t_all, n_all, color="darkgreen", lw=1.2)
ax[2].axvline(t_all[len(tb)], color="black", ls="--", alpha=0.4)
ax[2].set_ylabel("# constituent particles")
ax[2].set_xlabel("Time (years)")
ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_superplanet_life.png", dpi=120)
plt.close()

# X-Z trajectory in the disk plane
fig, ax = plt.subplots(figsize=(9, 9))
sc = ax.scatter(x_all, z_all, c=t_all, s=np.clip(m_all*0.3, 4, 200),
                cmap="plasma", alpha=0.8, edgecolors="none")
ax.plot(0, 0, marker="*", color="gold", markersize=25, markeredgecolor="black", zorder=10)
ax.set_aspect("equal")
ax.set_xlabel("x (AU)"); ax.set_ylabel("z (AU)")
ax.set_title("Trajectory of the super-planet (color=time, size=mass)")
ax.grid(alpha=0.3)
plt.colorbar(sc, label="time (yr)")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_superplanet_trajectory.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 2.  HOT INNER REGION: is there an inner ring near the star?
# ----------------------------------------------------------------------
print("\n=== Inner region (r < 0.5 AU) ===")
inner_history = []
for fr in frames[::50]:  # every 50 frames
    g = groups[fr]
    inner = g[g["dist_center_AU"] < 0.5]
    if len(inner) > 0:
        inner_history.append({
            "yrs": g["sim_time_years"].iloc[0],
            "n_bodies": len(inner),
            "n_particles": inner["n_particles"].sum(),
            "min_r": inner["dist_center_AU"].min(),
            "median_r": inner["dist_center_AU"].median(),
        })
ihist = pd.DataFrame(inner_history)
print(ihist.tail(10).to_string(index=False))

# Plot: inner-region timeline + final fine-grained radial profile near star
fig, ax = plt.subplots(2, 1, figsize=(11, 9))
ax[0].plot(ihist["yrs"], ihist["n_particles"], color="darkorange", lw=1.2, label="# particles r<0.5 AU")
ax[0].plot(ihist["yrs"], ihist["n_bodies"], color="purple", lw=1.2, label="# aggregates r<0.5 AU")
ax[0].set_xlabel("Time (yr)")
ax[0].set_ylabel("count")
ax[0].set_title("Inner-region accumulation (r < 0.5 AU)")
ax[0].grid(alpha=0.3)
ax[0].legend()

# Very fine radial profile of the final frame, zoomed inside 2 AU
final = df[df["frame"] == last_frame]
fine_bins = np.linspace(0, 2, 81)
r_mid_fine = 0.5*(fine_bins[1:] + fine_bins[:-1])
hist_f, _ = np.histogram(final["dist_center_AU"], bins=fine_bins,
                          weights=final["n_particles"])
area = 2 * np.pi * r_mid_fine * np.diff(fine_bins)
ax[1].plot(r_mid_fine, hist_f / area, color="darkblue", lw=1.4)
ax[1].set_xlabel("r (AU)")
ax[1].set_ylabel("Σ (particles / AU²)")
ax[1].set_title("Final-state fine-grained radial profile (0–2 AU)")
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_inner_region.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 3.  RESONANT CHAIN: how locked-in is the architecture?
# ----------------------------------------------------------------------
print("\n=== Resonant architecture of survivors ===")
final_sorted = final.sort_values("dist_center_AU").reset_index(drop=True)
# Keep only "planets" — bodies with > 5 particles (= > 5 M⊕)
planets = final_sorted[final_sorted["n_particles"] >= 5].reset_index(drop=True)
planets["r"]  = np.sqrt(planets["cx_AU"]**2 + planets["cz_AU"]**2)
planets["T"]  = planets["r"]**1.5
planets = planets.sort_values("r").reset_index(drop=True)
print(f"  Planet-class bodies (>=5 M⊕): {len(planets)}")

# Period ratio of each consecutive pair
pairs = []
for i in range(len(planets)-1):
    ratio = planets["T"].iloc[i+1] / planets["T"].iloc[i]
    pairs.append((i, ratio, planets["r"].iloc[i], planets["r"].iloc[i+1],
                  planets["mass_Mearth"].iloc[i], planets["mass_Mearth"].iloc[i+1]))

# Plot consecutive ratios — peaks at simple fractions reveal resonances
fig, ax = plt.subplots(figsize=(12, 6))
xs = np.arange(len(pairs))
ys = [p[1] for p in pairs]
ax.scatter(xs, ys, c="navy", s=40)
for ref, lbl in [(2.0, "2:1"), (1.5, "3:2"), (1.333, "4:3"), (1.25, "5:4"),
                 (1.4, "7:5"), (1.667, "5:3"), (1.2, "6:5")]:
    ax.axhline(ref, ls="--", alpha=0.4, color="red")
    ax.text(len(pairs)-0.5, ref, f" {lbl}", va="center", color="red")
ax.set_xlabel("planet-pair index (inner -> outer)")
ax.set_ylabel("period ratio T_{outer} / T_{inner}")
ax.set_title(f"Period-ratio chain of {len(planets)} surviving planets")
ax.set_ylim(1.0, 2.5)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "08_resonant_chain.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 4. Disk-scale-height evolution: vertical heating
# ----------------------------------------------------------------------
print("\n=== Vertical disk heating ===")
h_history = []
for fr in frames[::20]:
    g = groups[fr]
    # use mass-weighted variance of y (vertical axis in disk frame).
    # y is "cy_AU" because the disk lies in the XZ plane.
    w = g["n_particles"].values.astype(float)
    y = g["cy_AU"].values
    h = np.sqrt(np.average(y**2, weights=w))
    h_history.append({"yrs": g["sim_time_years"].iloc[0], "h_rms": h})
hh = pd.DataFrame(h_history)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(hh["yrs"], hh["h_rms"], color="darkgreen", lw=1.2)
ax.set_xlabel("Time (years)")
ax.set_ylabel("RMS vertical thickness (AU)")
ax.set_title("Disk vertical heating (mass-weighted ⟨y²⟩^½)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "09_disk_heating.png", dpi=120)
plt.close()
print(f"  initial h_rms = {hh['h_rms'].iloc[0]:.4f} AU")
print(f"  final   h_rms = {hh['h_rms'].iloc[-1]:.4f} AU  "
      f"(x{hh['h_rms'].iloc[-1]/hh['h_rms'].iloc[0]:.1f} thicker)")

# ----------------------------------------------------------------------
# 5.  Plot all three largest tracks together (1st, 2nd, 3rd by mass)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(per_frame["yrs"], per_frame["max_mass"], color="crimson", lw=0.8, label="1st heaviest")
ax.plot(per_frame["yrs"], per_frame["sec_mass"], color="steelblue", lw=0.8, label="2nd heaviest")
ax.plot(per_frame["yrs"], per_frame["third_mass"], color="darkgreen", lw=0.8, label="3rd heaviest")
ax.set_yscale("log")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Mass (M⊕)")
ax.set_title("Top-3 heaviest-body mass over time")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUT_DIR / "10_top3_mass.png", dpi=120)
plt.close()

print("\nAll detective plots written to", OUT_DIR)
print("  05_superplanet_life.png       — mass/radius/size history of peak body")
print("  06_superplanet_trajectory.png — X-Z trajectory of the super-planet")
print("  07_inner_region.png           — what happened near the star")
print("  08_resonant_chain.png         — period-ratio chain of survivors")
print("  09_disk_heating.png           — disk thickness evolution")
print("  10_top3_mass.png              — top-3 mass tracks")
