"""
Heatmap of disk-mass distribution across radial bins for each perturber mass.
Plus a complementary "polar" radial profile plot.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

OUT = Path(r"C:\Projects\nbody-sim\analysis_final_comparison")
OUT.mkdir(exist_ok=True)

RUNS = [
    ("control",  0.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv", 5026.5),
    ("0.3 Mю",   0.3,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_000833\stats.csv", 5000.0),
    ("1.0 Mю",   1.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_015429\stats.csv", 5000.0),
    ("3.0 Mю",   3.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_034408\stats.csv", 5000.0),
    ("5.0 Mю",   5.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_053325\stats.csv", 5000.0),
    ("10  Mю",   10.0, r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_073428\stats.csv", 5000.0),
]

def extract(path, target):
    dtypes = {"sim_time_T0": np.float32, "agg_id": np.int32, "n_particles": np.int32,
              "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32}
    rows = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=2_000_000, usecols=list(dtypes.keys())):
        m = (chunk["sim_time_T0"] >= target - 0.5) & (chunk["sim_time_T0"] <= target + 0.5)
        if m.any(): rows.append(chunk[m])
        if chunk["sim_time_T0"].iloc[-1] > target + 0.5 and rows: break
    s = pd.concat(rows, ignore_index=True)
    ct = s.loc[(s["sim_time_T0"] - target).abs().idxmin(), "sim_time_T0"]
    s = s[s["sim_time_T0"] == ct].copy()
    s = s[s["agg_id"] != -1]
    s["mass"] = s["n_particles"].astype(float)
    s["r"]    = np.sqrt(s["cx_AU"]**2 + s["cz_AU"]**2)
    return s

# Logarithmic radial bins
r_edges = np.logspace(np.log10(0.5), np.log10(150), 31)   # 30 bins from 0.5 AU to 150 AU
r_mid   = np.sqrt(r_edges[1:] * r_edges[:-1])

# Collect histograms
labels = [r[0] for r in RUNS]
hist_mass  = np.zeros((len(RUNS), len(r_edges)-1))   # mass per bin
hist_count = np.zeros((len(RUNS), len(r_edges)-1))   # count per bin
hist_sigma = np.zeros((len(RUNS), len(r_edges)-1))   # surface density (M⊕/AU²)

# Surface density normalization: bin area = π(r_out² - r_in²)
bin_area = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)

for i, (label, mj, path, tT) in enumerate(RUNS):
    print(f"Loading {label}...", end=" ", flush=True); t0 = time.time()
    s = extract(path, tT)
    print(f"{time.time()-t0:.0f}s")
    # mass-weighted histogram
    h_m, _ = np.histogram(s["r"], bins=r_edges, weights=s["mass"])
    h_n, _ = np.histogram(s["r"], bins=r_edges)
    hist_mass[i]  = h_m
    hist_count[i] = h_n
    hist_sigma[i] = h_m / bin_area

# ----------------------------------------------------------------------
# FIG 1 — Mass density heatmap (the main result)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
# Use LogNorm for dynamic range, clamp tiny values
data = hist_mass.copy()
data[data < 0.5] = 0.5    # avoid log(0)

# Show with imshow (extent = log10 radius)
extent = [np.log10(r_edges[0]), np.log10(r_edges[-1]), -0.5, len(RUNS)-0.5]
im = ax.imshow(data, aspect="auto", origin="lower", cmap="magma",
               norm=LogNorm(vmin=1, vmax=data.max()),
               extent=extent, interpolation="nearest")

# Y-axis: run labels
ax.set_yticks(range(len(RUNS)))
ax.set_yticklabels(labels, fontsize=12)

# X-axis: log radius — convert tick labels
xticks_au = [0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100]
ax.set_xticks([np.log10(x) for x in xticks_au])
ax.set_xticklabels([str(x) for x in xticks_au], fontsize=11)
ax.set_xlabel("Орбитальный радиус (а.е., логарифмическая шкала)", fontsize=12)
ax.set_ylabel("Масса возмутителя", fontsize=12)
ax.set_title("Тепловая карта: масса в радиальных бинах для каждого возмутителя",
             fontsize=13, fontweight="bold")

# Mark perturber orbit (r=20 AU) with a vertical line in all rows
ax.axvline(np.log10(20.0), color="lime", ls="--", lw=2, alpha=0.7, label="орбита возмутителя (20 а.е.)")

# Mark MMR resonances on log-x
mmrs = [
    ("1:2", 20.0/2**(2/3)),
    ("3:5", 20.0*(0.6)**(2/3)),
    ("2:3", 20.0*(2/3)**(2/3)),
    ("2:1", 20.0*2**(2/3)),
    ("3:1", 20.0*3**(2/3)),
    ("4:1", 20.0*4**(2/3)),
    ("5:1", 20.0*5**(2/3)),
]
for name, r in mmrs:
    if 0.5 < r < 150:
        ax.axvline(np.log10(r), color="cyan", ls=":", lw=1, alpha=0.5)
        ax.text(np.log10(r), len(RUNS)-0.4, name, ha="center", color="cyan", fontsize=8)

cbar = plt.colorbar(im, ax=ax, pad=0.01)
cbar.set_label("Масса в бине (M_земли, log)", fontsize=11)
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(OUT / "HEATMAP1_mass_per_bin.png", dpi=140)
plt.close()

# ----------------------------------------------------------------------
# FIG 2 — surface density profile heatmap (mass per AU² — physical)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
data2 = hist_sigma.copy()
data2[data2 < 1e-3] = 1e-3
im2 = ax.imshow(data2, aspect="auto", origin="lower", cmap="viridis",
                norm=LogNorm(vmin=0.01, vmax=data2.max()),
                extent=extent, interpolation="bilinear")
ax.set_yticks(range(len(RUNS))); ax.set_yticklabels(labels, fontsize=12)
ax.set_xticks([np.log10(x) for x in xticks_au]); ax.set_xticklabels([str(x) for x in xticks_au])
ax.set_xlabel("Орбитальный радиус (а.е., лог. шкала)", fontsize=12)
ax.set_ylabel("Масса возмутителя", fontsize=12)
ax.set_title("Поверхностная плотность массы Σ (M⊕ / а.е.²)",
             fontsize=13, fontweight="bold")
ax.axvline(np.log10(20.0), color="lime", ls="--", lw=2, alpha=0.7)
for name, r in mmrs:
    if 0.5 < r < 150:
        ax.axvline(np.log10(r), color="white", ls=":", lw=1, alpha=0.4)
cbar = plt.colorbar(im2, ax=ax)
cbar.set_label("Σ (M⊕ / а.е.²)")
plt.tight_layout()
plt.savefig(OUT / "HEATMAP2_surface_density.png", dpi=140)
plt.close()

# ----------------------------------------------------------------------
# FIG 3 — Polar plot (radius vs angle, mass = colour/size)
# Show actual positions of every body, six panels
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection="polar"))
for i, ((label, mj, path, tT), ax) in enumerate(zip(RUNS, axes.flat)):
    s = extract(path, tT)
    phi = np.arctan2(s["cz_AU"], s["cx_AU"])
    r = s["r"]
    ax.scatter(phi, r, s=np.clip(s["mass"]*0.5, 1, 400),
               c=np.log10(s["mass"].clip(lower=1)), cmap="plasma", alpha=0.7, edgecolors="none")
    if mj > 0:
        # draw perturber orbit
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(theta, np.ones_like(theta)*20, color="lime", ls="--", lw=1, alpha=0.7)
    ax.set_title(f"{label}  ({len(s)} тел)", fontsize=12, pad=15)
    ax.set_rmax(80)
    ax.set_rticks([10, 20, 30, 50, 80])
    ax.set_rlabel_position(135)
    ax.grid(alpha=0.3)

plt.suptitle("Полярные карты систем: радиус × угол × масса (логарифм цвет)",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(OUT / "HEATMAP3_polar_panels.png", dpi=130)
plt.close()

print(f"\nГрафики сохранены в {OUT}:")
for f in ["HEATMAP1_mass_per_bin.png", "HEATMAP2_surface_density.png", "HEATMAP3_polar_panels.png"]:
    print(f"  {f}")
