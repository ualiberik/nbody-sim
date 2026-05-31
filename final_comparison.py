"""Final comparison plot across all 5 perturber experiments + control."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(r"C:\Projects\nbody-sim\analysis_final_comparison")
OUT.mkdir(exist_ok=True)

# Each entry: (label, perturber_M_Jup, per_frame_csv, final_top30_csv)
RUNS = [
    ("control",  0.0, r"C:\Projects\nbody-sim\analysis_control\per_frame_summary.csv",
                       r"C:\Projects\nbody-sim\analysis_control\final_top30.csv"),
    ("0.3 M_Jup", 0.3, r"C:\Projects\nbody-sim\analysis_exp1_03Mjup\per_frame_summary.csv",
                       r"C:\Projects\nbody-sim\analysis_exp1_03Mjup\final_top30.csv"),
    ("1.0 M_Jup", 1.0, r"C:\Projects\nbody-sim\analysis_exp2_10Mjup\per_frame_summary.csv",
                       r"C:\Projects\nbody-sim\analysis_exp2_10Mjup\final_top30.csv"),
    ("3.0 M_Jup", 3.0, r"C:\Projects\nbody-sim\analysis_exp3_30Mjup\per_frame_summary.csv",
                       r"C:\Projects\nbody-sim\analysis_exp3_30Mjup\final_top30.csv"),
    ("5.0 M_Jup", 5.0, r"C:\Projects\nbody-sim\analysis_exp4_50Mjup\per_frame_summary.csv",
                       r"C:\Projects\nbody-sim\analysis_exp4_50Mjup\final_top30.csv"),
    ("10 M_Jup",  10.0, r"C:\Projects\nbody-sim\analysis_exp5_100Mjup\per_frame_summary.csv",
                       r"C:\Projects\nbody-sim\analysis_exp5_100Mjup\final_top30.csv"),
]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(RUNS)))

# ----------------------------------------------------------------------
# FIG 1 — n_agg and max_mass over time for all runs
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
for (label, mj, pf_path, _), c in zip(RUNS, colors):
    pf = pd.read_csv(pf_path)
    axes[0].plot(pf["yrs"], pf["n_agg"],    color=c, lw=0.6, label=label, alpha=0.85)
    axes[1].plot(pf["yrs"], pf["max_mass"], color=c, lw=0.6, label=label, alpha=0.85)

axes[0].set_ylabel("Aggregate count")
axes[0].set_title("Evolution comparison — all runs")
axes[0].grid(alpha=0.3)
axes[0].legend(ncol=2, fontsize=9)
axes[1].set_yscale("log")
axes[1].set_ylabel("Heaviest body (M⊕)")
axes[1].set_xlabel("Time (years)")
axes[1].grid(alpha=0.3, which="both")
axes[1].legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "01_evolution_all.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# FIG 2 — final architecture (radius of every aggregate ≥ 5 M_earth)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
y_pos = np.arange(len(RUNS))
for (label, mj, _, top_path), c, yp in zip(RUNS, colors, y_pos):
    top = pd.read_csv(top_path)
    top["mass"] = top["n_particles"].astype(float)
    top["r"]    = np.sqrt(top["cx_AU"]**2 + top["cz_AU"]**2)
    mask = top["mass"] >= 5
    sub = top[mask]
    sizes = np.clip(sub["mass"] * 1.2, 8, 600)
    ax.scatter(sub["r"], np.full(len(sub), yp), s=sizes, c=[c], alpha=0.7,
               edgecolors="black", linewidth=0.4)

# perturber positions (vertical line at r=20)
ax.axvline(20.0, color="limegreen", ls="--", alpha=0.5, label="perturber orbit")

# MMR markers (only those visible in plot)
mmr = [(0.5, "1:2"), (0.333, "1:3"), (0.667, "2:3"), (0.25, "1:4"),
       (0.6, "3:5"), (2.0, "2:1"), (3.0, "3:1"), (4.0, "4:1"), (5.0, "5:1")]
for ratio, name in mmr:
    r = 20.0 * ratio**(2/3) if ratio < 1 else 20.0 * ratio**(2/3)
    if 1 < r < 80:
        ax.axvline(r, color="red", ls=":", alpha=0.3)
        ax.text(r, -0.7, name, ha="center", color="red", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels([r[0] for r in RUNS])
ax.set_xscale("log")
ax.set_xlim(1.0, 80)
ax.set_xlabel("Orbital radius (AU)  — size ∝ mass")
ax.set_title("Final architecture — every aggregate ≥ 5 M⊕")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "02_architecture_panel.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# FIG 3 — mass-vs-radius scatter for all runs side-by-side
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
ss = [("Mer",0.39,0.055),("Ven",0.72,0.82),("Earth",1.0,1.0),("Mars",1.52,0.107),
      ("Jup",5.2,318),("Sat",9.5,95),("Ura",19.2,14.5),("Nep",30.1,17)]
for ((label, mj, _, top_path), c, ax) in zip(RUNS, colors, axes.flat):
    top = pd.read_csv(top_path)
    top["mass"] = top["n_particles"].astype(float)
    top["r"]    = np.sqrt(top["cx_AU"]**2 + top["cz_AU"]**2)
    ax.scatter(top["r"], top["mass"], s=np.clip(top["mass"]*0.5, 4, 400),
               c=[c], alpha=0.7, edgecolors="black", linewidth=0.3)
    if mj > 0:
        # perturber as green diamond
        pmass_mearth = mj * 318.0
        ax.scatter([20.0], [pmass_mearth], s=400, c="limegreen",
                   marker="D", edgecolors="black", linewidth=1.5, zorder=10)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.3, 100); ax.set_ylim(0.8, 2500)
    ax.set_title(label)
    ax.grid(alpha=0.3, which="both")
    for name, r, m in ss:
        ax.plot(r, m, marker="*", color="black", ms=8, mfc="none", mew=1)

for ax in axes[-1, :]:
    ax.set_xlabel("Orbital radius (AU)")
for ax in axes[:, 0]:
    ax.set_ylabel("Mass (M⊕)")
plt.suptitle("Mass vs radius — final state of each run (★ = real Solar System)")
plt.tight_layout()
plt.savefig(OUT / "03_mass_radius_grid.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# FIG 4 — Summary trend bar chart
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

n_giants = []
biggest = []
median_r = []
ejected = []
for (label, mj, _, top_path) in RUNS:
    top = pd.read_csv(top_path)
    top["mass"] = top["n_particles"].astype(float)
    top["r"]    = np.sqrt(top["cx_AU"]**2 + top["cz_AU"]**2)
    g = top[top["mass"] >= 30]
    n_giants.append(len(g))
    biggest.append(top["mass"].max())
    median_r.append(g["r"].median() if len(g) else 0)
    ejected.append((top["r"] > 30).sum())

xs = [r[0] for r in RUNS]
axes[0,0].bar(xs, n_giants, color=colors)
axes[0,0].set_title("Number of gas giants ≥30 M⊕")
axes[0,0].set_ylabel("count")
axes[0,0].grid(alpha=0.3, axis="y")

axes[0,1].bar(xs, biggest, color=colors)
axes[0,1].set_title("Mass of heaviest body (final)")
axes[0,1].set_ylabel("M⊕")
axes[0,1].grid(alpha=0.3, axis="y")

axes[1,0].bar(xs, median_r, color=colors)
axes[1,0].set_title("Median radius of gas giants")
axes[1,0].set_ylabel("AU")
axes[1,0].grid(alpha=0.3, axis="y")
axes[1,0].axhline(20, color="limegreen", ls="--", alpha=0.6, label="perturber orbit")
axes[1,0].legend()

axes[1,1].bar(xs, ejected, color=colors)
axes[1,1].set_title("Aggregates ejected to r > 30 AU")
axes[1,1].set_ylabel("count")
axes[1,1].grid(alpha=0.3, axis="y")

plt.suptitle("Trend across perturber mass — final state at t ≈ 800 yr (control extends further)")
plt.tight_layout()
plt.savefig(OUT / "04_trend_bars.png", dpi=120)
plt.close()

print("All comparison plots saved to:", OUT)
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
