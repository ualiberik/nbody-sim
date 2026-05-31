"""
Plot: how the disk-mass budget splits as a function of perturber mass.
  Left  : % of disk that ended up in PLANETS (≥1 M_earth bodies, grouped by class)
  Right : % of disk ejected to far orbits
Plus a stacked-bar showing the full radial mass budget.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(r"C:\Projects\nbody-sim\analysis_final_comparison")
OUT.mkdir(exist_ok=True)

# All metrics from full_comparison.py + test_hypothesis_h1.py, hard-coded here for clarity.
# (Recomputing these from raw CSVs takes ~3 minutes; the metrics were stable in print output.)
M_DISK = 10000.0   # M_earth total disk

labels = ["control", "0.3 M_J", "1.0 M_J", "3.0 M_J", "5.0 M_J", "10 M_J"]
M_jup_axis = np.array([0.0, 0.3, 1.0, 3.0, 5.0, 10.0])

# Per-aggregate mass class fractions (% of initial disk = 10000 M_earth)
#   Bodies with these masses ANYWHERE in the system:
#   Earth   (1-3 M_earth)
#   sEarth  (3-10)
#   Neptune (10-30)
#   Saturn  (30-95)
#   Jupiter (95-300)
#   sJupiter(300+)
# Values come from each analysis_*/final_top30.csv stratified by mass class
# (full distributions from the analyze_experiment.py outputs)
class_fracs = {
    # label:        Earth   sEarth  Neptune Saturn  Jupiter sJupit
    "control":     [11.28,  12.23,   2.10,   4.00,   0.00,   0.00],  # at 1592 yr
    "0.3 M_J":     [12.74,  26.87,   9.77,  11.67,  12.71,   0.00],  # at 800 yr
    "1.0 M_J":     [14.40,  18.89,  10.68,  21.92,   2.58,   0.00],
    "3.0 M_J":     [10.84,  20.31,   5.47,   5.06,   9.65,   0.00],
    "5.0 M_J":     [15.42,  33.41,   7.62,  11.46,   0.95,   0.00],
    "10 M_J":      [10.98,  21.94,   7.75,  13.27,   2.58,   0.00],
}

# Ejected mass fractions (r > 30 AU)
eject_pct = np.array([2.8, 2.8, 66.85, 5.57, 58.17, 56.46])

# Mass in planets ≥ 1 M_earth (sum of all classes above)
planet_pct = np.array([sum(v) for v in class_fracs.values()])
# (Also: M_giant_total ≥ 30 M_earth fraction, recomputed)
giant_pct = np.array([4.00, 24.38, 22.93, 14.71, 12.41, 15.85])    # %disk in giants ≥30
sgiant_pct = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])         # final supergiants

# Radial mass distribution (% of disk in each annulus)
# From test_hypothesis_h1 output
zone_bounds = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 100), (100, 300)]
radial_data = {
    "control":  [ 3, 22, 10,  2,  0,  1,  1,  0],
    "0.3 M_J":  [ 0,  0,  3, 62,  6,  3,  0,  0],
    "1.0 M_J":  [ 0,  0,  0,  0,  0, 64,  3,  0],
    "3.0 M_J":  [ 1, 15, 21,  7,  3,  5,  0,  0],
    "5.0 M_J":  [ 0,  0,  0,  0, 10, 48, 10,  0],
    "10 M_J":   [ 0,  0,  0,  0,  0, 53,  4,  0],
}

# ----------------------------------------------------------------------
# Figure 1 — Mass budget bar charts (two panels)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(labels)))

# Left: planets vs ejected
ax = axes[0]
x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, planet_pct, w, color="steelblue",  edgecolor="black", label="In planets (≥1 M⊕)")
b2 = ax.bar(x + w/2, eject_pct,  w, color="crimson",    edgecolor="black", label="Ejected (r > 30 AU)")
for bar, v in zip(b1, planet_pct):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.7, f"{v:.1f}%", ha="center", fontsize=9)
for bar, v in zip(b2, eject_pct):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.7, f"{v:.1f}%", ha="center", fontsize=9, color="darkred")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("% of initial disk mass (10 000 M⊕)")
ax.set_title("Mass budget: planet capture vs ejection")
ax.set_ylim(0, 80)
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper left")
# Add note about the "missing" mass (lost to dust/small singletons)
ax.text(0.5, 73, "Note: remainder is small singletons / unbound dust",
        ha="center", fontsize=8, style="italic", color="gray")

# Right: only gas giants (≥30 M_earth) - same data as left but cleaner
ax2 = axes[1]
b3 = ax2.bar(x, giant_pct, color=colors, edgecolor="black")
for bar, v in zip(b3, giant_pct):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("% of initial disk in gas-giant bodies (≥30 M⊕)")
ax2.set_title("Planet-formation efficiency: mass captured by gas giants")
ax2.set_ylim(0, 30)
ax2.grid(alpha=0.3, axis="y")
# Highlight the maximum
imax = int(np.argmax(giant_pct))
ax2.annotate("OPTIMUM",
             xy=(imax, giant_pct[imax]),
             xytext=(imax+1, giant_pct[imax]+5),
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "05_mass_budget.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# Figure 2 — Stacked horizontal bars: % disk in each radial zone
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7))
zone_colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(zone_bounds)))
left = np.zeros(len(labels))
for zi, ((lo, hi), c) in enumerate(zip(zone_bounds, zone_colors)):
    vals = np.array([radial_data[L][zi] for L in labels], dtype=float)
    bars = ax.barh(labels, vals, left=left, color=c,
                    edgecolor="white", linewidth=0.5,
                    label=f"r = {lo}-{hi} AU")
    # annotate non-zero
    for bar, v in zip(bars, vals):
        if v >= 4:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                    f"{int(v)}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    left += vals

ax.set_xlabel("% of initial disk mass (10 000 M⊕)")
ax.set_title("Radial distribution of remaining mass — by perturber mass")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.set_xlim(0, max(left)*1.05)
ax.grid(alpha=0.3, axis="x")
ax.invert_yaxis()                # control on top
plt.tight_layout()
plt.savefig(OUT / "06_radial_stacked.png", dpi=130)
plt.close()

# ----------------------------------------------------------------------
# Figure 3 — Line plot vs perturber mass on log axis
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
# Use log axis so 0 → 0.3 → 1 → 3 → 5 → 10 looks reasonable
M_disp = M_jup_axis.copy()
M_disp[0] = 0.05    # log-friendly placeholder for control

ax.plot(M_disp, planet_pct, '-o', color="steelblue", lw=2, markersize=10,
        label="Mass in planets (≥1 M⊕)")
ax.plot(M_disp, giant_pct,  '-s', color="darkgreen", lw=2, markersize=10,
        label="Mass in gas giants (≥30 M⊕)")
ax.plot(M_disp, eject_pct,  '-D', color="crimson", lw=2, markersize=10,
        label="Mass ejected (r > 30 AU)")

ax.set_xscale("log")
ax.set_xticks([0.05, 0.3, 1.0, 3.0, 5.0, 10.0])
ax.set_xticklabels(["0 (control)", "0.3", "1.0", "3.0", "5.0", "10"])
ax.set_xlabel("Perturber mass (M_Jup)")
ax.set_ylabel("% of initial disk mass")
ax.set_title("How disk mass is partitioned vs. perturber mass")
ax.set_ylim(0, 80)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
# Mark transitions
ax.axvspan(0.5, 2.0, alpha=0.1, color="red", label="catastrophe zone")
ax.axvspan(2.0, 4.0, alpha=0.1, color="green", label="stable window")
ax.axvspan(4.0, 15, alpha=0.1, color="orange", label="chaos zone")
ax.text(1.0, 75, "CATASTROPHE", ha="center", color="darkred", fontweight="bold")
ax.text(3.0, 75, "STABLE", ha="center", color="darkgreen", fontweight="bold")
ax.text(7.0, 75, "CHAOS", ha="center", color="darkorange", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "07_budget_line.png", dpi=130)
plt.close()

print("Plots written:")
for f in ["05_mass_budget.png", "06_radial_stacked.png", "07_budget_line.png"]:
    print(f"  {OUT}\\{f}")
