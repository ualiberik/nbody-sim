"""
Two clean overlay schemas — no labels, no axes, no legend.
Schema 1: control (blue) + 0.3 Mю (yellow) + 1.0 Mю (red)
Schema 2: 3.0 Mю (green) + 5.0 Mю (purple) + 10 Mю (orange)
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(r"C:\Projects\nbody-sim\analysis_final_comparison")
OUT.mkdir(exist_ok=True)

ALL_RUNS = {
    "control": (r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv", 5026.5),
    "0.3":     (r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_000833\stats.csv", 5000.0),
    "1.0":     (r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_015429\stats.csv", 5000.0),
    "3.0":     (r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_034408\stats.csv", 5000.0),
    "5.0":     (r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_053325\stats.csv", 5000.0),
    "10":      (r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_073428\stats.csv", 5000.0),
}

def extract(path, target):
    dtypes = {"sim_time_T0": np.float32, "agg_id": np.int32, "n_particles": np.int32,
              "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32}
    rows = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=2_000_000, usecols=list(dtypes.keys())):
        m = (chunk["sim_time_T0"] >= target-0.5) & (chunk["sim_time_T0"] <= target+0.5)
        if m.any(): rows.append(chunk[m])
        if chunk["sim_time_T0"].iloc[-1] > target+0.5 and rows: break
    s = pd.concat(rows, ignore_index=True)
    ct = s.loc[(s["sim_time_T0"] - target).abs().idxmin(), "sim_time_T0"]
    s = s[s["sim_time_T0"] == ct].copy()
    s = s[s["agg_id"] != -1]
    s["mass"] = s["n_particles"].astype(float)
    return s

snaps = {}
for k, (p, t) in ALL_RUNS.items():
    print(f"Loading {k}...", end=" ", flush=True); t0 = time.time()
    snaps[k] = extract(p, t)
    print(f"{time.time()-t0:.0f}s ({len(snaps[k])} agg)")

LIM = 80  # AU radius shown

# --------------------------- SCHEMA on WHITE ---------------------------
def make_schema(out_path, runs_colors):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")
    # Draw central star — small black dot so it doesn't dominate on white
    ax.scatter([0], [0], s=180, c="black", marker="*",
               edgecolors="gold", linewidths=1.5, zorder=20)
    # Draw each population
    for label, color in runs_colors:
        s = snaps[label]
        sizes = np.clip(s["mass"] * 0.5, 1.5, 250)
        ax.scatter(s["cx_AU"], s["cz_AU"], s=sizes,
                   c=color, alpha=0.55, edgecolors="none")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=160, facecolor="white",
                bbox_inches="tight", pad_inches=0)
    plt.close()

# Schema 1: blue / dark-yellow / red — darker shades so they pop on white
make_schema(OUT / "SCHEMA_1_control_03_10_white.png",
            [("control", "#1f5fcc"),    # deep blue
             ("0.3",     "#d4a017"),    # dark mustard yellow
             ("1.0",     "#cc1f1f")])   # deep red

# Schema 2: green / purple / orange
make_schema(OUT / "SCHEMA_2_30_50_100_white.png",
            [("3.0", "#1f9a4d"),        # forest green
             ("5.0", "#8a1fcc"),        # deep purple
             ("10",  "#e07000")])       # deep orange

print(f"\nSchemas saved:")
print(f"  {OUT}\\SCHEMA_1_control_03_10.png")
print(f"  {OUT}\\SCHEMA_2_30_50_100.png")
