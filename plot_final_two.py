"""
Two clean Russian-labeled charts:
  1) Mass budget (correct, non-overlapping): % in inner planets + % ejected
  2) Aggregate counts in the disk
Recomputes everything from raw stats.csv for correctness.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
M_DISK = 10000.0           # M⊕ — initial disk mass
R_EJECT = 30.0             # AU — boundary for "ejected"

def extract_frame(path, target_T0, tol=0.5):
    dtypes = {"sim_time_T0": np.float32, "agg_id": np.int32, "n_particles": np.int32,
              "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32}
    cols = list(dtypes.keys())
    rows = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=2_000_000, usecols=cols):
        m = (chunk["sim_time_T0"] >= target_T0 - tol) & (chunk["sim_time_T0"] <= target_T0 + tol)
        if m.any(): rows.append(chunk[m])
        if chunk["sim_time_T0"].iloc[-1] > target_T0 + tol and rows: break
    snap = pd.concat(rows, ignore_index=True)
    closest_t = snap.loc[(snap["sim_time_T0"] - target_T0).abs().idxmin(), "sim_time_T0"]
    snap = snap[snap["sim_time_T0"] == closest_t].copy()
    snap = snap[snap["agg_id"] != -1]
    snap["mass"] = snap["n_particles"].astype(float)
    snap["r"]    = np.sqrt(snap["cx_AU"]**2 + snap["cz_AU"]**2)
    return snap

# Compute for each run: % in planets (r<=30) and % ejected (r>30) — disjoint!
data = []
for label, mj, path, tT in RUNS:
    print(f"Loading {label} ...", end=" ", flush=True)
    t0 = time.time()
    snap = extract_frame(path, tT)
    print(f"{time.time()-t0:.0f}s  ({len(snap)} agg)")
    M_in   = snap.loc[snap["r"] <= R_EJECT, "mass"].sum()
    M_out  = snap.loc[snap["r"]  > R_EJECT, "mass"].sum()
    M_total_in_aggs = M_in + M_out
    M_dust = M_DISK - M_total_in_aggs     # singletons + small dispersed dust
    data.append({
        "label": label, "mj": mj,
        "pct_in":   100.0 * M_in   / M_DISK,
        "pct_out":  100.0 * M_out  / M_DISK,
        "pct_dust": 100.0 * M_dust / M_DISK,
        "n_agg":    len(snap),
    })

# Quick sanity check
print("\nSanity check (each row should sum to 100%):")
for d in data:
    s = d["pct_in"] + d["pct_out"] + d["pct_dust"]
    print(f"  {d['label']:<10}  in={d['pct_in']:>5.1f}  out={d['pct_out']:>5.1f}  dust={d['pct_dust']:>5.1f}  sum={s:.2f}")

# ----------------------------------------------------------------------
# Russian fonts — let matplotlib find a Cyrillic font available on Windows
# ----------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"

# ----------------------------------------------------------------------
# GRAPH 1 — main result
# ----------------------------------------------------------------------
labels  = [d["label"] for d in data]
mass_vals = [d["mj"] for d in data]
pct_in   = np.array([d["pct_in"]   for d in data])
pct_out  = np.array([d["pct_out"]  for d in data])
pct_dust = np.array([d["pct_dust"] for d in data])

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(labels))
w = 0.38
b1 = ax.bar(x - w/2, pct_in,  w, color="#3066be", edgecolor="black",
            label="В планетах (r ≤ 30 а.е.)")
b2 = ax.bar(x + w/2, pct_out, w, color="#c92424", edgecolor="black",
            label="Выброшено (r > 30 а.е.)")
for bar, v in zip(b1, pct_in):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1.0, f"{v:.1f}%",
            ha="center", fontsize=10, fontweight="bold", color="#1a3e76")
for bar, v in zip(b2, pct_out):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1.0, f"{v:.1f}%",
            ha="center", fontsize=10, fontweight="bold", color="#7a1313")

# annotate dust
for xi, dv in zip(x, pct_dust):
    ax.text(xi, -3, f"(пыль: {dv:.0f}%)", ha="center", fontsize=8, color="gray")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_xlabel("Масса возмутителя", fontsize=12)
ax.set_ylabel("% от исходного диска (10 000 M_земли)", fontsize=12)
ax.set_title("Распределение массы диска: планеты внутри vs выброс наружу",
             fontsize=13, fontweight="bold")
ax.set_ylim(-6, 80)
ax.axhline(0, color="black", lw=0.5)
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper left", fontsize=11)

# Highlight zones
ax.axvspan(1.5, 2.5, alpha=0.10, color="red")    # catastrophe = 1 M_J
ax.axvspan(2.5, 3.5, alpha=0.10, color="green")  # stable = 3 M_J
ax.text(2.0, 75, "КАТАСТРОФА", ha="center", color="darkred", fontweight="bold", fontsize=10)
ax.text(3.0, 75, "СТАБИЛЬНО", ha="center", color="darkgreen", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "GRAPH1_mass_budget_correct.png", dpi=140)
plt.close()

# ----------------------------------------------------------------------
# GRAPH 2 — aggregate counts
# ----------------------------------------------------------------------
n_aggs = np.array([d["n_agg"] for d in data])

fig, ax = plt.subplots(figsize=(11, 7))
colors = ["#4b4b4b", "#6233a5", "#c5378a", "#e26f3d", "#f0a82a", "#f5d836"]
b = ax.bar(x, n_aggs, color=colors, edgecolor="black", linewidth=1.2)
for bar, v in zip(b, n_aggs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 30, f"{v}",
            ha="center", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_xlabel("Масса возмутителя", fontsize=12)
ax.set_ylabel("Количество агрегатов в диске", fontsize=12)
ax.set_title("Число тел в системе на t ≈ 800 лет (без одиночных частиц)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, max(n_aggs) * 1.18)
ax.grid(alpha=0.3, axis="y")

# Regime labels under bars
regimes = ["естеств.", "усил.", "катастр.", "защита", "хаос-1", "хаос-2"]
for xi, r in zip(x, regimes):
    ax.text(xi, -90, r, ha="center", fontsize=10, style="italic", color="gray")

plt.tight_layout()
plt.savefig(OUT / "GRAPH2_aggregate_counts.png", dpi=140)
plt.close()

print(f"\n✓ Two clean Russian charts:")
print(f"  GRAPH1_mass_budget_correct.png")
print(f"  GRAPH2_aggregate_counts.png")
