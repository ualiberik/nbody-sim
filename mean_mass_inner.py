"""Mean mass of bodies inside r < 30 AU for each run, by mass class."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

RUNS = [
    ("control 800y",  0.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv", 5026.5),
    ("control 1592y", 0.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260525_202643\stats.csv", 10000.0),
    ("0.3 Mю",        0.3,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_000833\stats.csv", 5000.0),
    ("1.0 Mю",        1.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_015429\stats.csv", 5000.0),
    ("3.0 Mю",        3.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_034408\stats.csv", 5000.0),
    ("5.0 Mю",        5.0,  r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_053325\stats.csv", 5000.0),
    ("10  Mю",        10.0, r"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260526_073428\stats.csv", 5000.0),
]
R_MAX = 30.0

def snap(path, t_target, tol=0.5):
    dtypes = {"sim_time_T0": np.float32, "agg_id": np.int32, "n_particles": np.int32,
              "cx_AU": np.float32, "cy_AU": np.float32, "cz_AU": np.float32}
    cols = list(dtypes.keys())
    rows = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=2_000_000, usecols=cols):
        m = (chunk["sim_time_T0"] >= t_target - tol) & (chunk["sim_time_T0"] <= t_target + tol)
        if m.any(): rows.append(chunk[m])
        if chunk["sim_time_T0"].iloc[-1] > t_target + tol and rows: break
    s = pd.concat(rows, ignore_index=True)
    ct = s.loc[(s["sim_time_T0"] - t_target).abs().idxmin(), "sim_time_T0"]
    s = s[s["sim_time_T0"] == ct].copy()
    s = s[s["agg_id"] != -1]
    s["mass"] = s["n_particles"].astype(float)
    s["r"]    = np.sqrt(s["cx_AU"]**2 + s["cz_AU"]**2)
    return s

print(f"\n{'Run':<16}{'N_in':>7}{'M_total':>10}{'M_mean':>9}{'M_median':>10}"
      f"{'N≥5':>6}{'mean≥5':>9}{'N≥30':>6}{'mean≥30':>10}")
print("-" * 90)

for label, mj, path, tT in RUNS:
    print(f"  {label:<16} loading...", end="\r", flush=True)
    s = snap(path, tT)
    inner = s[s["r"] < R_MAX]
    n_in = len(inner)
    m_total = float(inner["mass"].sum())
    m_mean  = float(inner["mass"].mean()) if n_in else 0
    m_med   = float(inner["mass"].median()) if n_in else 0
    big = inner[inner["mass"] >= 5]
    mean5 = float(big["mass"].mean()) if len(big) else 0
    giant = inner[inner["mass"] >= 30]
    mean30 = float(giant["mass"].mean()) if len(giant) else 0
    print(f"{label:<16}{n_in:>7}{m_total:>10.0f}{m_mean:>9.2f}"
          f"{m_med:>10.1f}{len(big):>6}{mean5:>9.2f}{len(giant):>6}{mean30:>10.1f}")
