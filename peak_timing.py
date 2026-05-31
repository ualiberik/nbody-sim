"""Timing statistics for max_mass evolution in every run."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

RUNS = [
    ("control",  0.0,  r"C:\Projects\nbody-sim\analysis_control\per_frame_summary.csv"),
    ("0.3 Mю",   0.3,  r"C:\Projects\nbody-sim\analysis_exp1_03Mjup\per_frame_summary.csv"),
    ("1.0 Mю",   1.0,  r"C:\Projects\nbody-sim\analysis_exp2_10Mjup\per_frame_summary.csv"),
    ("3.0 Mю",   3.0,  r"C:\Projects\nbody-sim\analysis_exp3_30Mjup\per_frame_summary.csv"),
    ("5.0 Mю",   5.0,  r"C:\Projects\nbody-sim\analysis_exp4_50Mjup\per_frame_summary.csv"),
    ("10  Mю",   10.0, r"C:\Projects\nbody-sim\analysis_exp5_100Mjup\per_frame_summary.csv"),
]

THRESHOLDS = [30, 95, 300, 500, 1000]   # Saturn, Jupiter, sJup, mega, hyper

def first_reach(pf, m):
    over = pf[pf["max_mass"] >= m]
    if len(over) == 0:
        return None, None
    row = over.iloc[0]
    return float(row["yrs"]), float(row["max_mass"])

print(f"\n{'Run':<10}{'Mj':>6}{'M_peak':>9}{'t_peak':>9}{'t≥30':>9}{'t≥95':>9}{'t≥300':>9}{'t≥500':>9}{'t≥1000':>10}{'t_final':>10}{'τ_peak':>9}{'rate':>10}")
print("=" * 110)
print(f"{'':<10}{'M⊕':>6}{'M⊕':>9}{'years':>9}{'yr':>9}{'yr':>9}{'yr':>9}{'yr':>9}{'yr':>10}{'yr':>10}{'M⊕/yr':>9}")
print("-" * 110)

for label, mj, path in RUNS:
    pf = pd.read_csv(path)
    final_yrs = float(pf["yrs"].iloc[-1])
    peak_idx = pf["max_mass"].idxmax()
    M_peak = float(pf["max_mass"].iloc[peak_idx])
    t_peak = float(pf["yrs"].iloc[peak_idx])

    t30,   _ = first_reach(pf, 30)
    t95,   _ = first_reach(pf, 95)
    t300,  _ = first_reach(pf, 300)
    t500,  _ = first_reach(pf, 500)
    t1000, _ = first_reach(pf, 1000)

    # Average growth rate from first reaching 10 M⊕ to peak
    t10, _ = first_reach(pf, 10)
    rate = (M_peak - 10) / (t_peak - t10) if (t10 and t_peak > t10) else float('nan')

    # τ_peak — duration spent within 90% of M_peak (lifetime of peak object)
    mask = pf["max_mass"] >= 0.9 * M_peak
    if mask.sum() > 1:
        rows = pf[mask]
        tau_peak = float(rows["yrs"].max() - rows["yrs"].min())
    else:
        tau_peak = 0.0

    print(f"{label:<10}{mj:>6.1f}{M_peak:>9.0f}{t_peak:>9.0f}"
          f"{(f'{t30:.0f}' if t30 else '--'):>9}"
          f"{(f'{t95:.0f}' if t95 else '--'):>9}"
          f"{(f'{t300:.0f}' if t300 else '--'):>9}"
          f"{(f'{t500:.0f}' if t500 else '--'):>9}"
          f"{(f'{t1000:.0f}' if t1000 else '--'):>10}"
          f"{final_yrs:>10.0f}"
          f"{tau_peak:>9.0f}"
          f"{rate:>10.2f}")

print("\nКолонки:")
print("  M_peak    — максимальная масса самого тяжёлого тела за всю симуляцию")
print("  t_peak    — год, когда был достигнут этот пик")
print("  t≥30/95/300/500/1000 — год первого пересечения порога масс")
print("  t_final   — длительность симуляции")
print("  τ_peak    — длительность фазы 'почти-пик' (≥90% от M_peak)")
print("  rate      — средняя скорость роста от 10 M⊕ до пика")
