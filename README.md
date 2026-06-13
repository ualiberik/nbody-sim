# N-Body Protoplanetary Disk Simulation

<img width="1286" height="682" alt="unity-visualizer - Visualizer - Windows, Mac, Linux - Unity 6 (6000 0 24f1)_ _DX11_ 5_26_2026 7_33_41 PM" src="https://github.com/user-attachments/assets/a7672bb7-ce9a-4e30-ae06-06c45da28bd0" />

A GPU-accelerated N-body simulation of planet formation in a protoplanetary
disk, with a real-time Unity visualizer and a Python analysis pipeline.

The engine integrates **100,000+ self-gravitating particles** using a
Barnes-Hut octree on CUDA, models inelastic collisions and gas drag, and
tracks the emergence of planet-mass aggregates over time. A companion study
investigates how a **fixed outer gas-giant perturber** reshapes the inner
planetary system.

> **About this project.** A short (~3-day) computational exploration of protoplanetary-disk dynamics. The CUDA engine and analysis pipeline were built with heavy AI assistance (Claude); my part was the experiment design (the perturber-mass sweep) and interpreting the resulting dynamical regimes.
> This is **not** my original N-body work. I earlier wrote a full N-body gravity simulator from scratch in Unity / C# (Roche limit, star-system formation); that source wasn't preserved, but it's documented here: [YouTube ▶](https://youtu.be/286jV8SJmT8).

---

## Highlights

- **Barnes-Hut O(N log N) gravity** on the GPU (CUDA), opening-angle `θ` criterion
- **Leapfrog KDK integrator** in natural units (G = 1, M☉ = 1, AU, T₀ = 1/2π yr)
- **Inelastic collisions** via a two-pass impulse kernel (tree-assisted neighbour search)
- **Gas drag** that damps eccentricity / inclination toward circular Keplerian orbits
- **Friends-of-Friends** aggregate detection → per-frame planet catalogue
- **Optional analytic perturber**: a fixed outer body on a Keplerian orbit that
  exerts gravity but is itself never integrated, never collides, and is excluded
  from aggregate detection
- **Binary frame format** streamed by a **Unity 6** visualizer (spheres, star
  corona, perturber, run-select menu)
- **Python pipeline** producing mass-budget charts, radial heatmaps, migration
  tracks, and cross-experiment comparisons

---

## Repository layout

```
.
├─ src/                 # C++/CUDA simulation engine
│  ├─ config/           #   JSON config loader
│  ├─ disk/             #   MMSN disk initialisation
│  ├─ physics/          #   BH octree, leapfrog integrator, perturber kernel
│  ├─ analysis/         #   Friends-of-Friends aggregate detection
│  └─ io/               #   frames.bin + stats.csv writer
├─ include/             # Public headers
├─ tests/               # Catch2 unit tests (CPU + CUDA)
├─ config/              # Ready-to-run JSON configs (control + 5 perturber masses)
├─ docs/                # design.md, plan.md
├─ unity-visualizer/    # Unity 6 project (shaders, renderers, run-select menu)
├─ analyze_*.py         # Per-run and cross-run analysis scripts
├─ plot_*.py            # Figure generators (heatmaps, schemas, budgets)
└─ overnight_log.txt    # Full experiment sweep log + findings
```

---

## Building the engine

### Requirements
- CUDA Toolkit **12.0+** (tested with 13.2)
- CMake **3.20+**
- A C++17 host compiler (MSVC 2022 on Windows, GCC/Clang on Linux)
- An NVIDIA GPU of compute capability **8.6 or 8.9** (Ampere / Ada).
  Edit `CUDA_ARCHITECTURES` in `CMakeLists.txt` for other GPUs.

Dependencies (`nlohmann/json`, `Catch2`) are fetched automatically by CMake.

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Outputs:
- `build/nbody-sim`   — the simulation executable
- `build/nbody-tests` — the unit-test runner

Run the tests:

```bash
./build/nbody-tests
```

---

## Running a simulation

```bash
./build/nbody-sim config/planet_formation.json
```

Each run writes to a timestamped directory `data/<YYYYMMDD_HHMMSS>/`:
- `frames.bin` — binary particle positions per frame (for the visualizer)
- `stats.csv`  — per-aggregate statistics per frame (for analysis)

### Configuration

All parameters live in a JSON file. Key fields:

| Field | Meaning |
|-------|---------|
| `n_particles` | Number of disk particles |
| `disk_r_min` / `disk_r_max` | Disk inner / outer radius (AU) |
| `particle_mass_msun` | Mass per particle (solar masses) |
| `softening_AU` | Gravitational softening length |
| `theta` | Barnes-Hut opening angle (0.5 typical) |
| `collision_radius_factor` | Collision radius = factor × physical radius |
| `restitution` | Coefficient of restitution (0 = perfectly inelastic) |
| `gas_drag_rate` | Eccentricity/inclination damping rate (1/T₀); 0 disables |
| `dt` / `total_time` | Timestep / total duration (T₀) |
| `output_every_n_steps` | Frame cadence |
| `perturber_enabled` | Add a fixed outer perturber |
| `perturber_mass_mjup` | Perturber mass (Jupiter masses) |
| `perturber_radius_AU` | Perturber orbital radius |

Provided configs:
- `config/planet_formation.json` — baseline (no perturber)
- `config/perturber_03Mjup.json` … `perturber_100Mjup.json` — 0.3 → 10 M_Jup sweep

### Units

Natural units with `G = 1`, `M☉ = 1`, length in AU. One time unit
**T₀ = 1/(2π) yr ≈ 58.1 days**, so one orbit at 1 AU takes `2π` T₀ = 1 year.

---

## Output format (`frames.bin`)

```
Header (32 bytes):
  magic       char[4]   "NBOD"
  version     uint32    2
  n_particles uint32
  n_frames    uint32
  dt          double    (T₀)
  output_dt   double    (T₀ between frames)

Per frame (version 2):
  time_T0     double
  x[n]        float32   positions (AU)   — disk lies in the XZ plane
  y[n]        float32
  z[n]        float32
  agg_size[n] uint16    aggregate size containing each particle (1 = singleton)
```

The perturber (when enabled) is recorded in `stats.csv` as a sentinel row with
`agg_id = -1`.

---

## Visualizer (Unity 6)

Open `unity-visualizer/` in **Unity 6000.0.24f1** and press Play.

- Point the player at a `frames.bin` (Inspector field or the in-scene
  **run-select menu**)
- A `perturber.json` sidecar next to `frames.bin` drives the green perturber
  sphere; absent or `enabled:false` → no perturber drawn (control runs)
- Particle brightness scales with aggregate size, so planets stand out from dust
- Camera: orbit / zoom around the central star

---

## Analysis pipeline

The Python scripts read `stats.csv` (streamed in chunks for multi-GB files):

| Script | Produces |
|--------|----------|
| `analyze_experiment.py` | Per-run growth curves, top-down map, mass-radius plot |
| `full_comparison.py` | Unified metrics table across all runs |
| `plot_mass_budget.py` / `plot_final_two.py` | Disk-mass budget (planets vs ejected) |
| `plot_heatmap.py` | Radial mass-density heatmaps + polar maps |
| `plot_overlay_schemas.py` | Colour-coded population overlays |
| `peak_timing.py` | Timing statistics of the most massive body |

Requires `numpy`, `pandas`, `matplotlib`, `scipy`.

---

## The perturber study

Starting from an identical 10,000-particle disk (0.5–5 AU), six runs were
compared: a control with no perturber, and five with a fixed gas giant at
**20 AU** of mass **0.3, 1, 3, 5, 10 M_Jup**.

**Main result — the response is strongly non-monotonic.** The system settles
into one of three discrete regimes with sharp transitions, not a smooth trend:

| Perturber | Gas giants | Giant zone | Disk ejected (>30 AU) | Regime |
|-----------|-----------|------------|------------------------|--------|
| none      | 8–9 | 3–7 AU | ~3% | natural growth |
| 0.3 M_Jup | 33 | 10–17 AU | ~3% | **shepherding** (inner resonances) |
| 1.0 M_Jup | 48 | 49–57 AU | **67%** | **catastrophe** (outer 4:1 resonance) |
| 3.0 M_Jup | 15 | 4–9 AU | ~6% | **protective** (inner disk preserved) |
| 5.0 M_Jup | 25 | 25–70 AU | 58% | chaotic scattering |
| 10  M_Jup | 30 | 38–58 AU | 56% | chaotic ring |

Key findings:
- A **light** companion (≤ 0.3 M_Jup) *enhances* planet formation, capturing the
  largest fraction of disk mass into giants and protecting them from mutual
  destruction.
- Around **1 M_Jup** the disk undergoes a catastrophic rearrangement: a transient
  ~5-Jupiter-mass body forms and is then destroyed, and ⅔ of the disk is flung
  into an outer resonant ring.
- A **narrow stability window near 3 M_Jup** leaves the inner disk almost
  untouched — the heavy perturber absorbs would-be scattering bodies instead of
  ejecting them.

See `overnight_log.txt` and `analysis_final_comparison/` for full figures.

> Note: these are scaled, qualitative experiments (massive disk, short
> integration times) intended to explore dynamical regimes, not quantitative
> predictions of real planetary systems.

---

## License

Released under the MIT License. See `LICENSE` if present, otherwise treat as MIT.
