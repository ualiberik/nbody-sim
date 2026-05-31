# N-Body Simulation — Design Spec
**Date:** 2026-05-24  
**Project:** nbody-sim  
**Goal:** Research tool for studying giant planet influence on protoplanetary disk evolution

---

## 1. Overview

A 3D N-body simulation of a protoplanetary disk with Barnes-Hut gravity computed on GPU (CUDA). Physics runs in C++/CUDA, results written to binary frames file and CSV statistics file, visualized in Unity. First milestone: disk of 100,000 particles around a fixed central star, no giant planet yet.

---

## 2. Architecture

```
nbody-sim/
├── CMakeLists.txt
├── config/
│   └── simulation.json
├── src/
│   ├── main.cpp
│   ├── disk/
│   │   └── DiskInit.cpp/h
│   ├── physics/
│   │   ├── BHTree.cu/h
│   │   ├── Integrator.cu/h
│   │   └── Collision.cpp/h
│   ├── analysis/
│   │   └── Aggregate.cpp/h
│   └── io/
│       └── OutputWriter.cpp/h
└── data/
```

**Data flow:**
`Config → DiskInit → [Leapfrog → BHTree(CUDA) → Collision → Aggregate] × N → OutputWriter`

---

## 3. Physics

### Units (Natural)
| Quantity  | Unit    | SI equivalent       |
|-----------|---------|---------------------|
| Mass      | M☉      | 1.989 × 10³⁰ kg     |
| Distance  | AU      | 1.496 × 10¹¹ m      |
| Time      | T₀      | ≈ 58.1 days         |
| Velocity  | AU/T₀   | ≈ 29.8 km/s         |
| G         | 1       | —                   |

1 orbital period at 1 AU = 2π T₀ ≈ 1 year ✓

### Particle Mass
- m_particle = 0.1 M_Earth = 3.003 × 10⁻⁷ M☉
- 10 particles = 1 M_Earth
- 100 particles ≈ 1 M_Jupiter
- Total disk mass (100k particles) = 10,000 M_Earth ≈ 31.5 M_Jupiter ≈ 0.03 M☉

### Disk Initialization
- Inner radius: 0.5 AU
- Outer radius: 5.0 AU
- Surface density: Σ(r) ∝ r^(-3/2) (MMSN profile)
- Vertical profile: Gaussian, scale height h(r) = 0.05·r
- Initial velocity: circular v = √(GM☉/r) + 1% random noise

### Gravity
- Barnes-Hut algorithm, θ = 0.5
- Softening length: ε = 0.01 AU
- Force: F = G·m₁·m₂ / (r² + ε²)

### Integration
- Leapfrog (Störmer-Verlet) — symplectic, excellent energy conservation for orbits

### Collisions
- Detection: distance < r_coll = 100 × physical_radius (enhanced cross-section)
- Response: inelastic bounce, restitution coefficient e = 0.3
- No merging — particles remain separate, velocities corrected
- Detected during Barnes-Hut tree traversal (no separate O(N²) pass)

### Aggregates
- Friend-of-Friends (FOF) grouping, link length = r_coll
- Virtual labels only — no effect on physics
- Computed on CPU after each output step

---

## 4. CUDA Implementation (Burtscher & Pingali)

Each simulation step:
1. Build octree — CUDA kernels with atomic operations
2. Compute center-of-mass — CUDA kernel, bottom-up
3. Compute forces — CUDA kernel, 1 thread per particle
4. Leapfrog step — CUDA kernel
5. Collision detection & response — CUDA kernel (tree cell broad phase)

### Main Loop (main.cpp)
```
initialize disk on CPU
copy to GPU

for each step:
    BHTree::build()
    BHTree::computeForces()
    Integrator::leapfrogStep()
    Collision::detectAndRespond()

    if step % output_every == 0:
        copy positions GPU → CPU
        OutputWriter::writeFrame()
        Aggregate::detectFOF()
        OutputWriter::writeStats()
```

### Performance Estimate (RTX 30/40 series)
| Operation           | Time per step |
|---------------------|---------------|
| BH tree build       | ~2–5 ms       |
| BH forces (100k)    | ~8–15 ms      |
| Leapfrog            | ~0.5 ms       |
| Collisions          | ~2–4 ms       |
| **Total per step**  | **~15–25 ms** |
| 10,000 steps        | **~3–5 min**  |

15 runs ≈ 1–1.5 hours total.

---

## 5. Data Structures

### ParticleData (SoA — GPU-friendly)
```cpp
struct ParticleData {
    float* x,  *y,  *z;   // position (AU)
    float* vx, *vy, *vz;  // velocity (AU/T₀)
    float* ax, *ay, *az;  // acceleration
    float* mass;           // (M☉)
    int*   id;
    int    n;
};
```

---

## 6. Output Files

### frames.bin (visualization, for Unity)
```
[Header]
  char[4]  magic       = "NBOD"
  uint32   version     = 1
  uint32   n_particles
  uint32   n_frames
  float64  dt          (T₀)
  float64  output_dt   (T₀)

[Frame × n_frames]
  float64  sim_time    (T₀)
  float32  x[n]
  float32  y[n]
  float32  z[n]
```
Estimated size: 100k × 3 × 4 bytes × 1000 frames ≈ **1.2 GB**

### stats.csv (aggregates, for analysis)
```
frame,sim_time_T0,sim_time_years,n_aggregates,agg_id,n_particles,
mass_Msun,mass_Mearth,cx_AU,cy_AU,cz_AU,vx,vy,vz,dist_center_AU
```
One row per aggregate per frame. All aggregates recorded every output step.

---

## 7. Configuration (simulation.json)
```json
{
  "n_particles": 100000,
  "disk_r_min": 0.5,
  "disk_r_max": 5.0,
  "disk_h_factor": 0.05,
  "particle_mass_msun": 3.003e-7,
  "star_mass_msun": 1.0,
  "softening_AU": 0.01,
  "theta": 0.5,
  "collision_radius_factor": 100,
  "restitution": 0.3,
  "dt": 0.01,
  "total_time": 2000.0,
  "output_every_n_steps": 200
}
```

---

## 8. Future Extensions (not in scope for milestone 1)
- Giant planet on fixed circular orbit (configurable radius, mass, initial phase)
- Three research hypotheses TBD
- Unity visualization project
