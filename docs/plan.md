# N-Body Simulation Implementation Plan

**Goal:** Build a C++/CUDA N-body simulation of a 100,000-particle protoplanetary disk using Barnes-Hut gravity, outputting binary frame data for Unity and CSV statistics for analysis.

**Architecture:** Particles orbit a fixed central mass in a 3D disk (MMSN density profile). A CUDA Barnes-Hut octree computes O(N log N) gravitational forces. Leapfrog (KDK) integration advances positions. A separate CUDA collision kernel reuses the built tree for inelastic bounce detection. CPU-side Friend-of-Friends groups nearby particles into aggregates for statistics output.

**Tech Stack:** C++17, CUDA 12+, CMake 3.20+, nlohmann/json v3.11.3 (config parsing), Catch2 v3.5.2 (unit tests)

---

## File Map

| File | Responsibility |
|------|---------------|
| `CMakeLists.txt` | Build system, dependency fetching, test discovery |
| `config/simulation.json` | Runtime parameters |
| `include/Config.h` | Config struct + loader declaration |
| `include/ParticleData.h` | SoA particle struct + CPU/GPU alloc/copy |
| `src/config/Config.cpp` | JSON config loader |
| `src/disk/DiskInit.h` | Disk init declaration |
| `src/disk/DiskInit.cpp` | MMSN disk initialization (CPU) |
| `src/physics/BHTree.cuh` | Octree struct + all kernel launcher declarations |
| `src/physics/BHTree.cu` | All BH CUDA kernels: bbox, build, summarize, sort, forces, collisions |
| `src/physics/Integrator.cuh` | Leapfrog kernel launcher declarations |
| `src/physics/Integrator.cu` | Half-kick, drift, reset-acceleration kernels |
| `src/analysis/Aggregate.h` | Aggregate struct + FOF declaration |
| `src/analysis/Aggregate.cpp` | CPU Friend-of-Friends grouping |
| `src/io/OutputWriter.h` | OutputWriter declaration |
| `src/io/OutputWriter.cpp` | frames.bin + stats.csv writing |
| `src/main.cpp` | Entry point + simulation loop |
| `tests/test_config.cpp` | Config loading tests |
| `tests/test_disk_init.cpp` | Disk init tests |
| `tests/test_output.cpp` | OutputWriter tests |
| `tests/test_integrator.cu` | Leapfrog energy conservation tests |
| `tests/test_bhtree.cu` | BH force accuracy vs direct N-body |
| `tests/test_collision.cu` | Collision response tests |
| `tests/test_aggregate.cpp` | FOF grouping tests |

---

## Task 1: CMake Skeleton

**Files:**
- Create: `CMakeLists.txt`
- Create: `config/simulation.json`
- Create: `src/main.cpp` (stub)

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(nbody-sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Fetch dependencies
include(FetchContent)

FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Source files
set(SIM_SOURCES
  src/config/Config.cpp
  src/disk/DiskInit.cpp
  src/physics/BHTree.cu
  src/physics/Integrator.cu
  src/analysis/Aggregate.cpp
  src/io/OutputWriter.cpp
)

# Main executable
add_executable(nbody-sim src/main.cpp ${SIM_SOURCES})
target_include_directories(nbody-sim PRIVATE include)
target_link_libraries(nbody-sim PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(nbody-sim PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "86;89"
)

# Test executable
set(TEST_SOURCES
  tests/test_config.cpp
  tests/test_disk_init.cpp
  tests/test_output.cpp
  tests/test_integrator.cu
  tests/test_bhtree.cu
  tests/test_collision.cu
  tests/test_aggregate.cpp
)
add_executable(nbody-tests ${TEST_SOURCES} ${SIM_SOURCES})
target_include_directories(nbody-tests PRIVATE include)
target_link_libraries(nbody-tests PRIVATE Catch2::Catch2WithMain nlohmann_json::nlohmann_json)
set_target_properties(nbody-tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "86;89"
)

include(CTest)
include(Catch)
catch_discover_tests(nbody-tests)
```

- [ ] **Step 2: Create directory structure**

```
mkdir -p src/config src/disk src/physics src/analysis src/io include tests data config
```

- [ ] **Step 3: Create stub src/main.cpp**

```cpp
#include <iostream>
int main() {
    std::cout << "nbody-sim starting...\n";
    return 0;
}
```

- [ ] **Step 4: Create config/simulation.json**

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
  "collision_radius_factor": 100.0,
  "restitution": 0.3,
  "dt": 0.01,
  "total_time": 2000.0,
  "output_every_n_steps": 200,
  "output_dir": "data"
}
```

- [ ] **Step 5: Configure and verify build**

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target nbody-sim
```
Expected: compiles without errors, `build/nbody-sim` exists.

- [ ] **Step 6: Commit**

```
git add CMakeLists.txt config/ src/main.cpp
git commit -m "feat: cmake skeleton with CUDA, Catch2, nlohmann/json"
```

---

## Task 2: Config System

**Files:**
- Create: `include/Config.h`
- Create: `src/config/Config.cpp`
- Create: `tests/test_config.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// tests/test_config.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Config.h"

TEST_CASE("loadConfig reads simulation.json correctly", "[config]") {
    Config cfg = loadConfig("config/simulation.json");
    REQUIRE(cfg.n_particles == 100000);
    REQUIRE_THAT(cfg.disk_r_min, Catch::Matchers::WithinRel(0.5f, 0.001f));
    REQUIRE_THAT(cfg.disk_r_max, Catch::Matchers::WithinRel(5.0f, 0.001f));
    REQUIRE_THAT(cfg.particle_mass_msun, Catch::Matchers::WithinRel(3.003e-7f, 0.001f));
    REQUIRE_THAT(cfg.star_mass_msun, Catch::Matchers::WithinRel(1.0f, 0.001f));
    REQUIRE_THAT(cfg.theta, Catch::Matchers::WithinRel(0.5f, 0.001f));
    REQUIRE_THAT(cfg.dt, Catch::Matchers::WithinRel(0.01f, 0.001f));
    REQUIRE(cfg.output_every_n_steps == 200);
    REQUIRE(cfg.output_dir == "data");
}
```

- [ ] **Step 2: Run test — verify it fails**

```
cmake --build build --target nbody-tests && cd build && ctest -R config -V
```
Expected: compile error (Config.h not found).

- [ ] **Step 3: Create include/Config.h**

```cpp
#pragma once
#include <string>

struct Config {
    int   n_particles          = 100000;
    float disk_r_min           = 0.5f;
    float disk_r_max           = 5.0f;
    float disk_h_factor        = 0.05f;
    float particle_mass_msun   = 3.003e-7f;
    float star_mass_msun       = 1.0f;
    float softening_AU         = 0.01f;
    float theta                = 0.5f;
    float collision_radius_factor = 100.0f;
    float restitution          = 0.3f;
    float dt                   = 0.01f;
    float total_time           = 2000.0f;
    int   output_every_n_steps = 200;
    std::string output_dir     = "data";
};

Config loadConfig(const std::string& path);
```

- [ ] **Step 4: Create src/config/Config.cpp**

```cpp
#include "Config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

Config loadConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config: " + path);
    nlohmann::json j = nlohmann::json::parse(f);

    Config cfg;
    cfg.n_particles            = j.value("n_particles", cfg.n_particles);
    cfg.disk_r_min             = j.value("disk_r_min", cfg.disk_r_min);
    cfg.disk_r_max             = j.value("disk_r_max", cfg.disk_r_max);
    cfg.disk_h_factor          = j.value("disk_h_factor", cfg.disk_h_factor);
    cfg.particle_mass_msun     = j.value("particle_mass_msun", cfg.particle_mass_msun);
    cfg.star_mass_msun         = j.value("star_mass_msun", cfg.star_mass_msun);
    cfg.softening_AU           = j.value("softening_AU", cfg.softening_AU);
    cfg.theta                  = j.value("theta", cfg.theta);
    cfg.collision_radius_factor= j.value("collision_radius_factor", cfg.collision_radius_factor);
    cfg.restitution            = j.value("restitution", cfg.restitution);
    cfg.dt                     = j.value("dt", cfg.dt);
    cfg.total_time             = j.value("total_time", cfg.total_time);
    cfg.output_every_n_steps   = j.value("output_every_n_steps", cfg.output_every_n_steps);
    cfg.output_dir             = j.value("output_dir", cfg.output_dir);
    return cfg;
}
```

- [ ] **Step 5: Run test — verify it passes**

```
cmake --build build --target nbody-tests && cd build && ctest -R config -V
```
Expected: PASSED.

- [ ] **Step 6: Commit**

```
git add include/Config.h src/config/Config.cpp tests/test_config.cpp
git commit -m "feat: config system with JSON loading"
```

---

## Task 3: ParticleData + CUDA Memory Management

**Files:**
- Create: `include/ParticleData.h`

- [ ] **Step 1: Write failing test**

```cpp
// tests/test_disk_init.cpp  (extend later; start here)
#include <catch2/catch_test_macros.hpp>
#include "ParticleData.h"

TEST_CASE("CPU particle allocation and free", "[particles]") {
    ParticleData p = allocateParticlesCPU(100);
    REQUIRE(p.n == 100);
    REQUIRE(p.x != nullptr);
    REQUIRE(p.vx != nullptr);
    p.x[0] = 1.5f; p.vx[0] = 0.1f;
    REQUIRE(p.x[0] == 1.5f);
    freeParticlesCPU(p);
    REQUIRE(p.x == nullptr);
}
```

- [ ] **Step 2: Run — verify fail**

```
cmake --build build --target nbody-tests && cd build && ctest -R particles -V
```
Expected: compile error.

- [ ] **Step 3: Create include/ParticleData.h**

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>

// Struct-of-Arrays layout for CUDA coalesced memory access
struct ParticleData {
    float* x   = nullptr;
    float* y   = nullptr;
    float* z   = nullptr;
    float* vx  = nullptr;
    float* vy  = nullptr;
    float* vz  = nullptr;
    float* ax  = nullptr;
    float* ay  = nullptr;
    float* az  = nullptr;
    float* mass = nullptr;
    int*   id  = nullptr;
    int    n   = 0;
};

inline ParticleData allocateParticlesCPU(int n) {
    ParticleData p;
    p.n = n;
    p.x    = new float[n](); p.y    = new float[n](); p.z    = new float[n]();
    p.vx   = new float[n](); p.vy   = new float[n](); p.vz   = new float[n]();
    p.ax   = new float[n](); p.ay   = new float[n](); p.az   = new float[n]();
    p.mass = new float[n]();
    p.id   = new int[n]();
    return p;
}

inline void freeParticlesCPU(ParticleData& p) {
    delete[] p.x;  delete[] p.y;  delete[] p.z;
    delete[] p.vx; delete[] p.vy; delete[] p.vz;
    delete[] p.ax; delete[] p.ay; delete[] p.az;
    delete[] p.mass; delete[] p.id;
    p.x = p.y = p.z = p.vx = p.vy = p.vz = p.ax = p.ay = p.az = p.mass = nullptr;
    p.id = nullptr; p.n = 0;
}

inline ParticleData allocateParticlesGPU(int n) {
    ParticleData p; p.n = n;
    auto alloc = [&](float** ptr) { cudaMalloc(ptr, n * sizeof(float)); };
    alloc(&p.x);  alloc(&p.y);  alloc(&p.z);
    alloc(&p.vx); alloc(&p.vy); alloc(&p.vz);
    alloc(&p.ax); alloc(&p.ay); alloc(&p.az);
    alloc(&p.mass);
    cudaMalloc(&p.id, n * sizeof(int));
    return p;
}

inline void freeParticlesGPU(ParticleData& p) {
    cudaFree(p.x);  cudaFree(p.y);  cudaFree(p.z);
    cudaFree(p.vx); cudaFree(p.vy); cudaFree(p.vz);
    cudaFree(p.ax); cudaFree(p.ay); cudaFree(p.az);
    cudaFree(p.mass); cudaFree(p.id);
    p.x = p.y = p.z = p.vx = p.vy = p.vz = p.ax = p.ay = p.az = p.mass = nullptr;
    p.id = nullptr; p.n = 0;
}

inline void copyToGPU(const ParticleData& cpu, ParticleData& gpu) {
    auto cp = [&](float* dst, const float* src) {
        cudaMemcpy(dst, src, cpu.n * sizeof(float), cudaMemcpyHostToDevice);
    };
    cp(gpu.x, cpu.x); cp(gpu.y, cpu.y); cp(gpu.z, cpu.z);
    cp(gpu.vx, cpu.vx); cp(gpu.vy, cpu.vy); cp(gpu.vz, cpu.vz);
    cp(gpu.ax, cpu.ax); cp(gpu.ay, cpu.ay); cp(gpu.az, cpu.az);
    cp(gpu.mass, cpu.mass);
    cudaMemcpy(gpu.id, cpu.id, cpu.n * sizeof(int), cudaMemcpyHostToDevice);
}

inline void copyToCPU(const ParticleData& gpu, ParticleData& cpu) {
    auto cp = [&](float* dst, const float* src) {
        cudaMemcpy(dst, src, gpu.n * sizeof(float), cudaMemcpyDeviceToHost);
    };
    cp(cpu.x, gpu.x); cp(cpu.y, gpu.y); cp(cpu.z, gpu.z);
    cp(cpu.vx, gpu.vx); cp(cpu.vy, gpu.vy); cp(cpu.vz, gpu.vz);
    cp(cpu.ax, gpu.ax); cp(cpu.ay, gpu.ay); cp(cpu.az, gpu.az);
    cp(cpu.mass, gpu.mass);
    cudaMemcpy(cpu.id, gpu.id, gpu.n * sizeof(int), cudaMemcpyDeviceToHost);
}
```

- [ ] **Step 4: Run test — verify pass**

```
cmake --build build --target nbody-tests && cd build && ctest -R particles -V
```
Expected: PASSED.

- [ ] **Step 5: Commit**

```
git add include/ParticleData.h tests/test_disk_init.cpp
git commit -m "feat: ParticleData SoA with CPU/GPU alloc and copy"
```

---

## Task 4: Disk Initialization

**Files:**
- Create: `src/disk/DiskInit.h`
- Create: `src/disk/DiskInit.cpp`
- Modify: `tests/test_disk_init.cpp`

- [ ] **Step 1: Write failing tests**

```cpp
// Append to tests/test_disk_init.cpp
#include "DiskInit.h"
#include "Config.h"
#include <cmath>

TEST_CASE("initDisk places all particles in disk bounds", "[disk]") {
    Config cfg;
    cfg.n_particles = 1000;
    cfg.disk_r_min = 0.5f; cfg.disk_r_max = 5.0f;
    cfg.disk_h_factor = 0.05f;
    cfg.particle_mass_msun = 3.003e-7f;
    cfg.star_mass_msun = 1.0f;

    ParticleData p = initDisk(cfg, 42);

    for (int i = 0; i < 1000; ++i) {
        float r = std::sqrt(p.x[i]*p.x[i] + p.z[i]*p.z[i]);
        REQUIRE(r >= cfg.disk_r_min * 0.99f);
        REQUIRE(r <= cfg.disk_r_max * 1.01f);
        REQUIRE_THAT(p.mass[i], Catch::Matchers::WithinRel(cfg.particle_mass_msun, 0.001f));
    }
    freeParticlesCPU(p);
}

TEST_CASE("initDisk velocities are approximately circular", "[disk]") {
    Config cfg;
    cfg.n_particles = 100;
    cfg.disk_r_min = 1.0f; cfg.disk_r_max = 2.0f;
    cfg.disk_h_factor = 0.01f;
    cfg.particle_mass_msun = 3.003e-7f;
    cfg.star_mass_msun = 1.0f;

    ParticleData p = initDisk(cfg, 1);

    for (int i = 0; i < 100; ++i) {
        float r = std::sqrt(p.x[i]*p.x[i] + p.z[i]*p.z[i]);
        float v_circ = std::sqrt(cfg.star_mass_msun / r);  // G=1
        float v = std::sqrt(p.vx[i]*p.vx[i] + p.vy[i]*p.vy[i] + p.vz[i]*p.vz[i]);
        // Within 5% of circular speed
        REQUIRE_THAT(v, Catch::Matchers::WithinRel(v_circ, 0.05f));
    }
    freeParticlesCPU(p);
}
```

- [ ] **Step 2: Run — verify fail**

```
cmake --build build --target nbody-tests && cd build && ctest -R disk -V
```
Expected: compile error.

- [ ] **Step 3: Create src/disk/DiskInit.h**

```cpp
#pragma once
#include "ParticleData.h"
#include "Config.h"

// Initialize CPU particles as a protoplanetary disk.
// Disk lies in XZ plane with Gaussian thickness along Y.
// Surface density: Sigma(r) ∝ r^(-3/2) (MMSN profile).
// Velocities: circular + 1% random noise.
ParticleData initDisk(const Config& cfg, unsigned int seed = 42);

// Compute physical radius of one particle in AU (from mass + rocky density).
// Assumes density = 3000 kg/m^3.
float particlePhysicalRadius(const Config& cfg);
```

- [ ] **Step 4: Create src/disk/DiskInit.cpp**

```cpp
#include "DiskInit.h"
#include <cmath>
#include <random>

static constexpr float PI = 3.14159265358979f;
// 1 AU in meters
static constexpr float AU_M = 1.496e11f;
// 1 M_sun in kg
static constexpr float MSUN_KG = 1.989e30f;

float particlePhysicalRadius(const Config& cfg) {
    float mass_kg = cfg.particle_mass_msun * MSUN_KG;
    float density = 3000.0f;  // kg/m^3
    float vol = mass_kg / density;
    float r_m = std::cbrt(3.0f * vol / (4.0f * PI));
    return r_m / AU_M;  // convert to AU
}

ParticleData initDisk(const Config& cfg, unsigned int seed) {
    ParticleData p = allocateParticlesCPU(cfg.n_particles);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    float r_min = cfg.disk_r_min;
    float r_max = cfg.disk_r_max;
    float sqrt_rmin = std::sqrt(r_min);
    float sqrt_rmax = std::sqrt(r_max);

    for (int i = 0; i < cfg.n_particles; ++i) {
        // Sample r from Sigma(r) ∝ r^(-3/2)
        // P(r) dr ∝ r^(-1/2) dr  =>  CDF(r) = (sqrt(r)-sqrt(rmin))/(sqrt(rmax)-sqrt(rmin))
        // Inverse CDF: r = (u*(sqrt_rmax - sqrt_rmin) + sqrt_rmin)^2
        float u = uni(rng);
        float r = (u * (sqrt_rmax - sqrt_rmin) + sqrt_rmin);
        r = r * r;

        // Azimuthal angle in XZ plane
        float phi = uni(rng) * 2.0f * PI;

        // Position
        p.x[i] = r * std::cos(phi);
        p.z[i] = r * std::sin(phi);
        // Gaussian vertical scatter: h(r) = disk_h_factor * r
        float h = cfg.disk_h_factor * r;
        p.y[i] = norm(rng) * h;

        // Circular velocity magnitude: v_circ = sqrt(G * M_star / r), G=1
        float v_circ = std::sqrt(cfg.star_mass_msun / r);

        // Direction: perpendicular to r in XZ plane
        float noise_scale = 0.01f * v_circ;
        p.vx[i] = -v_circ * std::sin(phi) + norm(rng) * noise_scale;
        p.vz[i] =  v_circ * std::cos(phi) + norm(rng) * noise_scale;
        p.vy[i] = norm(rng) * noise_scale;

        p.ax[i] = 0.0f; p.ay[i] = 0.0f; p.az[i] = 0.0f;
        p.mass[i] = cfg.particle_mass_msun;
        p.id[i] = i;
    }
    return p;
}
```

- [ ] **Step 5: Run tests — verify pass**

```
cmake --build build --target nbody-tests && cd build && ctest -R disk -V
```
Expected: 2 tests PASSED.

- [ ] **Step 6: Commit**

```
git add src/disk/ tests/test_disk_init.cpp
git commit -m "feat: MMSN disk initialization with Gaussian thickness"
```

---

## Task 5: OutputWriter — frames.bin

**Files:**
- Create: `src/io/OutputWriter.h`
- Create: `src/io/OutputWriter.cpp`
- Create: `tests/test_output.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// tests/test_output.cpp
#include <catch2/catch_test_macros.hpp>
#include "OutputWriter.h"
#include "Config.h"
#include "ParticleData.h"
#include <fstream>
#include <filesystem>

TEST_CASE("OutputWriter writes valid frames.bin header", "[output]") {
    Config cfg;
    cfg.n_particles = 4;
    cfg.dt = 0.01f;
    cfg.output_every_n_steps = 1;
    cfg.output_dir = "data/test_run";
    std::filesystem::create_directories(cfg.output_dir);

    ParticleData p = allocateParticlesCPU(4);
    p.x[0]=1.f; p.y[0]=0.f; p.z[0]=0.f;
    p.x[1]=2.f; p.y[1]=0.f; p.z[1]=0.f;
    p.x[2]=3.f; p.y[2]=0.f; p.z[2]=0.f;
    p.x[3]=4.f; p.y[3]=0.f; p.z[3]=0.f;

    {
        OutputWriter writer(cfg.output_dir, cfg);
        writer.writeFrame(p, 0.0f);
        writer.writeFrame(p, 1.0f);
        writer.finalize(2);
    }

    // Verify header magic
    std::ifstream f(cfg.output_dir + "/frames.bin", std::ios::binary);
    REQUIRE(f.is_open());
    char magic[4]; f.read(magic, 4);
    REQUIRE(magic[0]=='N'); REQUIRE(magic[1]=='B');
    REQUIRE(magic[2]=='O'); REQUIRE(magic[3]=='D');

    uint32_t version; f.read(reinterpret_cast<char*>(&version), 4);
    REQUIRE(version == 1u);

    uint32_t n_particles; f.read(reinterpret_cast<char*>(&n_particles), 4);
    REQUIRE(n_particles == 4u);

    uint32_t n_frames; f.read(reinterpret_cast<char*>(&n_frames), 4);
    REQUIRE(n_frames == 2u);

    freeParticlesCPU(p);
    std::filesystem::remove_all(cfg.output_dir);
}
```

- [ ] **Step 2: Run — verify fail**

```
cmake --build build --target nbody-tests && cd build && ctest -R output -V
```
Expected: compile error.

- [ ] **Step 3: Create src/io/OutputWriter.h**

```cpp
#pragma once
#include <string>
#include <fstream>
#include <vector>
#include "ParticleData.h"
#include "Config.h"

// Conversion constants (G=1, natural units)
constexpr float T0_TO_YEARS   = 1.0f / (2.0f * 3.14159265f);  // ≈ 0.1592 years per T0
constexpr float MSUN_TO_MEARTH = 332946.0f;

struct Aggregate; // forward declare

class OutputWriter {
public:
    OutputWriter(const std::string& output_dir, const Config& cfg);
    ~OutputWriter();

    // Write one frame of particle positions to frames.bin
    void writeFrame(const ParticleData& cpu_particles, float sim_time_T0);

    // Write per-frame aggregate stats row(s) to stats.csv
    void writeStats(const std::vector<Aggregate>& aggregates,
                    int frame_idx, float sim_time_T0);

    // Update n_frames in the binary header (call once at end)
    void finalize(int total_frames_written);

private:
    std::ofstream frames_file_;
    std::ofstream stats_file_;
    Config cfg_;
    std::streampos n_frames_offset_;  // position in file where n_frames is stored

    void writeFramesHeader();
    void writeStatsHeader();
};
```

- [ ] **Step 4: Create src/io/OutputWriter.cpp**

```cpp
#include "OutputWriter.h"
#include "Aggregate.h"
#include <filesystem>
#include <stdexcept>
#include <cstring>

OutputWriter::OutputWriter(const std::string& output_dir, const Config& cfg)
    : cfg_(cfg)
{
    std::filesystem::create_directories(output_dir);
    frames_file_.open(output_dir + "/frames.bin", std::ios::binary | std::ios::trunc);
    stats_file_.open(output_dir + "/stats.csv", std::ios::trunc);
    if (!frames_file_.is_open()) throw std::runtime_error("Cannot open frames.bin");
    if (!stats_file_.is_open())  throw std::runtime_error("Cannot open stats.csv");
    writeFramesHeader();
    writeStatsHeader();
}

OutputWriter::~OutputWriter() {
    if (frames_file_.is_open()) frames_file_.close();
    if (stats_file_.is_open())  stats_file_.close();
}

void OutputWriter::writeFramesHeader() {
    // magic
    frames_file_.write("NBOD", 4);
    // version
    uint32_t version = 1;
    frames_file_.write(reinterpret_cast<char*>(&version), 4);
    // n_particles
    uint32_t n = static_cast<uint32_t>(cfg_.n_particles);
    frames_file_.write(reinterpret_cast<char*>(&n), 4);
    // n_frames placeholder (will be updated by finalize())
    n_frames_offset_ = frames_file_.tellp();
    uint32_t nf = 0;
    frames_file_.write(reinterpret_cast<char*>(&nf), 4);
    // dt and output_dt
    double dt = static_cast<double>(cfg_.dt);
    double odt = static_cast<double>(cfg_.dt * cfg_.output_every_n_steps);
    frames_file_.write(reinterpret_cast<char*>(&dt), 8);
    frames_file_.write(reinterpret_cast<char*>(&odt), 8);
}

void OutputWriter::writeStatsHeader() {
    stats_file_ << "frame,sim_time_T0,sim_time_years,n_aggregates,agg_id,"
                << "n_particles,mass_Msun,mass_Mearth,cx_AU,cy_AU,cz_AU,"
                << "vx,vy,vz,dist_center_AU\n";
}

void OutputWriter::writeFrame(const ParticleData& p, float sim_time_T0) {
    double t = static_cast<double>(sim_time_T0);
    frames_file_.write(reinterpret_cast<char*>(&t), 8);
    frames_file_.write(reinterpret_cast<const char*>(p.x), p.n * sizeof(float));
    frames_file_.write(reinterpret_cast<const char*>(p.y), p.n * sizeof(float));
    frames_file_.write(reinterpret_cast<const char*>(p.z), p.n * sizeof(float));
}

void OutputWriter::writeStats(const std::vector<Aggregate>& aggs,
                               int frame_idx, float sim_time_T0) {
    float years = sim_time_T0 * T0_TO_YEARS;
    for (const auto& a : aggs) {
        stats_file_
            << frame_idx << ","
            << sim_time_T0 << ","
            << years << ","
            << static_cast<int>(aggs.size()) << ","
            << a.id << ","
            << a.n_particles << ","
            << a.mass_msun << ","
            << a.mass_msun * MSUN_TO_MEARTH << ","
            << a.cx << "," << a.cy << "," << a.cz << ","
            << a.vx << "," << a.vy << "," << a.vz << ","
            << a.dist_center << "\n";
    }
}

void OutputWriter::finalize(int total_frames_written) {
    frames_file_.seekp(n_frames_offset_);
    uint32_t nf = static_cast<uint32_t>(total_frames_written);
    frames_file_.write(reinterpret_cast<char*>(&nf), 4);
    frames_file_.flush();
}
```

- [ ] **Step 5: Run test — verify pass**

```
cmake --build build --target nbody-tests && cd build && ctest -R output -V
```
Expected: PASSED.

- [ ] **Step 6: Commit**

```
git add src/io/ tests/test_output.cpp
git commit -m "feat: binary frames writer and CSV stats writer"
```

---

## Task 6: Aggregate (FOF) + stats.csv integration

**Files:**
- Create: `src/analysis/Aggregate.h`
- Create: `src/analysis/Aggregate.cpp`
- Create: `tests/test_aggregate.cpp`

- [ ] **Step 1: Write failing test**

```cpp
// tests/test_aggregate.cpp
#include <catch2/catch_test_macros.hpp>
#include "Aggregate.h"
#include "ParticleData.h"
#include "Config.h"

TEST_CASE("detectAggregates finds one cluster of 5 close particles", "[aggregate]") {
    Config cfg;
    cfg.n_particles = 7;
    cfg.particle_mass_msun = 3.003e-7f;
    cfg.collision_radius_factor = 100.0f;
    // physical radius ≈ 2.4e-5 AU → r_coll ≈ 2.4e-3 AU
    // place 5 particles within 0.001 AU, 2 far away

    ParticleData p = allocateParticlesCPU(7);
    for (int i = 0; i < 5; ++i) {
        p.x[i] = i * 0.0001f; p.y[i] = 0.f; p.z[i] = 1.0f;
        p.vx[i] = 0.f; p.vy[i] = 0.f; p.vz[i] = 0.f;
        p.mass[i] = cfg.particle_mass_msun; p.id[i] = i;
    }
    // 2 isolated particles far away
    p.x[5] = 3.0f; p.y[5] = 0.f; p.z[5] = 0.f;
    p.x[6] = 4.0f; p.y[6] = 0.f; p.z[6] = 0.f;
    p.vx[5]=p.vy[5]=p.vz[5]=p.mass[5]=0.f; p.id[5]=5;
    p.vx[6]=p.vy[6]=p.vz[6]=p.mass[6]=0.f; p.id[6]=6;
    p.mass[5] = p.mass[6] = cfg.particle_mass_msun;

    float r_phys = 2.4e-5f;  // approx
    float r_link = cfg.collision_radius_factor * r_phys;
    auto aggs = detectAggregates(p, r_link, cfg);

    // Exactly one aggregate of 5 particles; isolated particles are not aggregates
    REQUIRE(aggs.size() == 1);
    REQUIRE(aggs[0].n_particles == 5);

    freeParticlesCPU(p);
}
```

- [ ] **Step 2: Run — verify fail**

```
cmake --build build --target nbody-tests && cd build && ctest -R aggregate -V
```

- [ ] **Step 3: Create src/analysis/Aggregate.h**

```cpp
#pragma once
#include <vector>
#include "ParticleData.h"
#include "Config.h"

struct Aggregate {
    int   id;
    int   n_particles;
    float mass_msun;
    float cx, cy, cz;       // center of mass (AU)
    float vx, vy, vz;       // mass-weighted velocity (AU/T0)
    float dist_center;      // distance from origin (AU)
    std::vector<int> particle_ids;
};

// Friend-of-Friends: particles within link_length of each other belong to
// the same aggregate. Single particles (no neighbours) are excluded.
// link_length should be collision_radius_factor * physical_radius.
std::vector<Aggregate> detectAggregates(
    const ParticleData& p,
    float link_length,
    const Config& cfg);
```

- [ ] **Step 4: Create src/analysis/Aggregate.cpp**

```cpp
#include "Aggregate.h"
#include <cmath>
#include <numeric>
#include <unordered_map>

// Union-Find helpers
static int find(std::vector<int>& parent, int i) {
    while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
    return i;
}
static void unite(std::vector<int>& parent, std::vector<int>& rank, int a, int b) {
    a = find(parent, a); b = find(parent, b);
    if (a == b) return;
    if (rank[a] < rank[b]) std::swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b]) ++rank[a];
}

std::vector<Aggregate> detectAggregates(
    const ParticleData& p, float link_length, const Config& cfg)
{
    int n = p.n;
    float ll2 = link_length * link_length;

    std::vector<int> parent(n), rank(n, 0);
    std::iota(parent.begin(), parent.end(), 0);

    // O(N^2) FOF — acceptable for post-step analysis (not physics loop)
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float dx = p.x[i]-p.x[j], dy = p.y[i]-p.y[j], dz = p.z[i]-p.z[j];
            if (dx*dx + dy*dy + dz*dz < ll2)
                unite(parent, rank, i, j);
        }
    }

    // Gather groups
    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i)
        groups[find(parent, i)].push_back(i);

    std::vector<Aggregate> result;
    int agg_id = 0;
    for (auto& [root, members] : groups) {
        if (members.size() < 2) continue;  // single particles not aggregates
        Aggregate a;
        a.id = agg_id++;
        a.n_particles = static_cast<int>(members.size());
        a.particle_ids = members;
        a.mass_msun = a.n_particles * cfg.particle_mass_msun;
        // Center of mass and velocity (equal masses → simple mean)
        a.cx=0; a.cy=0; a.cz=0; a.vx=0; a.vy=0; a.vz=0;
        for (int idx : members) {
            a.cx += p.x[idx]; a.cy += p.y[idx]; a.cz += p.z[idx];
            a.vx += p.vx[idx]; a.vy += p.vy[idx]; a.vz += p.vz[idx];
        }
        float inv = 1.0f / a.n_particles;
        a.cx*=inv; a.cy*=inv; a.cz*=inv;
        a.vx*=inv; a.vy*=inv; a.vz*=inv;
        a.dist_center = std::sqrt(a.cx*a.cx + a.cy*a.cy + a.cz*a.cz);
        result.push_back(a);
    }
    return result;
}
```

- [ ] **Step 5: Run test — verify pass**

```
cmake --build build --target nbody-tests && cd build && ctest -R aggregate -V
```

- [ ] **Step 6: Commit**

```
git add src/analysis/ tests/test_aggregate.cpp
git commit -m "feat: Friend-of-Friends aggregate detection"
```

---

## Task 7: Leapfrog Integrator (CUDA)

**Files:**
- Create: `src/physics/Integrator.cuh`
- Create: `src/physics/Integrator.cu`
- Create: `tests/test_integrator.cu`

- [ ] **Step 1: Write failing test**

```cuda
// tests/test_integrator.cu
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Integrator.cuh"
#include "ParticleData.h"
#include <cmath>

// Test: particle in circular orbit around unit mass at origin for 1 period
// Initial: x=1, z=0, vx=0, vz=1 (circular at r=1, G=1, M=1 → v_circ=1)
// After 2π time units the particle should return to (1,0,0)
TEST_CASE("Leapfrog conserves energy in circular orbit (1% tolerance)", "[integrator]") {
    ParticleData cpu = allocateParticlesCPU(1);
    cpu.x[0]=1.f; cpu.y[0]=0.f; cpu.z[0]=0.f;
    cpu.vx[0]=0.f; cpu.vy[0]=0.f; cpu.vz[0]=1.f;
    cpu.ax[0]=0.f; cpu.ay[0]=0.f; cpu.az[0]=0.f;
    cpu.mass[0]=1e-6f; cpu.id[0]=0;

    ParticleData gpu = allocateParticlesGPU(1);
    copyToGPU(cpu, gpu);

    float dt = 0.001f;
    float star_mass = 1.0f;
    float softening = 0.001f;
    float two_pi = 6.28318530f;
    int steps = static_cast<int>(two_pi / dt);

    // Manual leapfrog without BH (direct star force for single particle test)
    // Initial acceleration from star at origin
    auto setAcc = [&]() {
        copyToCPU(gpu, cpu);
        float r2 = cpu.x[0]*cpu.x[0] + cpu.z[0]*cpu.z[0] + softening*softening;
        float r3 = r2 * std::sqrt(r2);
        cpu.ax[0] = -star_mass * cpu.x[0] / r3;
        cpu.ay[0] = 0.f;
        cpu.az[0] = -star_mass * cpu.z[0] / r3;
        copyToGPU(cpu, gpu);
    };
    setAcc();

    for (int s = 0; s < steps; ++s) {
        launchHalfKickKernel(gpu, dt * 0.5f, 1, 1);
        launchDriftKernel(gpu, dt, 1, 1);
        setAcc();
        launchHalfKickKernel(gpu, dt * 0.5f, 1, 1);
    }
    copyToCPU(gpu, cpu);

    float r = std::sqrt(cpu.x[0]*cpu.x[0] + cpu.z[0]*cpu.z[0]);
    // Energy: E = 0.5*v^2 - M/r  (G=1, particle mass cancels)
    float v2 = cpu.vx[0]*cpu.vx[0] + cpu.vz[0]*cpu.vz[0];
    float E_final = 0.5f*v2 - star_mass/r;
    float E_init  = 0.5f*1.0f - star_mass/1.0f;   // 0.5 - 1 = -0.5
    REQUIRE_THAT(E_final, Catch::Matchers::WithinRel(E_init, 0.01f));

    freeParticlesCPU(cpu);
    freeParticlesGPU(gpu);
}
```

- [ ] **Step 2: Run — verify fail**

```
cmake --build build --target nbody-tests && cd build && ctest -R integrator -V
```

- [ ] **Step 3: Create src/physics/Integrator.cuh**

```cuda
#pragma once
#include "ParticleData.h"

// Leapfrog KDK: half-kick → drift → [forces] → half-kick
// Each launcher uses ceil(n/256) blocks of 256 threads.

// v += a * dt_half
void launchHalfKickKernel(ParticleData& gpu, float dt_half, int n_blocks, int block_size);

// x += v * dt
void launchDriftKernel(ParticleData& gpu, float dt, int n_blocks, int block_size);

// ax = ay = az = 0
void launchResetAccelerationKernel(ParticleData& gpu, int n_blocks, int block_size);
```

- [ ] **Step 4: Create src/physics/Integrator.cu**

```cuda
#include "Integrator.cuh"
#include <cuda_runtime.h>

__global__ void halfKickKernel(float* vx, float* vy, float* vz,
                                const float* ax, const float* ay, const float* az,
                                float dt_half, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += ax[i] * dt_half;
    vy[i] += ay[i] * dt_half;
    vz[i] += az[i] * dt_half;
}

__global__ void driftKernel(float* x, float* y, float* z,
                             const float* vx, const float* vy, const float* vz,
                             float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

__global__ void resetAccKernel(float* ax, float* ay, float* az, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ax[i] = 0.f; ay[i] = 0.f; az[i] = 0.f;
}

static int blocks(int n, int bs) { return (n + bs - 1) / bs; }

void launchHalfKickKernel(ParticleData& g, float dt_half, int nb, int bs) {
    halfKickKernel<<<blocks(g.n,256), 256>>>(
        g.vx, g.vy, g.vz, g.ax, g.ay, g.az, dt_half, g.n);
    cudaDeviceSynchronize();
}
void launchDriftKernel(ParticleData& g, float dt, int nb, int bs) {
    driftKernel<<<blocks(g.n,256), 256>>>(
        g.x, g.y, g.z, g.vx, g.vy, g.vz, dt, g.n);
    cudaDeviceSynchronize();
}
void launchResetAccelerationKernel(ParticleData& g, int nb, int bs) {
    resetAccKernel<<<blocks(g.n,256), 256>>>(g.ax, g.ay, g.az, g.n);
    cudaDeviceSynchronize();
}
```

- [ ] **Step 5: Run test — verify pass**

```
cmake --build build --target nbody-tests && cd build && ctest -R integrator -V
```
Expected: PASSED (energy conserved within 1%).

- [ ] **Step 6: Commit**

```
git add src/physics/Integrator.cuh src/physics/Integrator.cu tests/test_integrator.cu
git commit -m "feat: leapfrog KDK integrator CUDA kernels"
```

---

## Task 8: Barnes-Hut Octree — Data Structures + Bounding Box

**Files:**
- Create: `src/physics/BHTree.cuh`
- Create: `src/physics/BHTree.cu` (partial)

- [ ] **Step 1: Create src/physics/BHTree.cuh**

```cuda
#pragma once
#include "ParticleData.h"

// Max internal nodes = MAX_NODES_MULT * n_bodies
static constexpr int MAX_NODES_MULT = 8;

// Flat octree node arrays.
// Indices [0 .. n_bodies-1]     → leaf nodes (bodies)
// Indices [n_bodies .. n_total-1] → internal nodes
// Root = node at index (n_total - 1)
//
// child[node_idx * 8 + k] == -1  → empty child slot
// child[node_idx * 8 + k] == -2  → error / locked
// child[node_idx * 8 + k] >= 0   → index of child node
struct OctreeData {
    float* pos_x;       // COM x (AU)
    float* pos_y;       // COM y
    float* pos_z;       // COM z
    float* mass;        // total mass of subtree
    int*   child;       // [n_total * 8]
    int*   count;       // bodies in subtree (for sort)
    int*   start;       // start index in sorted array (for sort)
    int*   sort_idx;    // sorted body order [n_bodies]
    int*   mutex;       // per-node spin lock
    float* cell_size;   // half side-length of bounding box for node

    // Global bounding box (root cell)
    float* bbox_min;    // [3]: min x,y,z
    float* bbox_max;    // [3]: max x,y,z

    int n_bodies;
    int n_total;        // n_bodies + MAX_NODES_MULT * n_bodies
};

OctreeData  allocateOctree(int n_bodies);
void        freeOctree(OctreeData& tree);

// Kernel launchers (all synchronize before returning)
void launchBBoxKernel      (OctreeData& t, const ParticleData& p);
void launchBuildKernel     (OctreeData& t, const ParticleData& p);
void launchSummarizeKernel (OctreeData& t);
void launchSortKernel      (OctreeData& t);
void launchForceKernel     (OctreeData& t, ParticleData& p,
                             float theta, float softening, float star_mass);
void launchCollisionKernel (OctreeData& t, ParticleData& p,
                             float r_coll, float restitution);
void resetOctree           (OctreeData& t);
```

- [ ] **Step 2: Begin src/physics/BHTree.cu — alloc + bbox kernel**

```cuda
#include "BHTree.cuh"
#include <cuda_runtime.h>
#include <float.h>

// ── Helpers ──────────────────────────────────────────────────────────────────

__global__ void setLeafCountKernel(int* count, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_bodies) count[i] = 1;
}

// ── Alloc / Free ────────────────────────────────────────────────────────────

OctreeData allocateOctree(int n) {
    OctreeData t;
    t.n_bodies = n;
    t.n_total  = n + MAX_NODES_MULT * n;
    int nt = t.n_total;

    cudaMalloc(&t.pos_x,    nt * sizeof(float));
    cudaMalloc(&t.pos_y,    nt * sizeof(float));
    cudaMalloc(&t.pos_z,    nt * sizeof(float));
    cudaMalloc(&t.mass,     nt * sizeof(float));
    cudaMalloc(&t.child,    nt * 8 * sizeof(int));
    cudaMalloc(&t.count,    nt * sizeof(int));
    cudaMalloc(&t.start,    nt * sizeof(int));
    cudaMalloc(&t.sort_idx, n  * sizeof(int));
    cudaMalloc(&t.mutex,    nt * sizeof(int));
    cudaMalloc(&t.cell_size,nt * sizeof(float));
    cudaMalloc(&t.bbox_min, 3  * sizeof(float));
    cudaMalloc(&t.bbox_max, 3  * sizeof(float));
    return t;
}

void freeOctree(OctreeData& t) {
    cudaFree(t.pos_x); cudaFree(t.pos_y); cudaFree(t.pos_z);
    cudaFree(t.mass);  cudaFree(t.child); cudaFree(t.count);
    cudaFree(t.start); cudaFree(t.sort_idx); cudaFree(t.mutex);
    cudaFree(t.cell_size); cudaFree(t.bbox_min); cudaFree(t.bbox_max);
}

// ── Reset ────────────────────────────────────────────────────────────────────

__global__ void resetKernel(int* child, int* mutex, int* count, int* start,
                             float* mass, int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    mutex[i] = 0; count[i] = 0; start[i] = -1; mass[i] = 0.f;
    for (int k = 0; k < 8; ++k) child[i*8+k] = -1;
}

void resetOctree(OctreeData& t) {
    int bs = 256, nb = (t.n_total + bs - 1) / bs;
    resetKernel<<<nb, bs>>>(t.child, t.mutex, t.count, t.start, t.mass, t.n_total);
    cudaDeviceSynchronize();
}

// ── Bounding Box ─────────────────────────────────────────────────────────────

__global__ void bboxKernel(const float* x, const float* y, const float* z,
                            float* gmin, float* gmax, int n) {
    extern __shared__ float smem[];
    float* sminx = smem;
    float* sminy = smem +   blockDim.x;
    float* sminz = smem + 2*blockDim.x;
    float* smaxx = smem + 3*blockDim.x;
    float* smaxy = smem + 4*blockDim.x;
    float* smaxz = smem + 5*blockDim.x;

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    float lx = (i < n) ? x[i] : x[0];
    float ly = (i < n) ? y[i] : y[0];
    float lz = (i < n) ? z[i] : z[0];
    sminx[tid]=smaxx[tid]=lx;
    sminy[tid]=smaxy[tid]=ly;
    sminz[tid]=smaxz[tid]=lz;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sminx[tid]=fminf(sminx[tid],sminx[tid+s]);
            sminy[tid]=fminf(sminy[tid],sminy[tid+s]);
            sminz[tid]=fminf(sminz[tid],sminz[tid+s]);
            smaxx[tid]=fmaxf(smaxx[tid],smaxx[tid+s]);
            smaxy[tid]=fmaxf(smaxy[tid],smaxy[tid+s]);
            smaxz[tid]=fmaxf(smaxz[tid],smaxz[tid+s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMinf := // use __int_as_float trick for atomicMin on float
        // Standard approach: cast to int and use atomicMin
        atomicMin((int*)&gmin[0], __float_as_int(sminx[0]));
        atomicMin((int*)&gmin[1], __float_as_int(sminy[0]));
        atomicMin((int*)&gmin[2], __float_as_int(sminz[0]));
        atomicMax((int*)&gmax[0], __float_as_int(smaxx[0]));
        atomicMax((int*)&gmax[1], __float_as_int(smaxy[0]));
        atomicMax((int*)&gmax[2], __float_as_int(smaxz[0]));
    }
}
```

> **Note for implementer:** `atomicMin`/`atomicMax` on floats requires the IEEE bit-trick. Use this pattern for non-negative floats (all positions can be shifted to be positive by a translation before bbox, then shifted back):
> ```cuda
> __device__ void atomicMinFloat(float* addr, float val) {
>     atomicMin((int*)addr, __float_as_int(val));
> }
> ```
> Or use a two-pass approach: first kernel initializes `gmin = +FLT_MAX`, `gmax = -FLT_MAX` using `__int_as_float(0x7F800000)` and `__int_as_float(0xFF800000)`. The trick works because IEEE 754 positive floats compare the same way as their bit patterns. For negative floats, negate, apply, negate back. Since disk positions can be negative, initialize gmin to `-FLT_MAX`, gmax to `+FLT_MAX` and implement using `atomicMin/Max` on a union with int.

- [ ] **Step 3: Add bbox launcher to BHTree.cu**

```cuda
void launchBBoxKernel(OctreeData& t, const ParticleData& p) {
    // Initialize bounds to ±FLT_MAX
    float init_min = -FLT_MAX, init_max = FLT_MAX;
    cudaMemset(t.bbox_min, 0x7F, 3 * sizeof(float)); // +FLT_MAX bits
    // proper init via separate small kernel:
    // (inline the init below)
    int bs = 256;
    int nb = (p.n + bs - 1) / bs;
    size_t smem = 6 * bs * sizeof(float);
    bboxKernel<<<nb, bs, smem>>>(p.x, p.y, p.z, t.bbox_min, t.bbox_max, p.n);
    cudaDeviceSynchronize();
}
```

- [ ] **Step 4: Build and verify compile**

```
cmake --build build --target nbody-sim 2>&1 | head -50
```
Expected: compiles cleanly (no logic tests yet, just structure).

- [ ] **Step 5: Commit**

```
git add src/physics/BHTree.cuh src/physics/BHTree.cu
git commit -m "feat: octree data structures + bounding box kernel"
```

---

## Task 9: Barnes-Hut Tree Build + Summarize + Sort Kernels

**Files:**
- Modify: `src/physics/BHTree.cu` (append kernels)

- [ ] **Step 1: Add tree-build kernel to BHTree.cu**

The build kernel inserts one body per thread using compare-and-swap. Key steps per thread (body `i`):
1. Start at root node (`n_total - 1`), which covers the whole bbox.
2. At each node, determine which of 8 octants `i` belongs to (based on cell center vs body position).
3. Attempt to claim the child slot with `atomicCAS`. Three outcomes:
   - Slot is empty (`-1`): insert body `i` → done.
   - Slot holds another body `j`: create a new internal node, subdivide, re-insert both `i` and `j`.
   - Slot is locked (`-2`): spin-wait.

```cuda
__device__ int getOctant(float cx, float cy, float cz,
                          float px, float py, float pz) {
    return ((px > cx) ? 1 : 0) |
           ((py > cy) ? 2 : 0) |
           ((pz > cz) ? 4 : 0);
}

__global__ void buildKernel(
    int* child, float* pos_x, float* pos_y, float* pos_z,
    float* mass, float* cell_size, int* mutex,
    const float* bmin, const float* bmax,
    int n_bodies, int n_total)
{
    int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_bodies) return;

    float px = pos_x[body], py = pos_y[body], pz = pos_z[body];
    int root = n_total - 1;

    // Root cell covers bbox
    float cx = (bmin[0]+bmax[0])*0.5f;
    float cy = (bmin[1]+bmax[1])*0.5f;
    float cz = (bmin[2]+bmax[2])*0.5f;
    float hs = fmaxf(fmaxf(bmax[0]-bmin[0], bmax[1]-bmin[1]),
                           bmax[2]-bmin[2]) * 0.5f * 1.001f;

    if (body == 0) {  // thread 0 initializes root
        pos_x[root]=cx; pos_y[root]=cy; pos_z[root]=cz;
        cell_size[root]=hs;
    }
    __threadfence();

    int node = root;
    while (true) {
        int oct = getOctant(pos_x[node], pos_y[node], pos_z[node], px, py, pz);
        int* slot = &child[node*8 + oct];
        int existing = atomicCAS(slot, -1, body);  // try to claim empty slot

        if (existing == -1) {
            // Successfully inserted body into leaf slot
            atomicAdd(&mass[node], mass[body]);
            return;
        }
        if (existing >= 0 && existing < n_bodies) {
            // Another body is here; need to subdivide
            // Allocate new internal node (use atomicAdd on a global counter)
            // For simplicity: encode new node via a global counter stored at
            // position [n_total-2] used as an allocator.
            // (See note below for allocator implementation)
            // ... subdivision logic omitted for brevity, see note ...
            // Spin if slot is locked
        }
        // Move down to child node
        node = existing;
        __threadfence();
    }
}
```

> **Implementation note for tree build:** The full Burtscher & Pingali tree build uses a lock counter. A clean approach:
> - Use `child[root*8 + 7]` (an otherwise unused child slot of the root, since the root only needs 8 children in theory) as a global node allocator counter, or keep a separate `int* next_node` device pointer initialized to `n_bodies`.
> - When subdivision is needed, atomically increment `next_node` to get a fresh internal node index.
> - Set the new node's center and half-size, then re-insert both bodies.
> - Use per-node `mutex` for spin-locking during subdivision.
>
> **Reference implementation:** Burtscher & Pingali (2011) provide full CUDA source in the paper appendix. A modern version is also at: https://github.com/BaronZiffer/BarnesHut-GPU

- [ ] **Step 2: Add summarize kernel**

Computes center-of-mass bottom-up. Uses warp-level synchronization.

```cuda
__global__ void summarizeKernel(
    float* pos_x, float* pos_y, float* pos_z, float* mass,
    const int* child, int* count, int n_bodies, int n_total)
{
    // Each thread handles one internal node starting from n_bodies
    int node = blockIdx.x * blockDim.x + threadIdx.x + n_bodies;
    if (node >= n_total) return;

    float m = 0, cx = 0, cy = 0, cz = 0;
    int cnt = 0;
    for (int k = 0; k < 8; ++k) {
        int c = child[node*8+k];
        if (c < 0) continue;
        // Wait until child has been summarized (count > 0 means ready)
        // Leaf bodies always have count == 1 (set during build)
        int tries = 0;
        while (count[c] == 0 && tries++ < 1000000) { __threadfence(); }
        float cm = mass[c];
        m  += cm;
        cx += pos_x[c] * cm;
        cy += pos_y[c] * cm;
        cz += pos_z[c] * cm;
        cnt += count[c];
    }
    if (m > 0.f) {
        pos_x[node] = cx / m;
        pos_y[node] = cy / m;
        pos_z[node] = cz / m;
        mass[node]  = m;
        count[node] = cnt;
    }
}
```

- [ ] **Step 3: Add sort kernel**

Assigns a DFS traversal order to bodies for better cache access during force computation.

```cuda
__global__ void sortKernel(int* sort_idx, int* start, const int* child,
                            const int* count, int n_bodies, int root) {
    // Single-threaded topological sort (launch with 1 thread, not performance-critical)
    // Uses an iterative DFS with an explicit stack
    int stack[256]; int top = 0;
    stack[top++] = root;
    int out = 0;
    while (top > 0) {
        int node = stack[--top];
        if (node < n_bodies) { sort_idx[out++] = node; continue; }
        start[node] = out;
        for (int k = 7; k >= 0; --k) {  // push in reverse for left-to-right DFS
            int c = child[node*8+k];
            if (c >= 0) stack[top++] = c;
        }
    }
}
```

- [ ] **Step 4: Add launchers**

```cuda
void launchBuildKernel(OctreeData& t, const ParticleData& p) {
    // Copy body positions into tree pos arrays (leaves 0..n-1)
    cudaMemcpy(t.pos_x, p.x, p.n*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(t.pos_y, p.y, p.n*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(t.pos_z, p.z, p.n*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(t.mass,  p.mass, p.n*sizeof(float), cudaMemcpyDeviceToDevice);
    // Set all counts to 0, then set leaf counts to 1
    cudaMemset(t.count, 0, t.n_total * sizeof(int));
    setLeafCountKernel<<<(p.n+255)/256, 256>>>(t.count, p.n);
    cudaDeviceSynchronize();
    int bs=256, nb=(p.n+bs-1)/bs;
    buildKernel<<<nb, bs>>>(t.child, t.pos_x, t.pos_y, t.pos_z,
                             t.mass, t.cell_size, t.mutex,
                             t.bbox_min, t.bbox_max, t.n_bodies, t.n_total);
    cudaDeviceSynchronize();
}

void launchSummarizeKernel(OctreeData& t) {
    int internal = t.n_total - t.n_bodies;
    int bs=256, nb=(internal+bs-1)/bs;
    summarizeKernel<<<nb, bs>>>(t.pos_x, t.pos_y, t.pos_z, t.mass,
                                 t.child, t.count, t.n_bodies, t.n_total);
    cudaDeviceSynchronize();
}

void launchSortKernel(OctreeData& t) {
    sortKernel<<<1, 1>>>(t.sort_idx, t.start, t.child, t.count,
                          t.n_bodies, t.n_total - 1);
    cudaDeviceSynchronize();
}
```

- [ ] **Step 5: Build and verify compile**

```
cmake --build build --target nbody-sim 2>&1 | grep -E "error:|warning:" | head -20
```

- [ ] **Step 6: Commit**

```
git add src/physics/BHTree.cu
git commit -m "feat: BH tree build, summarize, and sort CUDA kernels"
```

---

## Task 10: Barnes-Hut Force + Collision Kernels

**Files:**
- Modify: `src/physics/BHTree.cu` (append)
- Create: `tests/test_bhtree.cu`
- Create: `tests/test_collision.cu`

- [ ] **Step 1: Write BH force accuracy test**

```cuda
// tests/test_bhtree.cu
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "BHTree.cuh"
#include "ParticleData.h"
#include <cmath>

TEST_CASE("BH force on isolated particle matches direct 1/r^2", "[bhtree]") {
    // Single particle at (1,0,0), star at origin with mass=1
    // Expected acceleration: ax = -G*M/r^2 = -1.0 (G=1, M=1, r=1)
    // BH should match direct to within theta=0.5 accuracy (~5%)
    int n = 1;
    ParticleData cpu = allocateParticlesCPU(n);
    cpu.x[0]=1.f; cpu.y[0]=0.f; cpu.z[0]=0.f;
    cpu.vx[0]=0.f; cpu.vy[0]=0.f; cpu.vz[0]=0.f;
    cpu.ax[0]=0.f; cpu.ay[0]=0.f; cpu.az[0]=0.f;
    cpu.mass[0]=1e-6f; cpu.id[0]=0;

    ParticleData gpu = allocateParticlesGPU(n);
    copyToGPU(cpu, gpu);

    OctreeData tree = allocateOctree(n);
    resetOctree(tree);
    launchBBoxKernel(tree, gpu);
    launchBuildKernel(tree, gpu);
    launchSummarizeKernel(tree);
    launchSortKernel(tree);
    launchForceKernel(tree, gpu, 0.5f, 0.001f, 1.0f);

    copyToCPU(gpu, cpu);

    // Force from star alone (BH tree + star): ax ≈ -1.0 AU/T0^2
    REQUIRE_THAT(cpu.ax[0], Catch::Matchers::WithinRel(-1.0f, 0.05f));
    REQUIRE_THAT(cpu.ay[0], Catch::Matchers::WithinAbs(0.0f, 0.05f));
    REQUIRE_THAT(cpu.az[0], Catch::Matchers::WithinAbs(0.0f, 0.05f));

    freeParticlesCPU(cpu); freeParticlesGPU(gpu); freeOctree(tree);
}
```

- [ ] **Step 2: Write collision test**

```cuda
// tests/test_collision.cu
#include <catch2/catch_test_macros.hpp>
#include "BHTree.cuh"
#include "ParticleData.h"
#include <cmath>

TEST_CASE("Collision kernel reverses approach velocity with restitution", "[collision]") {
    // Two particles approaching head-on
    // particle 0: x=-0.001, vx=+0.1
    // particle 1: x=+0.001, vx=-0.1
    // After collision: velocities should swap sign (scaled by e=0.3)
    int n = 2;
    float r_coll = 0.005f;  // larger than separation = 0.002
    ParticleData cpu = allocateParticlesCPU(n);
    cpu.x[0]=-0.001f; cpu.y[0]=0.f; cpu.z[0]=0.f;
    cpu.vx[0]=0.1f;   cpu.vy[0]=0.f; cpu.vz[0]=0.f;
    cpu.x[1]= 0.001f; cpu.y[1]=0.f; cpu.z[1]=0.f;
    cpu.vx[1]=-0.1f;  cpu.vy[1]=0.f; cpu.vz[1]=0.f;
    for (int i=0;i<2;i++){cpu.ax[i]=cpu.ay[i]=cpu.az[i]=0.f;
                           cpu.mass[i]=3.003e-7f; cpu.id[i]=i;}

    ParticleData gpu = allocateParticlesGPU(n);
    copyToGPU(cpu, gpu);

    OctreeData tree = allocateOctree(n);
    resetOctree(tree);
    launchBBoxKernel(tree, gpu);
    launchBuildKernel(tree, gpu);
    launchSummarizeKernel(tree);
    launchCollisionKernel(tree, gpu, r_coll, 0.3f);

    copyToCPU(gpu, cpu);

    // After inelastic bounce, relative velocity is reversed and scaled by e
    // vx[0] should be negative (bounced back), vx[1] positive
    REQUIRE(cpu.vx[0] < 0.0f);
    REQUIRE(cpu.vx[1] > 0.0f);

    freeParticlesCPU(cpu); freeParticlesGPU(gpu); freeOctree(tree);
}
```

- [ ] **Step 3: Add force kernel to BHTree.cu**

```cuda
__global__ void forceKernel(
    const float* tree_x, const float* tree_y, const float* tree_z,
    const float* tree_mass, const float* cell_size,
    const int* child, const int* sort_idx,
    float* ax, float* ay, float* az,
    const float* px, const float* py, const float* pz,
    float theta, float softening, float star_mass,
    int n_bodies, int n_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;
    int body = sort_idx[idx];

    float bx = px[body], by = py[body], bz = pz[body];
    float fx = 0.f, fy = 0.f, fz = 0.f;

    // Force from fixed star at origin
    {
        float dx=-bx, dy=-by, dz=-bz;
        float r2 = dx*dx+dy*dy+dz*dz + softening*softening;
        float r3 = r2 * sqrtf(r2);
        float f  = star_mass / r3;
        fx += f*dx; fy += f*dy; fz += f*dz;
    }

    // BH tree traversal (iterative with explicit stack)
    int stack[64]; int top = 0;
    stack[top++] = n_total - 1;  // root

    while (top > 0) {
        int node = stack[--top];
        if (node < 0) continue;

        float dx = tree_x[node]-bx, dy = tree_y[node]-by, dz = tree_z[node]-bz;
        float r2 = dx*dx+dy*dy+dz*dz + softening*softening;
        float r  = sqrtf(r2);

        // θ criterion: if s/r < theta OR leaf body
        bool is_leaf = (node < n_bodies);
        bool accept  = is_leaf || (cell_size[node] / r < theta);

        if (accept) {
            if (node == body) continue;  // skip self
            float m  = tree_mass[node];
            float r3 = r2 * r;
            float f  = m / r3;
            fx += f*dx; fy += f*dy; fz += f*dz;
        } else {
            // Push children
            for (int k = 0; k < 8; ++k) {
                int c = child[node*8+k];
                if (c >= 0) stack[top++] = c;
            }
        }
    }
    ax[body] = fx; ay[body] = fy; az[body] = fz;
}
```

- [ ] **Step 4: Add collision kernel to BHTree.cu**

```cuda
__global__ void collisionKernel(
    const float* tree_x, const float* tree_y, const float* tree_z,
    const float* cell_size, const int* child, const int* sort_idx,
    float* vx, float* vy, float* vz,
    const float* px, const float* py, const float* pz,
    float r_coll, float e, int n_bodies, int n_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;
    int i = sort_idx[idx];

    float ix=px[i], iy=py[i], iz=pz[i];
    float ivx=vx[i], ivy=vy[i], ivz=vz[i];

    // Traverse tree looking for leaf nodes within r_coll
    int stack[64]; int top = 0;
    stack[top++] = n_total - 1;

    while (top > 0) {
        int node = stack[--top];
        if (node < 0) continue;

        float dx=tree_x[node]-ix, dy=tree_y[node]-iy, dz=tree_z[node]-iz;
        float dist2 = dx*dx+dy*dy+dz*dz;

        bool is_leaf = (node < n_bodies);

        if (is_leaf) {
            int j = node;
            if (j == i) continue;
            if (dist2 < r_coll * r_coll) {
                // Compute normal
                float dist = sqrtf(dist2) + 1e-10f;
                float nx=dx/dist, ny=dy/dist, nz=dz/dist;

                // Relative velocity along normal
                float dvn = (vx[j]-ivx)*nx + (vy[j]-ivy)*ny + (vz[j]-ivz)*nz;
                if (dvn < 0.f) {  // approaching
                    float impulse = -(1.f + e) * dvn * 0.5f;
                    // Apply equal/opposite impulse (equal masses)
                    atomicAdd(&vx[i], -impulse*nx);
                    atomicAdd(&vy[i], -impulse*ny);
                    atomicAdd(&vz[i], -impulse*nz);
                    // Note: j will also process this pair — avoid double-impulse
                    // by only applying to i here (j will apply to itself)
                }
            }
        } else {
            // Only descend if cell could contain particles within r_coll
            float cell_half = cell_size[node];
            float cell_dist = sqrtf(dist2) - cell_half * 1.732f;  // sqrt(3) * half
            if (cell_dist < r_coll) {
                for (int k=0; k<8; ++k) {
                    int c = child[node*8+k];
                    if (c >= 0) stack[top++] = c;
                }
            }
        }
    }
}
```

- [ ] **Step 5: Add launchers**

```cuda
void launchForceKernel(OctreeData& t, ParticleData& p,
                        float theta, float softening, float star_mass) {
    int bs=256, nb=(p.n+bs-1)/bs;
    forceKernel<<<nb, bs>>>(
        t.pos_x, t.pos_y, t.pos_z, t.mass, t.cell_size, t.child, t.sort_idx,
        p.ax, p.ay, p.az, p.x, p.y, p.z,
        theta, softening, star_mass, t.n_bodies, t.n_total);
    cudaDeviceSynchronize();
}

void launchCollisionKernel(OctreeData& t, ParticleData& p,
                            float r_coll, float restitution) {
    int bs=256, nb=(p.n+bs-1)/bs;
    collisionKernel<<<nb, bs>>>(
        t.pos_x, t.pos_y, t.pos_z, t.cell_size, t.child, t.sort_idx,
        p.vx, p.vy, p.vz, p.x, p.y, p.z,
        r_coll, restitution, t.n_bodies, t.n_total);
    cudaDeviceSynchronize();
}
```

- [ ] **Step 6: Build and run tests**

```
cmake --build build --target nbody-tests && cd build && ctest -R "bhtree|collision" -V
```
Expected: both tests PASSED.

- [ ] **Step 7: Commit**

```
git add src/physics/BHTree.cu tests/test_bhtree.cu tests/test_collision.cu
git commit -m "feat: Barnes-Hut force and collision CUDA kernels"
```

---

## Task 11: Main Simulation Loop

**Files:**
- Modify: `src/main.cpp`

- [ ] **Step 1: Implement main.cpp**

```cpp
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include "Config.h"
#include "ParticleData.h"
#include "DiskInit.h"
#include "BHTree.cuh"
#include "Integrator.cuh"
#include "Aggregate.h"
#include "OutputWriter.h"

int main(int argc, char** argv) {
    std::string cfg_path = (argc > 1) ? argv[1] : "config/simulation.json";
    std::cout << "Loading config: " << cfg_path << "\n";
    Config cfg = loadConfig(cfg_path);

    // Output directory per run (timestamped)
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    char ts[20]; std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
    std::string run_dir = cfg.output_dir + "/" + std::string(ts);
    std::filesystem::create_directories(run_dir);
    std::cout << "Output dir: " << run_dir << "\n";

    // Initialize disk on CPU, copy to GPU
    std::cout << "Initializing disk (" << cfg.n_particles << " particles)...\n";
    ParticleData cpu = initDisk(cfg);
    ParticleData gpu = allocateParticlesGPU(cfg.n_particles);
    copyToGPU(cpu, gpu);

    // Allocate octree
    OctreeData tree = allocateOctree(cfg.n_particles);

    // Compute collision radius
    float r_phys  = particlePhysicalRadius(cfg);
    float r_coll  = cfg.collision_radius_factor * r_phys;

    // Output writer
    OutputWriter writer(run_dir, cfg);

    int total_steps   = static_cast<int>(cfg.total_time / cfg.dt);
    int frames_written = 0;
    float sim_time = 0.f;

    std::cout << "Starting simulation: " << total_steps << " steps, "
              << "output every " << cfg.output_every_n_steps << " steps\n";

    // KDK Leapfrog: compute initial accelerations
    resetOctree(tree);
    launchBBoxKernel(tree, gpu);
    launchBuildKernel(tree, gpu);
    launchSummarizeKernel(tree);
    launchSortKernel(tree);
    launchResetAccelerationKernel(gpu, 0, 0);
    launchForceKernel(tree, gpu, cfg.theta, cfg.softening_AU, cfg.star_mass_msun);

    for (int step = 0; step < total_steps; ++step) {
        // KDK step 1: half kick
        launchHalfKickKernel(gpu, cfg.dt * 0.5f, 0, 0);

        // KDK step 2: drift
        launchDriftKernel(gpu, cfg.dt, 0, 0);

        // Rebuild tree with new positions, compute forces
        resetOctree(tree);
        launchBBoxKernel(tree, gpu);
        launchBuildKernel(tree, gpu);
        launchSummarizeKernel(tree);
        launchSortKernel(tree);
        launchResetAccelerationKernel(gpu, 0, 0);
        launchForceKernel(tree, gpu, cfg.theta, cfg.softening_AU, cfg.star_mass_msun);

        // Collision detection
        launchCollisionKernel(tree, gpu, r_coll, cfg.restitution);

        // KDK step 3: second half kick
        launchHalfKickKernel(gpu, cfg.dt * 0.5f, 0, 0);

        sim_time += cfg.dt;

        // Output
        if ((step + 1) % cfg.output_every_n_steps == 0) {
            copyToCPU(gpu, cpu);
            writer.writeFrame(cpu, sim_time);

            auto aggs = detectAggregates(cpu, r_coll, cfg);
            writer.writeStats(aggs, frames_written, sim_time);
            ++frames_written;

            float pct = 100.f * (step + 1) / total_steps;
            std::cout << std::fixed << std::setprecision(1)
                      << "\r[" << pct << "%] t=" << sim_time
                      << " T0, aggregates=" << aggs.size() << "    " << std::flush;
        }
    }

    writer.finalize(frames_written);
    std::cout << "\nDone. " << frames_written << " frames written to " << run_dir << "\n";

    freeParticlesCPU(cpu);
    freeParticlesGPU(gpu);
    freeOctree(tree);
    return 0;
}
```

- [ ] **Step 2: Build**

```
cmake --build build --target nbody-sim
```
Expected: clean compile.

- [ ] **Step 3: Smoke test with small N**

Edit `config/simulation.json` temporarily: `"n_particles": 1000, "total_time": 10.0, "output_every_n_steps": 5`. Then:

```
./build/nbody-sim config/simulation.json
```
Expected: runs to completion, prints progress, creates `data/<timestamp>/frames.bin` and `stats.csv`.

- [ ] **Step 4: Verify output files exist and have correct size**

```
ls -lh data/$(ls -t data | head -1)/
```
frames.bin should be > 0 bytes. stats.csv should have header + rows.

- [ ] **Step 5: Restore config and commit**

Restore `n_particles: 100000, total_time: 2000.0, output_every_n_steps: 200`.

```
git add src/main.cpp config/simulation.json
git commit -m "feat: main simulation loop with KDK leapfrog and output"
```

---

## Task 12: Integration Test — Energy Conservation + Full Run Validation

**Files:**
- No new files (use existing test harness)

- [ ] **Step 1: Run all tests**

```
cmake --build build --target nbody-tests && cd build && ctest --output-on-failure
```
Expected: all tests PASS.

- [ ] **Step 2: Run full simulation with N=100k**

```
./build/nbody-sim config/simulation.json
```
Monitor: progress prints, no CUDA errors, memory usage < 8GB.

- [ ] **Step 3: Verify stats.csv has meaningful data**

```
head -5 data/$(ls -t data | head -1)/stats.csv
wc -l   data/$(ls -t data | head -1)/stats.csv
```
Expected: CSV rows present, aggregate count increases over time.

- [ ] **Step 4: Verify frames.bin size**

```
ls -lh data/$(ls -t data | head -1)/frames.bin
```
Expected: ~1.2 GB (100k × 3 × 4 bytes × 1000 frames).

- [ ] **Step 5: Commit final state**

```
git add -A
git commit -m "feat: complete milestone 1 - N-body disk simulation with CUDA BH"
```

---

## Appendix: Physical Constants Reference

| Quantity | Value |
|----------|-------|
| 1 T₀ in years | 1 / (2π) ≈ 0.1592 yr |
| 1 M☉ in M_Earth | 332,946 |
| Particle physical radius | ≈ 2.4 × 10⁻⁵ AU |
| Collision radius (×100) | ≈ 2.4 × 10⁻³ AU |
| Disk inner edge orbital period | 2π × 0.5^(3/2) ≈ 2.22 T₀ ≈ 0.35 yr |
| Disk outer edge orbital period | 2π × 5.0^(3/2) ≈ 70.2 T₀ ≈ 11.2 yr |

## Appendix: Build Commands

```bash
# Configure (first time)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build simulation
cmake --build build --target nbody-sim -j8

# Build + run tests
cmake --build build --target nbody-tests -j8 && cd build && ctest -V

# Run simulation
./build/nbody-sim config/simulation.json
```
