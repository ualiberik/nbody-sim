#include <catch2/catch_test_macros.hpp>
#include "BHTree.cuh"
#include "ParticleData.h"
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: verify sort_idx is a permutation of [0..n-1]
// ---------------------------------------------------------------------------
static void checkPermutation(int* d_sort_idx, int n) {
    std::vector<int> sorted(n);
    cudaMemcpy(sorted.data(), d_sort_idx, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<bool> seen(n, false);
    for (int i = 0; i < n; ++i) {
        REQUIRE(sorted[i] >= 0);
        REQUIRE(sorted[i] < n);
        REQUIRE_FALSE(seen[sorted[i]]);
        seen[sorted[i]] = true;
    }
    for (int i = 0; i < n; ++i) REQUIRE(seen[i]);
}

// ---------------------------------------------------------------------------
// Small sanity test: 10 particles on a straight line (trivial tree structure)
// ---------------------------------------------------------------------------
TEST_CASE("BH octree builds correctly for 10 particles on a line", "[bhtree][small]") {
    int n = 10;
    ParticleData cpu = allocateParticlesCPU(n);

    for (int i = 0; i < n; ++i) {
        cpu.x[i]    = (float)i * 0.5f;   // 0, 0.5, 1.0 ... 4.5 AU on X axis
        cpu.y[i]    = 0.f;
        cpu.z[i]    = 0.f;
        cpu.vx[i]   = cpu.vy[i] = cpu.vz[i] = 0.f;
        cpu.ax[i]   = cpu.ay[i] = cpu.az[i] = 0.f;
        cpu.mass[i] = 3.003e-7f;
        cpu.id[i]   = i;
    }

    ParticleData gpu = allocateParticlesGPU(n);
    copyToGPU(cpu, gpu);

    OctreeData tree = allocateOctree(n);
    resetOctree(tree);

    REQUIRE_NOTHROW(launchBBoxKernel(tree, gpu));
    REQUIRE_NOTHROW(launchBuildKernel(tree, gpu));
    REQUIRE_NOTHROW(launchSummarizeKernel(tree));
    REQUIRE_NOTHROW(launchSortKernel(tree));

    checkPermutation(tree.sort_idx, n);

    freeParticlesCPU(cpu);
    freeParticlesGPU(gpu);
    freeOctree(tree);
}

// ---------------------------------------------------------------------------
// Main test: 1000 disk-like particles
// ---------------------------------------------------------------------------
TEST_CASE("BH octree builds without crash for 1000 random particles", "[bhtree]") {
    int n = 1000;
    ParticleData cpu = allocateParticlesCPU(n);

    // Disk-like distribution in XZ plane, r in [0.5, 5] AU
    float pi = 3.14159265f;
    for (int i = 0; i < n; ++i) {
        float r   = 0.5f + 4.5f * (float)i / (n - 1);
        float phi = 2.f * pi * (float)i / n;
        cpu.x[i]    = r * std::cos(phi);
        cpu.y[i]    = 0.01f * (float)(i % 10 - 5);  // small vertical spread
        cpu.z[i]    = r * std::sin(phi);
        cpu.vx[i]   = cpu.vy[i] = cpu.vz[i] = 0.f;
        cpu.ax[i]   = cpu.ay[i] = cpu.az[i] = 0.f;
        cpu.mass[i] = 3.003e-7f;
        cpu.id[i]   = i;
    }

    ParticleData gpu = allocateParticlesGPU(n);
    copyToGPU(cpu, gpu);

    OctreeData tree = allocateOctree(n);
    resetOctree(tree);

    REQUIRE_NOTHROW(launchBBoxKernel(tree, gpu));
    REQUIRE_NOTHROW(launchBuildKernel(tree, gpu));
    REQUIRE_NOTHROW(launchSummarizeKernel(tree));
    REQUIRE_NOTHROW(launchSortKernel(tree));

    // Verify sort_idx is a permutation of [0..n-1]
    checkPermutation(tree.sort_idx, n);

    freeParticlesCPU(cpu);
    freeParticlesGPU(gpu);
    freeOctree(tree);
}
