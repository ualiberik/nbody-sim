#include <catch2/catch_test_macros.hpp>
#include "BHTree.cuh"
#include "ParticleData.h"
#include <cmath>

TEST_CASE("Collision kernel reverses approach velocity with restitution", "[collision]") {
    // Two particles approaching head-on along x axis
    // particle 0: x=-0.001, vx=+0.1
    // particle 1: x=+0.001, vx=-0.1
    // separation = 0.002 < r_coll = 0.005
    // After collision: velocities should reverse (scaled by e=0.3)
    int n = 2;
    float r_coll = 0.005f;
    ParticleData cpu = allocateParticlesCPU(n);
    cpu.x[0]=-0.001f; cpu.y[0]=0.f; cpu.z[0]=0.f;
    cpu.vx[0]= 0.1f;  cpu.vy[0]=0.f; cpu.vz[0]=0.f;
    cpu.x[1]= 0.001f; cpu.y[1]=0.f; cpu.z[1]=0.f;
    cpu.vx[1]=-0.1f;  cpu.vy[1]=0.f; cpu.vz[1]=0.f;
    for (int i = 0; i < 2; ++i) {
        cpu.ax[i]=cpu.ay[i]=cpu.az[i]=0.f;
        cpu.mass[i]=3.003e-7f;
        cpu.id[i]=i;
    }

    ParticleData gpu = allocateParticlesGPU(n);
    copyToGPU(cpu, gpu);

    OctreeData tree = allocateOctree(n);
    resetOctree(tree);
    launchBBoxKernel(tree, gpu);
    launchBuildKernel(tree, gpu);
    launchSummarizeKernel(tree);
    launchSortKernel(tree);
    launchCollisionKernel(tree, gpu, r_coll, 0.3f);

    copyToCPU(gpu, cpu);

    // After inelastic bounce: both particles should reverse x-velocity direction
    REQUIRE(cpu.vx[0] < 0.0f);  // was +0.1, should go negative after being hit
    REQUIRE(cpu.vx[1] > 0.0f);  // was -0.1, should go positive

    freeParticlesCPU(cpu); freeParticlesGPU(gpu); freeOctree(tree);
}
