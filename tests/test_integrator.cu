#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Integrator.cuh"
#include "ParticleData.h"
#include <cmath>

// Particle at (1,0,0), velocity (0,0,1) orbits unit mass at origin.
// G=1, M=1, r=1 -> v_circ=1 -> period = 2*pi.
// After one full period, total mechanical energy should be conserved to 1%.
TEST_CASE("Leapfrog conserves energy in circular orbit (1% tolerance)", "[integrator]") {
    ParticleData cpu = allocateParticlesCPU(1);
    cpu.x[0]=1.f; cpu.y[0]=0.f; cpu.z[0]=0.f;
    cpu.vx[0]=0.f; cpu.vy[0]=0.f; cpu.vz[0]=1.f;
    cpu.ax[0]=0.f; cpu.ay[0]=0.f; cpu.az[0]=0.f;
    cpu.mass[0]=1e-6f; cpu.id[0]=0;

    ParticleData gpu = allocateParticlesGPU(1);
    copyToGPU(cpu, gpu);

    float dt          = 0.01f;
    float star_mass   = 1.0f;
    float softening   = 0.001f;
    int   steps       = static_cast<int>(6.28318530f / dt);  // one period (~629 steps)

    // Helper: recompute acceleration from star (CPU side, 1 particle)
    auto setAccFromStar = [&]() {
        copyToCPU(gpu, cpu);
        float r2 = cpu.x[0]*cpu.x[0] + cpu.y[0]*cpu.y[0] + cpu.z[0]*cpu.z[0]
                   + softening*softening;
        float r3 = r2 * std::sqrt(r2);
        cpu.ax[0] = -star_mass * cpu.x[0] / r3;
        cpu.ay[0] = -star_mass * cpu.y[0] / r3;
        cpu.az[0] = -star_mass * cpu.z[0] / r3;
        copyToGPU(cpu, gpu);
    };

    // Initial acceleration
    setAccFromStar();

    for (int s = 0; s < steps; ++s) {
        launchHalfKickKernel(gpu, dt * 0.5f);
        launchDriftKernel(gpu, dt);
        setAccFromStar();
        launchHalfKickKernel(gpu, dt * 0.5f);
    }

    copyToCPU(gpu, cpu);

    float r  = std::sqrt(cpu.x[0]*cpu.x[0] + cpu.y[0]*cpu.y[0] + cpu.z[0]*cpu.z[0]);
    float v2 = cpu.vx[0]*cpu.vx[0] + cpu.vy[0]*cpu.vy[0] + cpu.vz[0]*cpu.vz[0];

    float E_init  = 0.5f * 1.0f - star_mass / 1.0f;    // = -0.5
    float E_final = 0.5f * v2   - star_mass / r;

    REQUIRE_THAT(E_final, Catch::Matchers::WithinRel(E_init, 0.01f));

    freeParticlesCPU(cpu);
    freeParticlesGPU(gpu);
}
