#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "ParticleData.h"
#include "DiskInit.h"
#include "Config.h"
#include <cmath>

TEST_CASE("CPU particle allocation and free", "[particles]") {
    ParticleData p = allocateParticlesCPU(100);
    REQUIRE(p.n == 100);
    REQUIRE(p.x != nullptr);
    REQUIRE(p.vx != nullptr);
    REQUIRE(p.mass != nullptr);
    p.x[0] = 1.5f; p.vx[0] = 0.1f;
    REQUIRE(p.x[0] == 1.5f);
    freeParticlesCPU(p);
    REQUIRE(p.x == nullptr);
    REQUIRE(p.n == 0);
}

TEST_CASE("initDisk places all particles in disk bounds", "[disk]") {
    Config cfg;
    cfg.n_particles        = 1000;
    cfg.disk_r_min         = 0.5f;
    cfg.disk_r_max         = 5.0f;
    cfg.disk_h_factor      = 0.05f;
    cfg.particle_mass_msun = 3.003e-7f;
    cfg.star_mass_msun     = 1.0f;

    ParticleData p = initDisk(cfg, 42);

    for (int i = 0; i < 1000; ++i) {
        float r = std::sqrt(p.x[i]*p.x[i] + p.z[i]*p.z[i]);
        REQUIRE(r >= cfg.disk_r_min * 0.99f);
        REQUIRE(r <= cfg.disk_r_max * 1.01f);
        REQUIRE_THAT(p.mass[i],
            Catch::Matchers::WithinRel(cfg.particle_mass_msun, 0.001f));
    }
    freeParticlesCPU(p);
}

TEST_CASE("initDisk velocities are approximately circular", "[disk]") {
    Config cfg;
    cfg.n_particles        = 100;
    cfg.disk_r_min         = 1.0f;
    cfg.disk_r_max         = 2.0f;
    cfg.disk_h_factor      = 0.01f;
    cfg.particle_mass_msun = 3.003e-7f;
    cfg.star_mass_msun     = 1.0f;

    ParticleData p = initDisk(cfg, 1);

    for (int i = 0; i < 100; ++i) {
        float r      = std::sqrt(p.x[i]*p.x[i] + p.z[i]*p.z[i]);
        float v_circ = std::sqrt(cfg.star_mass_msun / r);  // G=1
        float v      = std::sqrt(p.vx[i]*p.vx[i] + p.vy[i]*p.vy[i] + p.vz[i]*p.vz[i]);
        // Within 5% of circular speed
        REQUIRE_THAT(v, Catch::Matchers::WithinRel(v_circ, 0.05f));
    }
    freeParticlesCPU(p);
}
