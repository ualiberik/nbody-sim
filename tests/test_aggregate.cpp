#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Aggregate.h"
#include "ParticleData.h"
#include "Config.h"
#include <cmath>

// Helper: build a zero-initialised ParticleData with CPU-allocated arrays
static ParticleData makeParticles(int n) {
    return allocateParticlesCPU(n);
}

static Config makeDefaultCfg() {
    Config cfg;
    cfg.particle_mass_msun      = 3.003e-7f;
    cfg.collision_radius_factor = 100.0f;
    return cfg;
}

TEST_CASE("detectAggregates: one cluster of 5 + 2 isolated singles", "[aggregate]") {
    // 5 tightly clustered particles near origin (separation ~0.0001 AU)
    // 2 isolated particles far away
    ParticleData p = makeParticles(7);

    // Cluster near (0,0,0)
    p.x[0] = 0.0000f; p.y[0] = 0.0000f; p.z[0] = 0.0000f;
    p.x[1] = 0.0001f; p.y[1] = 0.0000f; p.z[1] = 0.0000f;
    p.x[2] = 0.0000f; p.y[2] = 0.0001f; p.z[2] = 0.0000f;
    p.x[3] = 0.0001f; p.y[3] = 0.0001f; p.z[3] = 0.0000f;
    p.x[4] = 0.0000f; p.y[4] = 0.0000f; p.z[4] = 0.0001f;

    // Isolated singles
    p.x[5] = 3.0f;  p.y[5] = 0.0f; p.z[5] = 0.0f;
    p.x[6] = -4.0f; p.y[6] = 0.0f; p.z[6] = 0.0f;

    Config cfg = makeDefaultCfg();
    float link_length = 0.002f;  // well above cluster spacing, well below isolation distance

    auto aggs = detectAggregates(p, link_length, cfg);

    freeParticlesCPU(p);

    REQUIRE(aggs.size() == 1);
    REQUIRE(aggs[0].n_particles == 5);
}

TEST_CASE("detectAggregates: all particles isolated, returns empty result", "[aggregate]") {
    // 3 particles each 10 AU apart — no pair within link_length = 0.01 AU
    ParticleData p = makeParticles(3);
    p.x[0] = 0.0f;  p.y[0] = 0.0f; p.z[0] = 0.0f;
    p.x[1] = 10.0f; p.y[1] = 0.0f; p.z[1] = 0.0f;
    p.x[2] = 20.0f; p.y[2] = 0.0f; p.z[2] = 0.0f;

    Config cfg = makeDefaultCfg();
    float link_length = 0.01f;

    auto aggs = detectAggregates(p, link_length, cfg);

    freeParticlesCPU(p);

    REQUIRE(aggs.empty());
}

TEST_CASE("detectAggregates: two separate clusters of 3", "[aggregate]") {
    // Cluster A near (0,0,0), cluster B near (10,0,0)
    // Intra-cluster spacing ~0.001 AU, inter-cluster ~10 AU
    ParticleData p = makeParticles(6);

    p.x[0] = 0.000f; p.y[0] = 0.000f; p.z[0] = 0.000f;
    p.x[1] = 0.001f; p.y[1] = 0.000f; p.z[1] = 0.000f;
    p.x[2] = 0.000f; p.y[2] = 0.001f; p.z[2] = 0.000f;

    p.x[3] = 10.000f; p.y[3] = 0.000f; p.z[3] = 0.000f;
    p.x[4] = 10.001f; p.y[4] = 0.000f; p.z[4] = 0.000f;
    p.x[5] = 10.000f; p.y[5] = 0.001f; p.z[5] = 0.000f;

    Config cfg = makeDefaultCfg();
    float link_length = 0.01f;  // links intra-cluster pairs, not inter-cluster

    auto aggs = detectAggregates(p, link_length, cfg);

    freeParticlesCPU(p);

    REQUIRE(aggs.size() == 2);
    for (const auto& a : aggs) {
        REQUIRE(a.n_particles == 3);
    }
}

TEST_CASE("detectAggregates: mass and COM calculation for 2 particles", "[aggregate]") {
    // 2 particles at (1,0,0) and (3,0,0), equal mass
    // Expected COM: (2, 0, 0), mass = 2 * particle_mass_msun
    ParticleData p = makeParticles(2);
    p.x[0] = 1.0f; p.y[0] = 0.0f; p.z[0] = 0.0f;
    p.x[1] = 3.0f; p.y[1] = 0.0f; p.z[1] = 0.0f;
    p.vx[0] = 0.5f; p.vy[0] = 0.0f; p.vz[0] = 0.0f;
    p.vx[1] = 1.5f; p.vy[1] = 0.0f; p.vz[1] = 0.0f;

    Config cfg = makeDefaultCfg();
    float link_length = 5.0f;  // large enough to link the pair

    auto aggs = detectAggregates(p, link_length, cfg);

    freeParticlesCPU(p);

    REQUIRE(aggs.size() == 1);
    const Aggregate& a = aggs[0];
    REQUIRE(a.n_particles == 2);
    REQUIRE_THAT(a.cx, Catch::Matchers::WithinAbs(2.0f, 1e-5f));
    REQUIRE_THAT(a.cy, Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(a.cz, Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(a.mass_msun, Catch::Matchers::WithinRel(2.0f * 3.003e-7f, 1e-4f));
    // dist_center should be 2.0 AU
    REQUIRE_THAT(a.dist_center, Catch::Matchers::WithinAbs(2.0f, 1e-5f));
    // velocity COM: mean of (0.5, 0) and (1.5, 0) = (1.0, 0, 0)
    REQUIRE_THAT(a.vx, Catch::Matchers::WithinAbs(1.0f, 1e-5f));
    REQUIRE_THAT(a.vy, Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(a.vz, Catch::Matchers::WithinAbs(0.0f, 1e-5f));
}
