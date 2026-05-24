#include <catch2/catch_test_macros.hpp>
#include "ParticleData.h"

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
