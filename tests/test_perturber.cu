#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>
#include "Perturber.h"
#include "ParticleData.h"
#include "Config.h"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

static constexpr float PI = 3.14159265358979f;

// ---------------------------------------------------------------------------
//  Analytic orbit
// ---------------------------------------------------------------------------
TEST_CASE("Perturber Keplerian orbit", "[perturber]") {
    Config cfg;
    cfg.perturber_enabled   = true;
    cfg.perturber_mass_mjup = 1.0f;
    cfg.perturber_radius_AU = 10.0f;
    cfg.perturber_phase0    = 0.0f;
    cfg.star_mass_msun      = 1.0f;

    Perturber p = makePerturber(cfg);

    SECTION("Initial position on +X axis") {
        REQUIRE(p.enabled);
        REQUIRE_THAT(p.x, WithinAbs(10.0f, 1e-5f));
        REQUIRE_THAT(p.z, WithinAbs(0.0f,  1e-5f));
        REQUIRE_THAT(p.y, WithinAbs(0.0f,  1e-5f));
    }

    SECTION("Mass conversion M_Jup -> M_sun") {
        REQUIRE_THAT(p.mass_msun, WithinRel(9.5479e-4f, 1e-4f));
    }

    SECTION("Angular speed matches Kepler") {
        // omega = sqrt(M/r^3) = sqrt(1 / 1000) ~= 0.031623
        REQUIRE_THAT(p.omega, WithinRel(std::sqrt(1.0f / 1000.0f), 1e-5f));
    }

    SECTION("Full period returns to start") {
        float T = 2.0f * PI / p.omega;
        updatePerturber(p, T);
        REQUIRE_THAT(p.x, WithinAbs(10.0f, 1e-2f));
        REQUIRE_THAT(p.z, WithinAbs(0.0f,  1e-2f));
    }

    SECTION("Half period flips position") {
        float T = 2.0f * PI / p.omega;
        updatePerturber(p, T * 0.5f);
        REQUIRE_THAT(p.x, WithinAbs(-10.0f, 1e-2f));
        REQUIRE_THAT(p.z, WithinAbs(  0.0f, 1e-2f));
    }

    SECTION("Quarter period rotates onto +Z axis") {
        float T = 2.0f * PI / p.omega;
        updatePerturber(p, T * 0.25f);
        REQUIRE_THAT(p.x, WithinAbs( 0.0f, 1e-2f));
        REQUIRE_THAT(p.z, WithinAbs(10.0f, 1e-2f));
    }
}

TEST_CASE("Perturber disabled stays inert", "[perturber]") {
    Config cfg;  // perturber_enabled defaults to false
    Perturber p = makePerturber(cfg);
    REQUIRE_FALSE(p.enabled);

    // updatePerturber on disabled perturber should be a no-op
    p.x = p.y = p.z = 42.0f;
    updatePerturber(p, 1000.0f);
    REQUIRE(p.x == 42.0f);
    REQUIRE(p.y == 42.0f);
    REQUIRE(p.z == 42.0f);
}

// ---------------------------------------------------------------------------
//  Force kernel: place a single test particle at origin, perturber at
//  (10, 0, 0) with mass m. The expected acceleration is +m/(r^2+s^2)^{3/2}
//  in the +x direction. Use this to validate the kernel.
// ---------------------------------------------------------------------------
TEST_CASE("Perturber force kernel applies +X pull on origin particle",
          "[perturber][cuda]") {
    ParticleData gpu = allocateParticlesGPU(1);

    float zero = 0.f;
    cudaMemcpy(gpu.x, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.y, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.z, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.ax, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.ay, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.az, &zero, sizeof(float), cudaMemcpyHostToDevice);

    float perturber_x = 10.0f, perturber_y = 0.0f, perturber_z = 0.0f;
    float pmass = 1.0e-3f;     // ~1 M_Jup
    float softening = 0.01f;

    launchPerturberForceKernel(gpu, perturber_x, perturber_y, perturber_z,
                                pmass, softening);

    float ax, ay, az;
    cudaMemcpy(&ax, gpu.ax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ay, gpu.ay, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&az, gpu.az, sizeof(float), cudaMemcpyDeviceToHost);

    float r2 = 10.0f*10.0f + softening*softening;
    float r  = std::sqrt(r2);
    float expected_ax = pmass / (r2 * r) * 10.0f;   // points toward +x

    REQUIRE_THAT(ax, WithinRel(expected_ax, 1e-4f));
    REQUIRE_THAT(ay, WithinAbs(0.0f, 1e-7f));
    REQUIRE_THAT(az, WithinAbs(0.0f, 1e-7f));

    freeParticlesGPU(gpu);
}

TEST_CASE("Perturber force kernel ADDS to existing acceleration",
          "[perturber][cuda]") {
    ParticleData gpu = allocateParticlesGPU(1);

    float zero = 0.f;
    float pre_ax = 1.0e-3f;   // small but representable when added to perturber pull
    cudaMemcpy(gpu.x,  &zero,   sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.y,  &zero,   sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.z,  &zero,   sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.ax, &pre_ax, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.ay, &zero,   sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.az, &zero,   sizeof(float), cudaMemcpyHostToDevice);

    // Perturber at moderate distance — contribution ~1e-5, well above float epsilon.
    float perturber_mass = 1.0e-3f;
    float perturber_r    = 10.0f;
    float softening      = 0.01f;
    launchPerturberForceKernel(gpu, perturber_r, 0.f, 0.f,
                                perturber_mass, softening);

    float ax;
    cudaMemcpy(&ax, gpu.ax, sizeof(float), cudaMemcpyDeviceToHost);

    float r2 = perturber_r * perturber_r + softening * softening;
    float r  = std::sqrt(r2);
    float expected_pulse = perturber_mass / (r2 * r) * perturber_r;
    float expected = pre_ax + expected_pulse;

    REQUIRE_THAT(ax, WithinRel(expected, 1e-4f));

    freeParticlesGPU(gpu);
}
