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
    REQUIRE_THAT(cfg.disk_h_factor, Catch::Matchers::WithinRel(0.05f, 0.001f));
    REQUIRE_THAT(cfg.softening_AU, Catch::Matchers::WithinRel(0.01f, 0.001f));
    REQUIRE_THAT(cfg.collision_radius_factor, Catch::Matchers::WithinRel(100.0f, 0.001f));
    REQUIRE_THAT(cfg.restitution, Catch::Matchers::WithinRel(0.3f, 0.001f));
    REQUIRE_THAT(cfg.total_time, Catch::Matchers::WithinRel(2000.0f, 0.001f));
}

TEST_CASE("loadConfig throws on missing file", "[config]") {
    REQUIRE_THROWS_AS(loadConfig("nonexistent/path.json"), std::runtime_error);
}
