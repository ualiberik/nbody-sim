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
    // Gas drag rate (1/T0). Damps deviation from local circular Keplerian
    // orbit (and all vertical motion) with decay exp(-gas_drag_rate * dt).
    // 0.0 = disabled. Typical values: 0.001–0.05 (gentle to strong).
    float gas_drag_rate        = 0.0f;
    float dt                   = 0.01f;
    float total_time           = 2000.0f;
    int   output_every_n_steps = 200;
    std::string output_dir     = "data";

    // ------------------------------------------------------------------
    // Optional outer perturber: a single gas-giant body on a fixed
    // circular Keplerian orbit in the XZ plane around the star.
    // It exerts gravity on every particle but is itself NOT integrated:
    //   - its position is recomputed analytically each step;
    //   - it does not participate in collisions or FOF aggregates;
    //   - it is recorded in stats.csv as a special row with agg_id = -1.
    // ------------------------------------------------------------------
    bool  perturber_enabled    = false;
    float perturber_mass_mjup  = 1.0f;     // mass in Jupiter masses
    float perturber_radius_AU  = 20.0f;    // orbital (semi-major) radius
    float perturber_phase0     = 0.0f;     // initial phase angle, radians
};

Config loadConfig(const std::string& path);
