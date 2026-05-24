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
