#include "Config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

Config loadConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config: " + path);
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parse error in '" + path + "': " + e.what());
    }

    Config cfg;
    cfg.n_particles            = j.value("n_particles", cfg.n_particles);
    cfg.disk_r_min             = j.value("disk_r_min", cfg.disk_r_min);
    cfg.disk_r_max             = j.value("disk_r_max", cfg.disk_r_max);
    cfg.disk_h_factor          = j.value("disk_h_factor", cfg.disk_h_factor);
    cfg.particle_mass_msun     = j.value("particle_mass_msun", cfg.particle_mass_msun);
    cfg.star_mass_msun         = j.value("star_mass_msun", cfg.star_mass_msun);
    cfg.softening_AU           = j.value("softening_AU", cfg.softening_AU);
    cfg.theta                  = j.value("theta", cfg.theta);
    cfg.collision_radius_factor= j.value("collision_radius_factor", cfg.collision_radius_factor);
    cfg.restitution            = j.value("restitution", cfg.restitution);
    cfg.dt                     = j.value("dt", cfg.dt);
    cfg.total_time             = j.value("total_time", cfg.total_time);
    cfg.output_every_n_steps   = j.value("output_every_n_steps", cfg.output_every_n_steps);
    cfg.output_dir             = j.value("output_dir", cfg.output_dir);

    if (cfg.n_particles <= 0)
        throw std::runtime_error("n_particles must be positive, got: " + std::to_string(cfg.n_particles));
    if (cfg.dt <= 0.f)
        throw std::runtime_error("dt must be positive");
    if (cfg.disk_r_min >= cfg.disk_r_max)
        throw std::runtime_error("disk_r_min must be less than disk_r_max");

    return cfg;
}
