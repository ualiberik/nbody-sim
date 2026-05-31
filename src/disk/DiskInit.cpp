#include "DiskInit.h"
#include <cmath>
#include <random>
#include <stdexcept>

static constexpr float PI        = 3.14159265358979f;
static constexpr float AU_M      = 1.496e11f;   // 1 AU in metres
static constexpr float MSUN_KG   = 1.989e30f;   // 1 M_sun in kg

float particlePhysicalRadius(const Config& cfg) {
    float mass_kg = cfg.particle_mass_msun * MSUN_KG;
    float density = 3000.0f;  // kg/m^3, rocky material
    float vol     = mass_kg / density;
    float r_m     = std::cbrt(3.0f * vol / (4.0f * PI));
    return r_m / AU_M;  // convert to AU
}

ParticleData initDisk(const Config& cfg, unsigned int seed) {
    if (cfg.n_particles <= 0)
        throw std::invalid_argument("initDisk: n_particles must be positive");
    if (cfg.disk_r_min <= 0.0f)
        throw std::invalid_argument("initDisk: disk_r_min must be positive");
    if (cfg.disk_r_min >= cfg.disk_r_max)
        throw std::invalid_argument("initDisk: disk_r_min must be less than disk_r_max");

    ParticleData p = allocateParticlesCPU(cfg.n_particles);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float>       norm(0.0f, 1.0f);

    const float r_min     = cfg.disk_r_min;
    const float r_max     = cfg.disk_r_max;
    const float sqrt_rmin = std::sqrt(r_min);
    const float sqrt_rmax = std::sqrt(r_max);

    for (int i = 0; i < cfg.n_particles; ++i) {
        // Sample r from Sigma(r) ~ r^(-3/2)
        // P(r) dr ~ r^(-1/2) dr
        // CDF: (sqrt(r) - sqrt(r_min)) / (sqrt(r_max) - sqrt(r_min))
        // Inverse CDF: r = (u*(sqrt_rmax - sqrt_rmin) + sqrt_rmin)^2
        float u = uni(rng);
        float sr = u * (sqrt_rmax - sqrt_rmin) + sqrt_rmin;
        float r  = sr * sr;

        // Azimuthal angle (disk in XZ plane)
        float phi = uni(rng) * 2.0f * PI;

        // Position
        p.x[i] = r * std::cos(phi);
        p.z[i] = r * std::sin(phi);
        // Vertical Gaussian scatter: h(r) = disk_h_factor * r
        p.y[i] = norm(rng) * (cfg.disk_h_factor * r);

        // Circular speed: v_circ = sqrt(G * M_star / r), G = 1
        float v_circ     = std::sqrt(cfg.star_mass_msun / r);
        float noise_scale = 0.01f * v_circ;

        // Velocity perpendicular to r in XZ plane + small noise
        p.vx[i] = -v_circ * std::sin(phi) + norm(rng) * noise_scale;
        p.vz[i] =  v_circ * std::cos(phi) + norm(rng) * noise_scale;
        p.vy[i] =  norm(rng) * noise_scale;

        p.ax[i] = 0.0f; p.ay[i] = 0.0f; p.az[i] = 0.0f;
        p.mass[i] = cfg.particle_mass_msun;
        p.id[i]   = i;
    }
    return p;
}
