#pragma once
#include "ParticleData.h"
#include "Config.h"
#include <cmath>

// ---------------------------------------------------------------------------
//  Perturber — a single massive body on a *fixed* circular Keplerian orbit.
//
//  Used to study how an outer gas giant affects in-disk planet formation.
//  It exerts gravity on every particle but is itself NOT integrated; its
//  position is recomputed analytically from sim_time each step.
//
//  - Excluded from the Barnes-Hut tree (no influence on collision search).
//  - Excluded from FOF aggregate detection.
//  - Recorded in stats.csv as a special row with agg_id = -1.
//
//  Units:
//      mass    : M_sun  (config gives M_Jupiter, converted via MJUP_TO_MSUN)
//      radius  : AU
//      phase   : radians, measured from +x axis in the XZ plane
//      time    : T0 (sim_time)
// ---------------------------------------------------------------------------

// 1 Jupiter mass = 9.5479e-4 solar masses
constexpr float MJUP_TO_MSUN = 9.5479e-4f;

struct Perturber {
    bool  enabled    = false;
    float r          = 0.f;     // orbital radius (AU)
    float mass_msun  = 0.f;     // mass (M_sun)
    float phase0     = 0.f;     // phase at t=0 (rad)
    float omega      = 0.f;     // angular speed (rad / T0)

    // Position at the latest update_at_time() call.
    float x = 0.f, y = 0.f, z = 0.f;
};

// Construct from config and initialise position at sim_time = 0.
inline Perturber makePerturber(const Config& cfg) {
    Perturber p;
    p.enabled   = cfg.perturber_enabled;
    if (!p.enabled) return p;

    p.r         = cfg.perturber_radius_AU;
    p.mass_msun = cfg.perturber_mass_mjup * MJUP_TO_MSUN;
    p.phase0    = cfg.perturber_phase0;

    // Kepler's third law in G=1, M_star=star_mass units:
    //   omega = sqrt(M_star / r^3)
    p.omega = std::sqrt(cfg.star_mass_msun / (p.r * p.r * p.r));

    // Position at t=0
    p.x = p.r * std::cos(p.phase0);
    p.z = p.r * std::sin(p.phase0);
    p.y = 0.f;
    return p;
}

// Update analytic position to the given sim_time (T0).
inline void updatePerturber(Perturber& p, float sim_time) {
    if (!p.enabled) return;
    float phi = p.phase0 + p.omega * sim_time;
    p.x = p.r * std::cos(phi);
    p.z = p.r * std::sin(phi);
    p.y = 0.f;
}

#ifndef PARTICLE_DATA_NO_CUDA
// Kernel launcher (defined in src/physics/Perturber.cu).
// Adds gravitational acceleration from the perturber to every particle's
// (ax, ay, az) — uses += so it must be called AFTER the main force kernel
// (which writes), and BEFORE the second half-kick.
void launchPerturberForceKernel(ParticleData& gpu,
                                 float px, float py, float pz,
                                 float pmass, float softening);
#endif
