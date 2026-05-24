#pragma once
#include <vector>

struct Aggregate {
    int   id;
    int   n_particles;
    float mass_msun;
    float cx, cy, cz;        // center of mass (AU)
    float vx, vy, vz;        // mass-weighted velocity (AU/T0)
    float dist_center;       // distance from origin (AU)
    std::vector<int> particle_ids;
};

// Implementation in src/analysis/Aggregate.cpp
#include "ParticleData.h"
#include "Config.h"
std::vector<Aggregate> detectAggregates(
    const ParticleData& p, float link_length, const Config& cfg);
