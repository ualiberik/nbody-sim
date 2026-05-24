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
