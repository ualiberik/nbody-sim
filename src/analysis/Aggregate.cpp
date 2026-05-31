#include "Aggregate.h"
#include "ParticleData.h"
#include "Config.h"
#include <cmath>
#include <numeric>
#include <unordered_map>

// Union-Find with path compression and union by rank
static int find(std::vector<int>& parent, int i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];  // path halving
        i = parent[i];
    }
    return i;
}

static void unite(std::vector<int>& parent, std::vector<int>& rank, int a, int b) {
    a = find(parent, a);
    b = find(parent, b);
    if (a == b) return;
    if (rank[a] < rank[b]) std::swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b]) ++rank[a];
}

std::vector<Aggregate> detectAggregates(
    const ParticleData& p, float link_length, const Config& cfg)
{
    int n = p.n;
    float ll2 = link_length * link_length;

    std::vector<int> parent(n), rank(n, 0);
    std::iota(parent.begin(), parent.end(), 0);

    // O(N²) FOF — only called at output intervals, not in tight physics loop
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float dx = p.x[i]-p.x[j], dy = p.y[i]-p.y[j], dz = p.z[i]-p.z[j];
            if (dx*dx + dy*dy + dz*dz < ll2)
                unite(parent, rank, i, j);
        }
    }

    // Gather groups by root
    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i)
        groups[find(parent, i)].push_back(i);

    // Build Aggregate objects — skip single-particle groups
    std::vector<Aggregate> result;
    int agg_id = 0;
    for (auto& [root, members] : groups) {
        if (static_cast<int>(members.size()) < 2) continue;

        Aggregate a;
        a.id = agg_id++;
        a.n_particles = static_cast<int>(members.size());
        a.particle_ids = members;
        a.mass_msun = a.n_particles * cfg.particle_mass_msun;

        // Center of mass and velocity (equal masses → simple mean)
        a.cx = a.cy = a.cz = 0.0f;
        a.vx = a.vy = a.vz = 0.0f;
        for (int idx : members) {
            a.cx += p.x[idx]; a.cy += p.y[idx]; a.cz += p.z[idx];
            a.vx += p.vx[idx]; a.vy += p.vy[idx]; a.vz += p.vz[idx];
        }
        float inv = 1.0f / static_cast<float>(a.n_particles);
        a.cx *= inv; a.cy *= inv; a.cz *= inv;
        a.vx *= inv; a.vy *= inv; a.vz *= inv;

        a.dist_center = std::sqrt(a.cx*a.cx + a.cy*a.cy + a.cz*a.cz);
        result.push_back(std::move(a));
    }
    return result;
}
