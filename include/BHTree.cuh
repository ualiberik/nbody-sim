#pragma once
#include "ParticleData.h"

static constexpr int MAX_NODES_MULT = 8;  // internal nodes = 8 * n_bodies

// Flat octree node arrays.
// Indices [0 .. n_bodies-1]        -> leaf nodes (one body per leaf slot)
// Indices [n_bodies .. n_total-1]  -> internal nodes
// Root = n_total - 1
//
// child[node*8 + k] == -1  -> empty child slot
// child[node*8 + k] >= 0   -> index of child node or leaf
struct OctreeData {
    float* pos_x;      // COM x (AU)
    float* pos_y;      // COM y
    float* pos_z;      // COM z
    float* mass;       // total mass of subtree (M_sun)
    int*   child;      // [n_total * 8]
    int*   count;      // bodies in subtree
    int*   start;      // DFS start index for sort
    int*   sort_idx;   // sorted body order [n_bodies]
    int*   mutex;      // per-node spin lock (reserved; build uses CAS-slot, not mutex)
    float* cell_size;  // half side-length of bounding cube for node

    float* bbox_min;   // [3]: global min x,y,z
    float* bbox_max;   // [3]: global max x,y,z

    int*   next_node;  // atomic counter for internal node allocation, init = n_bodies

    float* dv_x;      // per-body velocity delta for two-pass collision [n_bodies]
    float* dv_y;
    float* dv_z;

    int n_bodies;
    int n_total;       // n_bodies + MAX_NODES_MULT * n_bodies
};

OctreeData  allocateOctree(int n_bodies);
void        freeOctree(OctreeData& tree);
void        resetOctree(OctreeData& tree);  // zero all arrays, set child=-1

// Kernel launchers (all synchronize before returning)
void launchBBoxKernel      (OctreeData& t, const ParticleData& p);
void launchBuildKernel     (OctreeData& t, const ParticleData& p);
void launchSummarizeKernel (OctreeData& t);
void launchSortKernel      (OctreeData& t);
void launchForceKernel     (OctreeData& t, ParticleData& p,
                             float theta, float softening, float star_mass);
void launchCollisionKernel (OctreeData& t, ParticleData& p,
                             float r_coll, float restitution);
