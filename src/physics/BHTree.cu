#include "BHTree.cuh"
#include <cuda_runtime.h>
#include <float.h>

// All __global__ symbols prefixed "bhtree_" to prevent link collisions
// under CUDA_SEPARABLE_COMPILATION.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static int numBlocks(int n, int bs) { return (n + bs - 1) / bs; }

// Correct atomic float min/max for signed floats (IEEE 754 ordering).
// Standard atomicMin/Max on int bits only works for non-negative floats.
__device__ static void bhtree_atomicMinFloat(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, expected;
    do {
        expected = old;
        float cur = __int_as_float(old);
        float desired_f = (cur < val) ? cur : val;
        int desired = __float_as_int(desired_f);
        old = atomicCAS(iaddr, expected, desired);
    } while (old != expected);
}

__device__ static void bhtree_atomicMaxFloat(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, expected;
    do {
        expected = old;
        float cur = __int_as_float(old);
        float desired_f = (cur > val) ? cur : val;
        int desired = __float_as_int(desired_f);
        old = atomicCAS(iaddr, expected, desired);
    } while (old != expected);
}

// ---------------------------------------------------------------------------
// Task 8: Reset kernel
// ---------------------------------------------------------------------------
__global__ void bhtree_resetKernel(
    int* child, int* mutex, int* count, int* start,
    float* mass, float* pos_x, float* pos_y, float* pos_z,
    float* cell_size, float* bbox_min, float* bbox_max,
    int n_total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) {
        mutex[i]     = 0;
        count[i]     = 0;
        start[i]     = -1;
        mass[i]      = 0.f;
        pos_x[i]     = 0.f;
        pos_y[i]     = 0.f;
        pos_z[i]     = 0.f;
        cell_size[i] = 0.f;
        for (int k = 0; k < 8; ++k)
            child[i * 8 + k] = -1;
    }
    // Thread 0 initialises bbox
    if (i == 0) {
        // +FLT_MAX for min, -FLT_MAX for max
        // 0x7F7FFFFF = largest positive finite float  (+FLT_MAX)
        // 0xFF7FFFFF = largest negative finite float  (-FLT_MAX)
        float pos_inf = __int_as_float(0x7F7FFFFF);
        float neg_inf = __int_as_float(0xFF7FFFFF);
        bbox_min[0] = pos_inf; bbox_min[1] = pos_inf; bbox_min[2] = pos_inf;
        bbox_max[0] = neg_inf; bbox_max[1] = neg_inf; bbox_max[2] = neg_inf;
    }
}

// ---------------------------------------------------------------------------
// Task 8: Bounding-box kernel
// ---------------------------------------------------------------------------
__global__ void bhtree_bboxKernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    float* bbox_min, float* bbox_max,
    int n)
{
    extern __shared__ float sdata[];
    // Layout: sdata[0..bs-1]=minx, sdata[bs..2bs-1]=miny, sdata[2bs..3bs-1]=minz
    //         sdata[3bs..4bs-1]=maxx, sdata[4bs..5bs-1]=maxy, sdata[5bs..6bs-1]=maxz
    int bs = blockDim.x;
    float* s_minx = sdata;
    float* s_miny = sdata + bs;
    float* s_minz = sdata + 2 * bs;
    float* s_maxx = sdata + 3 * bs;
    float* s_maxy = sdata + 4 * bs;
    float* s_maxz = sdata + 5 * bs;

    float pos_inf = __int_as_float(0x7F7FFFFF);
    float neg_inf = __int_as_float(0xFF7FFFFF);

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    float lx = pos_inf, ly = pos_inf, lz = pos_inf;
    float hx = neg_inf, hy = neg_inf, hz = neg_inf;

    if (i < n) {
        lx = hx = px[i];
        ly = hy = py[i];
        lz = hz = pz[i];
    }

    s_minx[tid] = lx; s_miny[tid] = ly; s_minz[tid] = lz;
    s_maxx[tid] = hx; s_maxy[tid] = hy; s_maxz[tid] = hz;
    __syncthreads();

    // Standard reduction
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_minx[tid] = fminf(s_minx[tid], s_minx[tid + stride]);
            s_miny[tid] = fminf(s_miny[tid], s_miny[tid + stride]);
            s_minz[tid] = fminf(s_minz[tid], s_minz[tid + stride]);
            s_maxx[tid] = fmaxf(s_maxx[tid], s_maxx[tid + stride]);
            s_maxy[tid] = fmaxf(s_maxy[tid], s_maxy[tid + stride]);
            s_maxz[tid] = fmaxf(s_maxz[tid], s_maxz[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        bhtree_atomicMinFloat(&bbox_min[0], s_minx[0]);
        bhtree_atomicMinFloat(&bbox_min[1], s_miny[0]);
        bhtree_atomicMinFloat(&bbox_min[2], s_minz[0]);
        bhtree_atomicMaxFloat(&bbox_max[0], s_maxx[0]);
        bhtree_atomicMaxFloat(&bbox_max[1], s_maxy[0]);
        bhtree_atomicMaxFloat(&bbox_max[2], s_maxz[0]);
    }
}

// ---------------------------------------------------------------------------
// Task 9: Leaf-count init kernel
// ---------------------------------------------------------------------------
__global__ void bhtree_setLeafCountKernel(int* count, int n_bodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_bodies) count[i] = 1;
}

// ---------------------------------------------------------------------------
// Task 9: Copy particle positions/masses into leaf slots
// ---------------------------------------------------------------------------
__global__ void bhtree_copyLeafDataKernel(
    float* tree_pos_x, float* tree_pos_y, float* tree_pos_z, float* tree_mass,
    const float* p_x, const float* p_y, const float* p_z, const float* p_mass,
    int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_bodies) {
        tree_pos_x[i] = p_x[i];
        tree_pos_y[i] = p_y[i];
        tree_pos_z[i] = p_z[i];
        tree_mass[i]  = p_mass[i];
    }
}

// ---------------------------------------------------------------------------
// Task 9: Octant helper
// ---------------------------------------------------------------------------
__device__ static int bhtree_getOctant(float cx, float cy, float cz,
                                        float px, float py, float pz)
{
    return ((px > cx) ? 1 : 0) |
           ((py > cy) ? 2 : 0) |
           ((pz > cz) ? 4 : 0);
}

// ---------------------------------------------------------------------------
// Task 9: Build kernel (Burtscher & Pingali style with CAS-based insertion)
// ---------------------------------------------------------------------------
__global__ void bhtree_buildKernel(
    int* child, float* pos_x, float* pos_y, float* pos_z,
    float* cell_size, int* mutex, int* next_node,
    const float* bmin, const float* bmax,
    int n_bodies, int n_total)
{
    int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= n_bodies) return;

    float px = pos_x[body], py = pos_y[body], pz = pos_z[body];

    // Root properties — same for all threads
    float rcx = (bmin[0] + bmax[0]) * 0.5f;
    float rcy = (bmin[1] + bmax[1]) * 0.5f;
    float rcz = (bmin[2] + bmax[2]) * 0.5f;
    float rhs = fmaxf(fmaxf(bmax[0] - bmin[0], bmax[1] - bmin[1]),
                      bmax[2] - bmin[2]) * 0.5f * 1.001f;

    int root = n_total - 1;
    // Root geometry is initialized by launchBuildKernel (CPU-side) before
    // this kernel is launched, so no cross-block race here.

    // Insertion loop
    int node = root;
    float ncx = rcx, ncy = rcy, ncz = rcz, nhs = rhs;
    int max_iter = 128;

    while (max_iter-- > 0) {
        int oct  = bhtree_getOctant(ncx, ncy, ncz, px, py, pz);
        int* slot = &child[node * 8 + oct];

        int old = atomicCAS(slot, -1, body);

        if (old == -1) {
            // Claimed an empty slot — we're done
            __threadfence();
            return;
        }

        if (old == body) {
            // Should not happen, but guard against self-collision
            return;
        }

        if (old == -2) {
            // Slot locked by another thread creating an internal node — retry
            node = root; ncx = rcx; ncy = rcy; ncz = rcz; nhs = rhs;
            max_iter = 128;
            continue;
        }

        if (old >= n_bodies) {
            // old is an internal node — descend into it
            node = old;
            ncx = pos_x[node]; ncy = pos_y[node]; ncz = pos_z[node];
            nhs = cell_size[node];
            continue;
        }

        // old is another body (leaf) — need to subdivide.
        // Lock the slot with CAS(-2).
        int locked = atomicCAS(slot, old, -2);
        if (locked != old) {
            // Someone else grabbed it first; back off to root
            node = root; ncx = rcx; ncy = rcy; ncz = rcz; nhs = rhs;
            max_iter = 128;
            continue;
        }

        // We hold the lock. Allocate a new internal node.
        int new_node = atomicAdd(next_node, 1);
        if (new_node >= n_total) {
            // Overflow — release lock and bail
            atomicExch(slot, old);
            return;
        }

        // Compute new node geometry
        float child_hs = nhs * 0.5f;
        float new_cx   = ncx + ((px > ncx) ? child_hs : -child_hs);
        float new_cy   = ncy + ((py > ncy) ? child_hs : -child_hs);
        float new_cz   = ncz + ((pz > ncz) ? child_hs : -child_hs);

        pos_x[new_node]     = new_cx;
        pos_y[new_node]     = new_cy;
        pos_z[new_node]     = new_cz;
        cell_size[new_node] = child_hs;

        // Place the displaced body in the new node
        int disp_oct = bhtree_getOctant(new_cx, new_cy, new_cz,
                                         pos_x[old], pos_y[old], pos_z[old]);
        child[new_node * 8 + disp_oct] = old;

        // Publish the new internal node (unlock)
        __threadfence();
        atomicExch(slot, new_node);

        // Now try to insert our own body into new_node
        node = new_node;
        ncx = new_cx; ncy = new_cy; ncz = new_cz; nhs = child_hs;
    }
    // max_iter exhausted — body may be lost (should not happen with correct bbox)
}

// ---------------------------------------------------------------------------
// Task 9: Summarize kernel — propagate COM bottom-up, multi-pass safe
// ---------------------------------------------------------------------------
__global__ void bhtree_summarizeKernel(
    float* pos_x, float* pos_y, float* pos_z, float* mass,
    const int* child, int* count, int n_bodies, int n_total)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x + n_bodies;
    if (node >= n_total) return;

    // Skip nodes that are already summarized
    if (count[node] > 0) return;

    float m = 0.f, cx = 0.f, cy = 0.f, cz = 0.f;
    int   cnt = 0;

    for (int k = 0; k < 8; ++k) {
        int c = child[node * 8 + k];
        if (c < 0) continue;
        // Check if child is ready (count > 0)
        int c_count = count[c];
        if (c_count == 0) return;   // not ready yet — multi-pass handles it
        __threadfence();
        float cm = mass[c];
        m  += cm;
        cx += pos_x[c] * cm;
        cy += pos_y[c] * cm;
        cz += pos_z[c] * cm;
        cnt += c_count;
    }

    if (m > 0.f) {
        pos_x[node] = cx / m;
        pos_y[node] = cy / m;
        pos_z[node] = cz / m;
        mass[node]  = m;
        count[node] = cnt;
    }
}

// ---------------------------------------------------------------------------
// Task 9: Sort kernel — single-thread iterative DFS
// ---------------------------------------------------------------------------
__global__ void bhtree_sortKernel(
    int* sort_idx, int* start, const int* child,
    const int* count, int n_bodies, int root)
{
    // Single-threaded DFS with an explicit stack
    int stack[2048];
    int top = 0;
    stack[top++] = root;
    int out = 0;

    while (top > 0) {
        int node = stack[--top];
        if (node < 0) continue;

        if (node < n_bodies) {
            // Leaf body — record in sorted output
            sort_idx[out++] = node;
            continue;
        }

        start[node] = out;
        // Push children in reverse order so leftmost octant is processed first
        for (int k = 7; k >= 0; --k) {
            int c = child[node * 8 + k];
            if (c >= 0) {
                if (top < 2047)
                    stack[top++] = c;
            }
        }
    }
}

// ===========================================================================
// Host-side launchers
// ===========================================================================

OctreeData allocateOctree(int n) {
    OctreeData t;
    t.n_bodies = n;
    t.n_total  = n + MAX_NODES_MULT * n;
    int nt = t.n_total;

    nbody_check_cuda(cudaMalloc(&t.pos_x,     nt * sizeof(float)),   "allocOctree pos_x");
    nbody_check_cuda(cudaMalloc(&t.pos_y,     nt * sizeof(float)),   "allocOctree pos_y");
    nbody_check_cuda(cudaMalloc(&t.pos_z,     nt * sizeof(float)),   "allocOctree pos_z");
    nbody_check_cuda(cudaMalloc(&t.mass,      nt * sizeof(float)),   "allocOctree mass");
    nbody_check_cuda(cudaMalloc(&t.child,     nt * 8 * sizeof(int)), "allocOctree child");
    nbody_check_cuda(cudaMalloc(&t.count,     nt * sizeof(int)),     "allocOctree count");
    nbody_check_cuda(cudaMalloc(&t.start,     nt * sizeof(int)),     "allocOctree start");
    nbody_check_cuda(cudaMalloc(&t.sort_idx,  n  * sizeof(int)),     "allocOctree sort_idx");
    nbody_check_cuda(cudaMalloc(&t.mutex,     nt * sizeof(int)),     "allocOctree mutex");
    nbody_check_cuda(cudaMalloc(&t.cell_size, nt * sizeof(float)),   "allocOctree cell_size");
    nbody_check_cuda(cudaMalloc(&t.bbox_min,  3  * sizeof(float)),   "allocOctree bbox_min");
    nbody_check_cuda(cudaMalloc(&t.bbox_max,  3  * sizeof(float)),   "allocOctree bbox_max");
    nbody_check_cuda(cudaMalloc(&t.next_node, sizeof(int)),          "allocOctree next_node");
    return t;
}

void freeOctree(OctreeData& t) {
    cudaFree(t.pos_x);  cudaFree(t.pos_y);  cudaFree(t.pos_z);
    cudaFree(t.mass);   cudaFree(t.child);  cudaFree(t.count);
    cudaFree(t.start);  cudaFree(t.sort_idx); cudaFree(t.mutex);
    cudaFree(t.cell_size); cudaFree(t.bbox_min); cudaFree(t.bbox_max);
    cudaFree(t.next_node);
    t.pos_x = t.pos_y = t.pos_z = nullptr;
    t.mass  = t.cell_size = t.bbox_min = t.bbox_max = nullptr;
    t.child = t.count = t.start = t.sort_idx = t.mutex = t.next_node = nullptr;
    t.n_bodies = t.n_total = 0;
}

void resetOctree(OctreeData& tree) {
    int nt = tree.n_total;
    int bs = 256;
    int nb = (nt + bs - 1) / bs;

    bhtree_resetKernel<<<nb, bs>>>(
        tree.child, tree.mutex, tree.count, tree.start,
        tree.mass, tree.pos_x, tree.pos_y, tree.pos_z,
        tree.cell_size, tree.bbox_min, tree.bbox_max,
        nt);
    nbody_check_cuda(cudaGetLastError(),       "bhtree_resetKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(),  "bhtree_resetKernel sync");

    // Initialize next_node counter to n_bodies (first available internal node)
    nbody_check_cuda(cudaMemcpy(tree.next_node, &tree.n_bodies,
                                sizeof(int), cudaMemcpyHostToDevice),
                     "resetOctree next_node init");
}

void launchBBoxKernel(OctreeData& t, const ParticleData& p) {
    int n  = p.n;
    int bs = 256;
    int nb = (n + bs - 1) / bs;

    // Shared memory: 6 arrays of bs floats (minx,miny,minz,maxx,maxy,maxz)
    size_t smem = 6 * bs * sizeof(float);
    bhtree_bboxKernel<<<nb, bs, smem>>>(p.x, p.y, p.z, t.bbox_min, t.bbox_max, n);
    nbody_check_cuda(cudaGetLastError(),      "bhtree_bboxKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_bboxKernel sync");

    // Apply 1% padding CPU-side
    float bmin[3], bmax[3];
    nbody_check_cuda(cudaMemcpy(bmin, t.bbox_min, 3 * sizeof(float), cudaMemcpyDeviceToHost),
                     "launchBBoxKernel download min");
    nbody_check_cuda(cudaMemcpy(bmax, t.bbox_max, 3 * sizeof(float), cudaMemcpyDeviceToHost),
                     "launchBBoxKernel download max");

    // Expand each side by 0.5% so the cube strictly contains all particles
    for (int i = 0; i < 3; ++i) {
        float mid  = (bmin[i] + bmax[i]) * 0.5f;
        float half = (bmax[i] - bmin[i]) * 0.5f * 1.001f;
        bmin[i] = mid - half;
        bmax[i] = mid + half;
    }

    nbody_check_cuda(cudaMemcpy(t.bbox_min, bmin, 3 * sizeof(float), cudaMemcpyHostToDevice),
                     "launchBBoxKernel upload min");
    nbody_check_cuda(cudaMemcpy(t.bbox_max, bmax, 3 * sizeof(float), cudaMemcpyHostToDevice),
                     "launchBBoxKernel upload max");
}

void launchBuildKernel(OctreeData& t, const ParticleData& p) {
    int n  = t.n_bodies;
    int bs = 256;
    int nb = (n + bs - 1) / bs;

    // Copy particle positions/masses into leaf slots
    bhtree_copyLeafDataKernel<<<nb, bs>>>(
        t.pos_x, t.pos_y, t.pos_z, t.mass,
        p.x, p.y, p.z, p.mass, n);
    nbody_check_cuda(cudaGetLastError(),      "bhtree_copyLeafDataKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_copyLeafDataKernel sync");

    // Set leaf counts to 1
    bhtree_setLeafCountKernel<<<nb, bs>>>(t.count, n);
    nbody_check_cuda(cudaGetLastError(),      "bhtree_setLeafCountKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_setLeafCountKernel sync");

    // Initialize root geometry CPU-side before the build kernel launches.
    // __threadfence() only orders memory within a single kernel launch — it
    // cannot synchronize across blocks. Doing this here (on the host, after
    // cudaDeviceSynchronize) guarantees all threads see consistent root data.
    {
        float bmin[3], bmax[3];
        nbody_check_cuda(cudaMemcpy(bmin, t.bbox_min, 3*sizeof(float), cudaMemcpyDeviceToHost),
                         "launchBuildKernel download bbox_min");
        nbody_check_cuda(cudaMemcpy(bmax, t.bbox_max, 3*sizeof(float), cudaMemcpyDeviceToHost),
                         "launchBuildKernel download bbox_max");
        float rcx = (bmin[0]+bmax[0]) * 0.5f;
        float rcy = (bmin[1]+bmax[1]) * 0.5f;
        float rcz = (bmin[2]+bmax[2]) * 0.5f;
        float rhs = fmaxf(fmaxf(bmax[0]-bmin[0], bmax[1]-bmin[1]),
                               bmax[2]-bmin[2]) * 0.5f * 1.001f;
        int root = t.n_total - 1;
        nbody_check_cuda(cudaMemcpy(t.pos_x     + root, &rcx, sizeof(float), cudaMemcpyHostToDevice), "root pos_x");
        nbody_check_cuda(cudaMemcpy(t.pos_y     + root, &rcy, sizeof(float), cudaMemcpyHostToDevice), "root pos_y");
        nbody_check_cuda(cudaMemcpy(t.pos_z     + root, &rcz, sizeof(float), cudaMemcpyHostToDevice), "root pos_z");
        nbody_check_cuda(cudaMemcpy(t.cell_size + root, &rhs, sizeof(float), cudaMemcpyHostToDevice), "root cell_size");
    }

    // Build the tree
    bhtree_buildKernel<<<nb, bs>>>(
        t.child, t.pos_x, t.pos_y, t.pos_z, t.cell_size,
        t.mutex, t.next_node, t.bbox_min, t.bbox_max, n, t.n_total);
    nbody_check_cuda(cudaGetLastError(),      "bhtree_buildKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_buildKernel sync");
}

void launchSummarizeKernel(OctreeData& t) {
    int internal = t.n_total - t.n_bodies;
    int bs = 256;
    int nb = (internal + bs - 1) / bs;

    // Multi-pass: 20 passes ensures convergence for trees up to depth ~20.
    // Each pass propagates COM one level up from the bottom.
    // This avoids the deadlock risk of a spin-wait in a single pass.
    for (int i = 0; i < 20; ++i) {
        bhtree_summarizeKernel<<<nb, bs>>>(
            t.pos_x, t.pos_y, t.pos_z, t.mass,
            t.child, t.count, t.n_bodies, t.n_total);
        nbody_check_cuda(cudaGetLastError(),      "bhtree_summarizeKernel launch");
        nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_summarizeKernel sync");
    }
}

void launchSortKernel(OctreeData& t) {
    int root = t.n_total - 1;
    // Single-thread kernel
    bhtree_sortKernel<<<1, 1>>>(
        t.sort_idx, t.start, t.child, t.count, t.n_bodies, root);
    nbody_check_cuda(cudaGetLastError(),      "bhtree_sortKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_sortKernel sync");
}

// ---------------------------------------------------------------------------
// Task 10: Force kernel — iterative Barnes-Hut traversal with per-thread stack
// ---------------------------------------------------------------------------
__global__ void bhtree_forceKernel(
    const float* tree_x, const float* tree_y, const float* tree_z,
    const float* tree_mass, const float* cell_size,
    const int* child, const int* sort_idx,
    float* ax, float* ay, float* az,
    const float* px, const float* py, const float* pz,
    float theta, float softening, float star_mass,
    int n_bodies, int n_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;
    int body = sort_idx[idx];  // use DFS-sorted order for cache locality

    float bx = px[body], by = py[body], bz = pz[body];
    float fx = 0.f, fy = 0.f, fz = 0.f;

    // Force from fixed star at origin
    {
        float dx = -bx, dy = -by, dz = -bz;
        float r2 = dx*dx + dy*dy + dz*dz + softening*softening;
        float r  = sqrtf(r2);
        float f  = star_mass / (r2 * r);
        fx += f*dx; fy += f*dy; fz += f*dz;
    }

    // BH tree traversal
    int stack[64];
    int top = 0;
    stack[top++] = n_total - 1;  // root

    while (top > 0) {
        int node = stack[--top];
        if (node < 0) continue;

        float dx = tree_x[node] - bx;
        float dy = tree_y[node] - by;
        float dz = tree_z[node] - bz;
        float r2 = dx*dx + dy*dy + dz*dz + softening*softening;
        float r  = sqrtf(r2);

        bool is_leaf = (node < n_bodies);
        bool accept  = is_leaf || (cell_size[node] / r < theta);

        if (accept) {
            if (node == body) continue;  // skip self
            float m  = tree_mass[node];
            float f  = m / (r2 * r);
            fx += f*dx; fy += f*dy; fz += f*dz;
        } else {
            // Push children onto stack
            for (int k = 0; k < 8; ++k) {
                int c = child[node*8+k];
                if (c >= 0) {
                    if (top < 63) stack[top++] = c;
                }
            }
        }
    }

    ax[body] = fx;
    ay[body] = fy;
    az[body] = fz;
}

void launchForceKernel(OctreeData& t, ParticleData& p,
                        float theta, float softening, float star_mass) {
    int bs = 256, nb = (p.n + bs - 1) / bs;
    bhtree_forceKernel<<<nb, bs>>>(
        t.pos_x, t.pos_y, t.pos_z, t.mass, t.cell_size, t.child, t.sort_idx,
        p.ax, p.ay, p.az, p.x, p.y, p.z,
        theta, softening, star_mass, t.n_bodies, t.n_total);
    nbody_check_cuda(cudaGetLastError(),      "bhtree_forceKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_forceKernel sync");
}

// ---------------------------------------------------------------------------
// Task 10: Collision kernel — tree-assisted near-body search + inelastic impulse
//
// Two-pass design to avoid race conditions:
//   Pass 1: each thread reads original velocities (vx_in) and accumulates
//           velocity deltas into dvx via atomicAdd.
//   Pass 2: a separate kernel adds dvx -> vx.
// This ensures all threads see the same pre-collision velocities.
// ---------------------------------------------------------------------------
__global__ void bhtree_collisionKernel(
    const float* tree_x, const float* tree_y, const float* tree_z,
    const float* cell_size, const int* child, const int* sort_idx,
    const float* vx_in, const float* vy_in, const float* vz_in,
    float* dvx, float* dvy, float* dvz,
    const float* px, const float* py, const float* pz,
    float r_coll, float e, int n_bodies, int n_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bodies) return;
    int i = sort_idx[idx];

    float ix=px[i], iy=py[i], iz=pz[i];
    float vi_x=vx_in[i], vi_y=vy_in[i], vi_z=vz_in[i];

    int stack[64]; int top = 0;
    stack[top++] = n_total - 1;

    while (top > 0) {
        int node = stack[--top];
        if (node < 0) continue;

        float dx = tree_x[node]-ix, dy = tree_y[node]-iy, dz = tree_z[node]-iz;
        float dist2 = dx*dx + dy*dy + dz*dz;

        bool is_leaf = (node < n_bodies);

        if (is_leaf) {
            int j = node;
            if (j == i) continue;
            if (dist2 < r_coll * r_coll) {
                float dist = sqrtf(dist2) + 1e-10f;
                float nx = dx/dist, ny = dy/dist, nz = dz/dist;
                // Relative velocity along normal (j relative to i), using original velocities
                float dvn = (vx_in[j]-vi_x)*nx + (vy_in[j]-vi_y)*ny + (vz_in[j]-vi_z)*nz;
                if (dvn < 0.f) {
                    float J = -(1.f + e) * dvn * 0.5f;  // impulse per unit mass
                    atomicAdd(&dvx[i], -J*nx);
                    atomicAdd(&dvy[i], -J*ny);
                    atomicAdd(&dvz[i], -J*nz);
                }
            }
        } else {
            float cell_half = cell_size[node];
            float cell_dist = sqrtf(dist2) - cell_half * 1.732f;  // sqrt(3)*half = max extent
            if (cell_dist < r_coll) {
                for (int k = 0; k < 8; ++k) {
                    int c = child[node*8+k];
                    if (c >= 0 && top < 63) stack[top++] = c;
                }
            }
        }
    }
}

// Apply accumulated velocity deltas
__global__ void bhtree_applyCollisionDeltaKernel(
    float* vx, float* vy, float* vz,
    const float* dvx, const float* dvy, const float* dvz,
    int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;
    vx[i] += dvx[i];
    vy[i] += dvy[i];
    vz[i] += dvz[i];
}

void launchCollisionKernel(OctreeData& t, ParticleData& p,
                            float r_coll, float restitution) {
    int n = p.n;
    int bs = 256, nb = (n + bs - 1) / bs;

    // Allocate temporary delta arrays, initialized to zero
    float *dvx, *dvy, *dvz;
    nbody_check_cuda(cudaMalloc(&dvx, n * sizeof(float)), "collisionKernel dvx alloc");
    nbody_check_cuda(cudaMalloc(&dvy, n * sizeof(float)), "collisionKernel dvy alloc");
    nbody_check_cuda(cudaMalloc(&dvz, n * sizeof(float)), "collisionKernel dvz alloc");
    nbody_check_cuda(cudaMemset(dvx, 0, n * sizeof(float)), "collisionKernel dvx zero");
    nbody_check_cuda(cudaMemset(dvy, 0, n * sizeof(float)), "collisionKernel dvy zero");
    nbody_check_cuda(cudaMemset(dvz, 0, n * sizeof(float)), "collisionKernel dvz zero");

    bhtree_collisionKernel<<<nb, bs>>>(
        t.pos_x, t.pos_y, t.pos_z, t.cell_size, t.child, t.sort_idx,
        p.vx, p.vy, p.vz, dvx, dvy, dvz, p.x, p.y, p.z,
        r_coll, restitution, t.n_bodies, t.n_total);
    nbody_check_cuda(cudaGetLastError(),      "bhtree_collisionKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_collisionKernel sync");

    bhtree_applyCollisionDeltaKernel<<<nb, bs>>>(p.vx, p.vy, p.vz, dvx, dvy, dvz, n);
    nbody_check_cuda(cudaGetLastError(),      "bhtree_applyCollisionDeltaKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "bhtree_applyCollisionDeltaKernel sync");

    cudaFree(dvx); cudaFree(dvy); cudaFree(dvz);
}
