#include "Perturber.h"
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
//  perturber_forceKernel
//      ai += G * m_p * (r_p - r_i) / |r_p - r_i + softening|^3
//
//  Note: += (NOT =) — we add to the acceleration already computed by the
//  Barnes-Hut force kernel for star + particle-particle gravity.
//
//  In G=1 units, "G" is omitted.
// ---------------------------------------------------------------------------
__global__ void perturber_forceKernel(
    float* ax, float* ay, float* az,
    const float* px, const float* py, const float* pz,
    float perturber_x, float perturber_y, float perturber_z,
    float perturber_mass, float softening, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float dx = perturber_x - px[i];
    float dy = perturber_y - py[i];
    float dz = perturber_z - pz[i];
    // Use the same softening as for the rest of the gravity solver so the
    // perturber does not exhibit a singularity when a particle wanders very
    // close to it.
    float r2 = dx*dx + dy*dy + dz*dz + softening*softening;
    float r  = sqrtf(r2);
    float f  = perturber_mass / (r2 * r);

    ax[i] += f * dx;
    ay[i] += f * dy;
    az[i] += f * dz;
}

static int numBlocks(int n, int bs) { return (n + bs - 1) / bs; }

void launchPerturberForceKernel(ParticleData& g,
                                 float px, float py, float pz,
                                 float pmass, float softening)
{
    int bs = 256;
    int nb = numBlocks(g.n, bs);
    perturber_forceKernel<<<nb, bs>>>(
        g.ax, g.ay, g.az,
        g.x, g.y, g.z,
        px, py, pz, pmass, softening, g.n);
    nbody_check_cuda(cudaGetLastError(),      "perturber_forceKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "perturber_forceKernel sync");
}
