#include "Integrator.cuh"
#include <cuda_runtime.h>

// Prefix "integrator_" on all __global__ symbols to avoid link collisions
// with other .cu translation units under CUDA_SEPARABLE_COMPILATION.

__global__ void integrator_halfKickKernel(float* vx, float* vy, float* vz,
                                           const float* ax, const float* ay, const float* az,
                                           float dt_half, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += ax[i] * dt_half;
    vy[i] += ay[i] * dt_half;
    vz[i] += az[i] * dt_half;
}

__global__ void integrator_driftKernel(float* x, float* y, float* z,
                                        const float* vx, const float* vy, const float* vz,
                                        float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

__global__ void integrator_resetAccKernel(float* ax, float* ay, float* az, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ax[i] = 0.f;
    ay[i] = 0.f;
    az[i] = 0.f;
}

// Gas-drag damping. Decompose particle velocity into (radial, tangential,
// vertical) relative to the XZ disk plane around the star. A circular
// Keplerian orbit has v_r = 0, |v_phi| = sqrt(M*/r_xz), v_y = 0. We damp
// the deviations of v_r, v_y, and (|v_phi| - v_kep) toward zero while
// preserving the sign of v_phi (so prograde stays prograde).
__global__ void integrator_gasDragKernel(
        const float* x,  const float* y,  const float* z,
        float* vx, float* vy, float* vz,
        float decay, float star_mass_msun, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = x[i], yi = y[i], zi = z[i];
    float r_xz2 = xi*xi + zi*zi;
    if (r_xz2 < 1e-12f) return;                  // skip particle ~at star
    float r_xz  = sqrtf(r_xz2);
    float inv_r = 1.0f / r_xz;

    // Decompose horizontal velocity:
    //   radial unit vector  r̂ = ( xi, 0, zi) / r_xz
    //   tangential          φ̂ = (-zi, 0, xi) / r_xz   (prograde when v_phi>0)
    float vxi = vx[i];
    float vyi = vy[i];
    float vzi = vz[i];
    float v_r   = (vxi * xi + vzi * zi) * inv_r;
    float v_phi = (vxi * (-zi) + vzi * xi) * inv_r;

    // Target circular Keplerian speed at this radius, signed to keep direction.
    float v_kep_mag = sqrtf(star_mass_msun * inv_r);
    float v_kep     = (v_phi >= 0.0f) ? v_kep_mag : -v_kep_mag;

    // Damp deviation from target (radial=0, v_phi=v_kep, vertical=0).
    float v_r_new   = v_r * decay;
    float v_phi_new = v_kep + (v_phi - v_kep) * decay;
    float v_y_new   = vyi * decay;

    // Reassemble cartesian velocity.
    vx[i] = v_r_new * (xi * inv_r) + v_phi_new * (-zi * inv_r);
    vz[i] = v_r_new * (zi * inv_r) + v_phi_new * ( xi * inv_r);
    vy[i] = v_y_new;
}

static int numBlocks(int n, int bs) { return (n + bs - 1) / bs; }

void launchHalfKickKernel(ParticleData& g, float dt_half) {
    integrator_halfKickKernel<<<numBlocks(g.n, 256), 256>>>(
        g.vx, g.vy, g.vz, g.ax, g.ay, g.az, dt_half, g.n);
    nbody_check_cuda(cudaGetLastError(),   "integrator_halfKickKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "integrator_halfKickKernel sync");
}

void launchDriftKernel(ParticleData& g, float dt) {
    integrator_driftKernel<<<numBlocks(g.n, 256), 256>>>(
        g.x, g.y, g.z, g.vx, g.vy, g.vz, dt, g.n);
    nbody_check_cuda(cudaGetLastError(),   "integrator_driftKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "integrator_driftKernel sync");
}

void launchResetAccelerationKernel(ParticleData& g) {
    integrator_resetAccKernel<<<numBlocks(g.n, 256), 256>>>(g.ax, g.ay, g.az, g.n);
    nbody_check_cuda(cudaGetLastError(),   "integrator_resetAccKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "integrator_resetAccKernel sync");
}

void launchGasDragKernel(ParticleData& g, float drag_rate, float dt,
                          float star_mass_msun) {
    if (drag_rate <= 0.0f) return;                   // disabled — no-op
    float decay = expf(-drag_rate * dt);             // per-step retention factor
    integrator_gasDragKernel<<<numBlocks(g.n, 256), 256>>>(
        g.x, g.y, g.z, g.vx, g.vy, g.vz,
        decay, star_mass_msun, g.n);
    nbody_check_cuda(cudaGetLastError(),   "integrator_gasDragKernel launch");
    nbody_check_cuda(cudaDeviceSynchronize(), "integrator_gasDragKernel sync");
}
