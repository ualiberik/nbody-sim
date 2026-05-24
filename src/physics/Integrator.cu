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
