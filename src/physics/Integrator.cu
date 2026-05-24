#include "Integrator.cuh"
#include <cuda_runtime.h>

__global__ void halfKickKernel(float* vx, float* vy, float* vz,
                                const float* ax, const float* ay, const float* az,
                                float dt_half, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += ax[i] * dt_half;
    vy[i] += ay[i] * dt_half;
    vz[i] += az[i] * dt_half;
}

__global__ void driftKernel(float* x, float* y, float* z,
                             const float* vx, const float* vy, const float* vz,
                             float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

__global__ void resetAccKernel(float* ax, float* ay, float* az, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ax[i] = 0.f;
    ay[i] = 0.f;
    az[i] = 0.f;
}

static int numBlocks(int n, int bs) { return (n + bs - 1) / bs; }

void launchHalfKickKernel(ParticleData& g, float dt_half) {
    halfKickKernel<<<numBlocks(g.n, 256), 256>>>(
        g.vx, g.vy, g.vz, g.ax, g.ay, g.az, dt_half, g.n);
    cudaDeviceSynchronize();
}

void launchDriftKernel(ParticleData& g, float dt) {
    driftKernel<<<numBlocks(g.n, 256), 256>>>(
        g.x, g.y, g.z, g.vx, g.vy, g.vz, dt, g.n);
    cudaDeviceSynchronize();
}

void launchResetAccelerationKernel(ParticleData& g) {
    resetAccKernel<<<numBlocks(g.n, 256), 256>>>(g.ax, g.ay, g.az, g.n);
    cudaDeviceSynchronize();
}
