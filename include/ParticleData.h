#pragma once

// Include CUDA runtime when compiled by nvcc or when CUDA headers are available.
// CPU-only translation units (plain .cpp) only use the CPU alloc/free functions.
#ifdef __CUDACC__
#  include <cuda_runtime.h>
#elif defined(CUDA_VERSION)
#  include <cuda_runtime.h>
#else
// Forward-declare minimal CUDA types so GPU helpers compile in CUDA TUs only.
#  define PARTICLE_DATA_NO_CUDA
#endif

// Struct-of-Arrays layout for CUDA coalesced memory access.
// Positions in AU, velocities in AU/T0, accelerations in AU/T0^2, mass in M_sun.
struct ParticleData {
    float* x    = nullptr;
    float* y    = nullptr;
    float* z    = nullptr;
    float* vx   = nullptr;
    float* vy   = nullptr;
    float* vz   = nullptr;
    float* ax   = nullptr;
    float* ay   = nullptr;
    float* az   = nullptr;
    float* mass = nullptr;
    int*   id   = nullptr;
    int    n    = 0;
};

// CPU allocation (new[])
inline ParticleData allocateParticlesCPU(int n) {
    ParticleData p;
    p.n    = n;
    p.x    = new float[n](); p.y    = new float[n](); p.z    = new float[n]();
    p.vx   = new float[n](); p.vy   = new float[n](); p.vz   = new float[n]();
    p.ax   = new float[n](); p.ay   = new float[n](); p.az   = new float[n]();
    p.mass = new float[n]();
    p.id   = new int[n]();
    return p;
}

inline void freeParticlesCPU(ParticleData& p) {
    delete[] p.x;  delete[] p.y;  delete[] p.z;
    delete[] p.vx; delete[] p.vy; delete[] p.vz;
    delete[] p.ax; delete[] p.ay; delete[] p.az;
    delete[] p.mass;
    delete[] p.id;
    p.x = p.y = p.z = nullptr;
    p.vx = p.vy = p.vz = nullptr;
    p.ax = p.ay = p.az = nullptr;
    p.mass = nullptr; p.id = nullptr;
    p.n = 0;
}

#ifndef PARTICLE_DATA_NO_CUDA
// GPU allocation (cudaMalloc)
inline ParticleData allocateParticlesGPU(int n) {
    ParticleData p; p.n = n;
    cudaMalloc(&p.x,    n * sizeof(float));
    cudaMalloc(&p.y,    n * sizeof(float));
    cudaMalloc(&p.z,    n * sizeof(float));
    cudaMalloc(&p.vx,   n * sizeof(float));
    cudaMalloc(&p.vy,   n * sizeof(float));
    cudaMalloc(&p.vz,   n * sizeof(float));
    cudaMalloc(&p.ax,   n * sizeof(float));
    cudaMalloc(&p.ay,   n * sizeof(float));
    cudaMalloc(&p.az,   n * sizeof(float));
    cudaMalloc(&p.mass, n * sizeof(float));
    cudaMalloc(&p.id,   n * sizeof(int));
    return p;
}

inline void freeParticlesGPU(ParticleData& p) {
    cudaFree(p.x);  cudaFree(p.y);  cudaFree(p.z);
    cudaFree(p.vx); cudaFree(p.vy); cudaFree(p.vz);
    cudaFree(p.ax); cudaFree(p.ay); cudaFree(p.az);
    cudaFree(p.mass); cudaFree(p.id);
    p.x = p.y = p.z = nullptr;
    p.vx = p.vy = p.vz = nullptr;
    p.ax = p.ay = p.az = nullptr;
    p.mass = nullptr; p.id = nullptr;
    p.n = 0;
}

// Host -> Device copy
inline void copyToGPU(const ParticleData& cpu, ParticleData& gpu) {
    int n = cpu.n;
    cudaMemcpy(gpu.x,    cpu.x,    n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.y,    cpu.y,    n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.z,    cpu.z,    n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.vx,   cpu.vx,   n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.vy,   cpu.vy,   n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.vz,   cpu.vz,   n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.ax,   cpu.ax,   n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.ay,   cpu.ay,   n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.az,   cpu.az,   n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.mass, cpu.mass, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.id,   cpu.id,   n * sizeof(int),   cudaMemcpyHostToDevice);
}

// Device -> Host copy
inline void copyToCPU(const ParticleData& gpu, ParticleData& cpu) {
    int n = gpu.n;
    cudaMemcpy(cpu.x,    gpu.x,    n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.y,    gpu.y,    n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.z,    gpu.z,    n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.vx,   gpu.vx,   n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.vy,   gpu.vy,   n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.vz,   gpu.vz,   n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.ax,   gpu.ax,   n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.ay,   gpu.ay,   n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.az,   gpu.az,   n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.mass, gpu.mass, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.id,   gpu.id,   n * sizeof(int),   cudaMemcpyDeviceToHost);
}
#endif // !PARTICLE_DATA_NO_CUDA
