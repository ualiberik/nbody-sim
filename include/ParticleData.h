#pragma once

#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#  include <cuda_runtime.h>

inline void nbody_check_cuda(cudaError_t err, const char* op) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + op + ": " + cudaGetErrorString(err));
}
#else
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
    nbody_check_cuda(cudaMalloc(&p.x,    n * sizeof(float)), "cudaMalloc x");
    nbody_check_cuda(cudaMalloc(&p.y,    n * sizeof(float)), "cudaMalloc y");
    nbody_check_cuda(cudaMalloc(&p.z,    n * sizeof(float)), "cudaMalloc z");
    nbody_check_cuda(cudaMalloc(&p.vx,   n * sizeof(float)), "cudaMalloc vx");
    nbody_check_cuda(cudaMalloc(&p.vy,   n * sizeof(float)), "cudaMalloc vy");
    nbody_check_cuda(cudaMalloc(&p.vz,   n * sizeof(float)), "cudaMalloc vz");
    nbody_check_cuda(cudaMalloc(&p.ax,   n * sizeof(float)), "cudaMalloc ax");
    nbody_check_cuda(cudaMalloc(&p.ay,   n * sizeof(float)), "cudaMalloc ay");
    nbody_check_cuda(cudaMalloc(&p.az,   n * sizeof(float)), "cudaMalloc az");
    nbody_check_cuda(cudaMalloc(&p.mass, n * sizeof(float)), "cudaMalloc mass");
    nbody_check_cuda(cudaMalloc(&p.id,   n * sizeof(int)),   "cudaMalloc id");
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
    if (cpu.n != gpu.n)
        throw std::runtime_error("copyToGPU: CPU n=" + std::to_string(cpu.n) + " != GPU n=" + std::to_string(gpu.n));
    int n = cpu.n;
    nbody_check_cuda(cudaMemcpy(gpu.x,    cpu.x,    n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU x");
    nbody_check_cuda(cudaMemcpy(gpu.y,    cpu.y,    n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU y");
    nbody_check_cuda(cudaMemcpy(gpu.z,    cpu.z,    n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU z");
    nbody_check_cuda(cudaMemcpy(gpu.vx,   cpu.vx,   n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU vx");
    nbody_check_cuda(cudaMemcpy(gpu.vy,   cpu.vy,   n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU vy");
    nbody_check_cuda(cudaMemcpy(gpu.vz,   cpu.vz,   n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU vz");
    nbody_check_cuda(cudaMemcpy(gpu.ax,   cpu.ax,   n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU ax");
    nbody_check_cuda(cudaMemcpy(gpu.ay,   cpu.ay,   n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU ay");
    nbody_check_cuda(cudaMemcpy(gpu.az,   cpu.az,   n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU az");
    nbody_check_cuda(cudaMemcpy(gpu.mass, cpu.mass, n * sizeof(float), cudaMemcpyHostToDevice), "copyToGPU mass");
    nbody_check_cuda(cudaMemcpy(gpu.id,   cpu.id,   n * sizeof(int),   cudaMemcpyHostToDevice), "copyToGPU id");
}

// Device -> Host copy
inline void copyToCPU(const ParticleData& gpu, ParticleData& cpu) {
    if (gpu.n != cpu.n)
        throw std::runtime_error("copyToCPU: GPU n=" + std::to_string(gpu.n) + " != CPU n=" + std::to_string(cpu.n));
    int n = gpu.n;
    nbody_check_cuda(cudaMemcpy(cpu.x,    gpu.x,    n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU x");
    nbody_check_cuda(cudaMemcpy(cpu.y,    gpu.y,    n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU y");
    nbody_check_cuda(cudaMemcpy(cpu.z,    gpu.z,    n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU z");
    nbody_check_cuda(cudaMemcpy(cpu.vx,   gpu.vx,   n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU vx");
    nbody_check_cuda(cudaMemcpy(cpu.vy,   gpu.vy,   n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU vy");
    nbody_check_cuda(cudaMemcpy(cpu.vz,   gpu.vz,   n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU vz");
    nbody_check_cuda(cudaMemcpy(cpu.ax,   gpu.ax,   n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU ax");
    nbody_check_cuda(cudaMemcpy(cpu.ay,   gpu.ay,   n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU ay");
    nbody_check_cuda(cudaMemcpy(cpu.az,   gpu.az,   n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU az");
    nbody_check_cuda(cudaMemcpy(cpu.mass, gpu.mass, n * sizeof(float), cudaMemcpyDeviceToHost), "copyToCPU mass");
    nbody_check_cuda(cudaMemcpy(cpu.id,   gpu.id,   n * sizeof(int),   cudaMemcpyDeviceToHost), "copyToCPU id");
}
#endif // !PARTICLE_DATA_NO_CUDA
