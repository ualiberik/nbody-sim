#pragma once
#include "ParticleData.h"

// Leapfrog KDK integration kernels.
// Each launcher uses ceil(n/256) blocks x 256 threads.
// All synchronize (cudaDeviceSynchronize) before returning.

// KDK step 1 & 3: v += a * dt_half
void launchHalfKickKernel(ParticleData& gpu, float dt_half);

// KDK step 2: x += v * dt
void launchDriftKernel(ParticleData& gpu, float dt);

// Reset: ax = ay = az = 0
void launchResetAccelerationKernel(ParticleData& gpu);

// Gas-drag damping: damps the deviation of each particle's velocity from
// the local circular Keplerian orbit (XZ-plane around the star) and damps
// vertical (Y) motion. Orbital direction (prograde/retrograde) is preserved.
// drag_rate is in units of 1/T0; the per-step decay factor is exp(-drag_rate * dt).
void launchGasDragKernel(ParticleData& gpu, float drag_rate, float dt,
                          float star_mass_msun);
