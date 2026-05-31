#pragma once
#include "ParticleData.h"
#include "Config.h"

// Initialize CPU particles as a protoplanetary disk.
// Disk lies in XZ plane with Gaussian thickness along Y.
// Surface density: Sigma(r) proportional to r^(-3/2) (MMSN profile).
// Velocities: circular v = sqrt(G*M_star/r) + 1% random noise (G=1).
// Returns initialized CPU ParticleData — caller owns memory (call freeParticlesCPU).
ParticleData initDisk(const Config& cfg, unsigned int seed = 42);

// Physical radius of one particle in AU.
// Assumes rocky density = 3000 kg/m^3, mass = cfg.particle_mass_msun.
float particlePhysicalRadius(const Config& cfg);
