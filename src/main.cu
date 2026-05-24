#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <string>
#include <filesystem>
#include "Config.h"
#include "ParticleData.h"
#include "DiskInit.h"
#include "BHTree.cuh"
#include "Integrator.cuh"
#include "Aggregate.h"
#include "OutputWriter.h"

int main(int argc, char** argv) {
    std::string cfg_path = (argc > 1) ? argv[1] : "config/simulation.json";
    std::cout << "Loading config: " << cfg_path << "\n";
    Config cfg = loadConfig(cfg_path);

    // Create timestamped output directory so each run is independent
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    char ts[20];
    std::tm tm_buf{};
    localtime_s(&tm_buf, &t);   // MSVC thread-safe variant
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
    std::string run_dir = cfg.output_dir + "/" + std::string(ts);

    {
        std::error_code ec;
        std::filesystem::create_directories(run_dir, ec);
        if (ec) {
            std::cerr << "Failed to create output dir '" << run_dir
                      << "': " << ec.message() << "\n";
            return 1;
        }
    }
    std::cout << "Output dir: " << run_dir << "\n";

    // ----- Initialise disk -----
    std::cout << "Initializing disk (" << cfg.n_particles << " particles)...\n";
    ParticleData cpu = initDisk(cfg);          // returns CPU SoA
    ParticleData gpu = allocateParticlesGPU(cfg.n_particles);

    // Wrap simulation body so GPU memory is freed even if a CUDA kernel throws.
    try {
        copyToGPU(cpu, gpu);

        // ----- Allocate octree -----
        OctreeData tree = allocateOctree(cfg.n_particles);

        try {
            // Collision detection radius = collision_radius_factor × physical radius
            float r_phys = particlePhysicalRadius(cfg);
            float r_coll = cfg.collision_radius_factor * r_phys;

            // ----- Output writer -----
            OutputWriter writer(run_dir, cfg);

            int   total_steps    = static_cast<int>(cfg.total_time / cfg.dt);
            int   frames_written = 0;

            std::cout << "Starting simulation: " << total_steps << " steps, "
                      << "output every " << cfg.output_every_n_steps << " steps\n";

            // -------------------------------------------------------------------
            // KDK Leapfrog: compute accelerations at t=0 before the first half-kick
            // -------------------------------------------------------------------
            resetOctree(tree);
            launchBBoxKernel(tree, gpu);
            launchBuildKernel(tree, gpu);
            launchSummarizeKernel(tree);
            launchSortKernel(tree);
            launchResetAccelerationKernel(gpu);
            launchForceKernel(tree, gpu, cfg.theta, cfg.softening_AU, cfg.star_mass_msun);

            // -------------------------------------------------------------------
            // Main loop
            // -------------------------------------------------------------------
            for (int step = 0; step < total_steps; ++step) {

                // KDK step 1 — half kick:  v += a * (dt/2)
                launchHalfKickKernel(gpu, cfg.dt * 0.5f);

                // KDK step 2 — drift:      x += v * dt
                launchDriftKernel(gpu, cfg.dt);

                // Rebuild tree with updated positions, compute new forces
                resetOctree(tree);
                launchBBoxKernel(tree, gpu);
                launchBuildKernel(tree, gpu);
                launchSummarizeKernel(tree);
                launchSortKernel(tree);
                launchResetAccelerationKernel(gpu);
                launchForceKernel(tree, gpu, cfg.theta, cfg.softening_AU, cfg.star_mass_msun);

                // Inelastic collision detection + impulse response (reuses built tree)
                launchCollisionKernel(tree, gpu, r_coll, cfg.restitution);

                // KDK step 3 — second half kick:  v += a * (dt/2)
                launchHalfKickKernel(gpu, cfg.dt * 0.5f);

                // Derive sim_time from step count to avoid float accumulation error
                // (dt=0.01 is not exactly representable; 200k additions would drift ~1e-4)
                float sim_time = static_cast<float>(step + 1) * cfg.dt;

                // ----- Output every output_every_n_steps -----
                if ((step + 1) % cfg.output_every_n_steps == 0) {
                    copyToCPU(gpu, cpu);
                    writer.writeFrame(cpu, sim_time);

                    auto aggs = detectAggregates(cpu, r_coll, cfg);
                    writer.writeStats(aggs, frames_written, sim_time);
                    ++frames_written;

                    float pct = 100.f * (step + 1) / total_steps;
                    std::cout << std::fixed << std::setprecision(1)
                              << "\r[" << pct << "%] t=" << sim_time
                              << " T0  aggregates=" << aggs.size() << "    "
                              << std::flush;
                }
            }

            writer.finalize(frames_written);
            std::cout << "\nDone. " << frames_written << " frames written to "
                      << run_dir << "\n";

        } catch (...) {
            freeOctree(tree);
            throw;
        }
        freeOctree(tree);

    } catch (const std::exception& e) {
        std::cerr << "\nSimulation error: " << e.what() << "\n";
        freeParticlesCPU(cpu);
        freeParticlesGPU(gpu);
        return 1;
    }

    freeParticlesCPU(cpu);
    freeParticlesGPU(gpu);
    return 0;
}
