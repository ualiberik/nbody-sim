#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "Config.h"
#include "ParticleData.h"
#include "DiskInit.h"
#include "BHTree.cuh"
#include "Integrator.cuh"
#include "Aggregate.h"
#include "OutputWriter.h"
#include "Perturber.h"

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
            // Optional outer perturber (fixed Keplerian orbit)
            // -------------------------------------------------------------------
            Perturber pert = makePerturber(cfg);
            if (pert.enabled) {
                std::cout << "Perturber: " << cfg.perturber_mass_mjup
                          << " M_Jup at r=" << pert.r
                          << " AU, omega=" << pert.omega
                          << " rad/T0, period="
                          << (2.0 * 3.14159265 / pert.omega) << " T0\n";
            }

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
            if (pert.enabled) {
                updatePerturber(pert, 0.0f);
                launchPerturberForceKernel(gpu, pert.x, pert.y, pert.z,
                                            pert.mass_msun, cfg.softening_AU);
            }

            // -------------------------------------------------------------------
            // Main loop
            // -------------------------------------------------------------------
            for (int step = 0; step < total_steps; ++step) {

                // KDK step 1 — half kick:  v += a * (dt/2)
                launchHalfKickKernel(gpu, cfg.dt * 0.5f);

                // KDK step 2 — drift:      x += v * dt
                launchDriftKernel(gpu, cfg.dt);

                // sim_time of the new (post-drift) particle positions
                float sim_time_now = static_cast<float>(step + 1) * cfg.dt;

                // Rebuild tree with updated positions, compute new forces
                resetOctree(tree);
                launchBBoxKernel(tree, gpu);
                launchBuildKernel(tree, gpu);
                launchSummarizeKernel(tree);
                launchSortKernel(tree);
                launchResetAccelerationKernel(gpu);
                launchForceKernel(tree, gpu, cfg.theta, cfg.softening_AU, cfg.star_mass_msun);

                // Optional outer perturber: analytic Keplerian orbit, adds
                // gravity ONTO ax/ay/az (uses += inside the kernel).
                if (pert.enabled) {
                    updatePerturber(pert, sim_time_now);
                    launchPerturberForceKernel(gpu, pert.x, pert.y, pert.z,
                                                pert.mass_msun, cfg.softening_AU);
                }

                // Inelastic collision detection + impulse response (reuses built tree).
                // Perturber is NOT in the tree, so particles cannot "collide" with it.
                launchCollisionKernel(tree, gpu, r_coll, cfg.restitution);

                // KDK step 3 — second half kick:  v += a * (dt/2)
                launchHalfKickKernel(gpu, cfg.dt * 0.5f);

                // Gas drag (no-op when cfg.gas_drag_rate == 0).
                // Applied AFTER full KDK so it acts on the integrated velocity.
                launchGasDragKernel(gpu, cfg.gas_drag_rate, cfg.dt,
                                    cfg.star_mass_msun);

                // sim_time_now already computed above as (step+1)*dt.
                // ----- Output every output_every_n_steps -----
                if ((step + 1) % cfg.output_every_n_steps == 0) {
                    copyToCPU(gpu, cpu);

                    // Detect aggregates BEFORE writing frame so we can attach
                    // per-particle aggregate sizes to the binary frame.
                    auto aggs = detectAggregates(cpu, r_coll, cfg);

                    // Build per-particle aggregate-size lookup. Default 1 means
                    // "singleton, not in any FOF group". Saturate at uint16 max
                    // (65535) for the unlikely case of a planet engulfing it all.
                    std::vector<uint16_t> agg_sizes(cfg.n_particles, 1);
                    for (const Aggregate& a : aggs) {
                        uint16_t n_clamped = static_cast<uint16_t>(
                            std::min(a.n_particles, 65535));
                        for (int pid : a.particle_ids) {
                            if (pid >= 0 && pid < cfg.n_particles)
                                agg_sizes[pid] = n_clamped;
                        }
                    }

                    writer.writeFrame(cpu, sim_time_now, agg_sizes);
                    writer.writeStats(aggs, frames_written, sim_time_now);
                    // Append the perturber as a special row (agg_id = -1) so
                    // downstream analysis can locate / filter it.
                    if (pert.enabled) {
                        writer.writePerturberStats(
                            frames_written, sim_time_now,
                            static_cast<int>(aggs.size()),
                            pert.x, pert.y, pert.z,
                            pert.mass_msun, pert.omega);
                    }
                    ++frames_written;

                    float pct = 100.f * (step + 1) / total_steps;
                    std::cout << std::fixed << std::setprecision(1)
                              << "\r[" << pct << "%] t=" << sim_time_now
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
