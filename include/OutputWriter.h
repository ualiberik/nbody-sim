#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <cstdint>
#include "ParticleData.h"
#include "Config.h"
#include "Aggregate.h"

// Unit conversion (G=1, natural units)
// 1 orbital period at 1 AU = 2*pi*T0 ~= 1 year. T0 ~= 1/(2*pi) years ~= 58.1 days.
constexpr float T0_TO_YEARS    = 1.0f / (2.0f * 3.14159265f);  // ~= 0.1592 yr/T0
constexpr float MSUN_TO_MEARTH = 332946.0f;

class OutputWriter {
public:
    // Opens/creates output_dir/frames.bin and output_dir/stats.csv
    OutputWriter(const std::string& output_dir, const Config& cfg);
    ~OutputWriter();

    // Write one frame of particle positions + timestamp to frames.bin.
    // agg_sizes_per_particle: array of length n_particles; element i is the
    //   size (particle count) of the aggregate containing particle i, or 1
    //   if the particle is not in any aggregate. Used by the visualizer to
    //   modulate brightness — bigger clumps glow brighter.
    void writeFrame(const ParticleData& cpu_particles, float sim_time_T0,
                    const std::vector<uint16_t>& agg_sizes_per_particle);

    // Write per-aggregate stats row(s) to stats.csv
    void writeStats(const std::vector<Aggregate>& aggregates,
                    int frame_idx, float sim_time_T0);

    // Write a single perturber row to stats.csv. agg_id is hard-coded to -1
    // and n_particles to 0 so downstream analysis can filter it out.
    void writePerturberStats(int frame_idx, float sim_time_T0,
                              int n_aggregates,
                              float px, float py, float pz,
                              float mass_msun, float omega);

    // Patch n_frames count in binary header (call once at end)
    void finalize(int total_frames_written);

private:
    std::ofstream frames_file_;
    std::ofstream stats_file_;
    Config cfg_;
    std::streampos n_frames_offset_;  // byte offset where n_frames is stored

    void writeFramesHeader();
    void writeStatsHeader();
};
