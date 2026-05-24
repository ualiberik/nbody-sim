#pragma once
#include <string>
#include <fstream>
#include <vector>
#include "ParticleData.h"
#include "Config.h"
#include "Aggregate.h"

// Unit conversion (G=1, natural units)
// 1 orbital period at 1 AU = 2pi T0 ~= 365.25 days
// T0 ~= 58.1 days ~= 1/(2pi) years
constexpr float T0_TO_YEARS    = 1.0f / (2.0f * 3.14159265f);  // ~= 0.1592 yr/T0
constexpr float MSUN_TO_MEARTH = 332946.0f;

class OutputWriter {
public:
    // Opens/creates output_dir/frames.bin and output_dir/stats.csv
    OutputWriter(const std::string& output_dir, const Config& cfg);
    ~OutputWriter();

    // Write one frame of particle positions + timestamp to frames.bin
    void writeFrame(const ParticleData& cpu_particles, float sim_time_T0);

    // Write per-aggregate stats row(s) to stats.csv
    void writeStats(const std::vector<Aggregate>& aggregates,
                    int frame_idx, float sim_time_T0);

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
