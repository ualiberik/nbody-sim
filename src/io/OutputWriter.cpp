#include "OutputWriter.h"
#include <stdexcept>
#include <cstdint>

// ---------------------------------------------------------------------------
// Helper: write a POD value as raw bytes
// ---------------------------------------------------------------------------
template<typename T>
static void writeRaw(std::ofstream& f, const T& val) {
    f.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
OutputWriter::OutputWriter(const std::string& output_dir, const Config& cfg)
    : cfg_(cfg)
{
    // Open frames.bin in binary write mode
    std::string frames_path = output_dir + "/frames.bin";
    frames_file_.open(frames_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!frames_file_.is_open())
        throw std::runtime_error("OutputWriter: cannot open " + frames_path);

    // Open stats.csv in text write mode
    std::string stats_path = output_dir + "/stats.csv";
    stats_file_.open(stats_path, std::ios::out | std::ios::trunc);
    if (!stats_file_.is_open())
        throw std::runtime_error("OutputWriter: cannot open " + stats_path);

    writeFramesHeader();
    writeStatsHeader();
}

// ---------------------------------------------------------------------------
// Destructor — flush both files
// ---------------------------------------------------------------------------
OutputWriter::~OutputWriter() {
    if (frames_file_.is_open()) frames_file_.flush();
    if (stats_file_.is_open())  stats_file_.flush();
}

// ---------------------------------------------------------------------------
// Binary header
//   Bytes  0-3 : magic "NBOD"
//   Bytes  4-7 : version = 1 (uint32_t)
//   Bytes  8-11: n_particles (uint32_t)
//   Bytes 12-15: n_frames = 0 placeholder (uint32_t) — patched by finalize()
//   Bytes 16-23: dt (double)
//   Bytes 24-31: output_dt = dt * output_every_n_steps (double)
// ---------------------------------------------------------------------------
void OutputWriter::writeFramesHeader() {
    // Magic
    frames_file_.write("NBOD", 4);

    // Version
    uint32_t version = 1u;
    writeRaw(frames_file_, version);

    // n_particles
    uint32_t n_part = static_cast<uint32_t>(cfg_.n_particles);
    writeRaw(frames_file_, n_part);

    // n_frames placeholder — remember offset so finalize() can patch it
    n_frames_offset_ = frames_file_.tellp();
    uint32_t n_frames_placeholder = 0u;
    writeRaw(frames_file_, n_frames_placeholder);

    // dt and output_dt as double
    double dt_d       = static_cast<double>(cfg_.dt);
    double output_dt  = static_cast<double>(cfg_.dt) *
                        static_cast<double>(cfg_.output_every_n_steps);
    writeRaw(frames_file_, dt_d);
    writeRaw(frames_file_, output_dt);
}

// ---------------------------------------------------------------------------
// CSV header
// ---------------------------------------------------------------------------
void OutputWriter::writeStatsHeader() {
    stats_file_ << "frame,sim_time_T0,sim_time_years,n_aggregates,"
                << "agg_id,n_particles,mass_Msun,mass_Mearth,"
                << "cx_AU,cy_AU,cz_AU,vx,vy,vz,dist_center_AU\n";
}

// ---------------------------------------------------------------------------
// writeFrame — one frame: sim_time (double) + x/y/z arrays (float32 each)
// ---------------------------------------------------------------------------
void OutputWriter::writeFrame(const ParticleData& cpu_particles, float sim_time_T0) {
    double t = static_cast<double>(sim_time_T0);
    writeRaw(frames_file_, t);

    int n = cpu_particles.n;
    frames_file_.write(reinterpret_cast<const char*>(cpu_particles.x), n * sizeof(float));
    frames_file_.write(reinterpret_cast<const char*>(cpu_particles.y), n * sizeof(float));
    frames_file_.write(reinterpret_cast<const char*>(cpu_particles.z), n * sizeof(float));
}

// ---------------------------------------------------------------------------
// writeStats — one CSV row per aggregate
// ---------------------------------------------------------------------------
void OutputWriter::writeStats(const std::vector<Aggregate>& aggregates,
                               int frame_idx, float sim_time_T0)
{
    int n_agg = static_cast<int>(aggregates.size());
    float sim_years = sim_time_T0 * T0_TO_YEARS;

    for (const Aggregate& agg : aggregates) {
        stats_file_
            << frame_idx                              << ','
            << sim_time_T0                            << ','
            << sim_years                              << ','
            << n_agg                                  << ','
            << agg.id                                 << ','
            << agg.n_particles                        << ','
            << agg.mass_msun                          << ','
            << (agg.mass_msun * MSUN_TO_MEARTH)       << ','
            << agg.cx                                 << ','
            << agg.cy                                 << ','
            << agg.cz                                 << ','
            << agg.vx                                 << ','
            << agg.vy                                 << ','
            << agg.vz                                 << ','
            << agg.dist_center                        << '\n';
    }
}

// ---------------------------------------------------------------------------
// finalize — seek back and patch n_frames in the binary header
// ---------------------------------------------------------------------------
void OutputWriter::finalize(int total_frames_written) {
    uint32_t n_frames = static_cast<uint32_t>(total_frames_written);
    frames_file_.seekp(n_frames_offset_);
    writeRaw(frames_file_, n_frames);
    frames_file_.flush();
    stats_file_.flush();
}
