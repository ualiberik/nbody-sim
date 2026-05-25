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
// Destructor — close both files (implicitly flushes)
// ---------------------------------------------------------------------------
OutputWriter::~OutputWriter() {
    if (frames_file_.is_open()) frames_file_.close();
    if (stats_file_.is_open())  stats_file_.close();
}

// ---------------------------------------------------------------------------
// Binary header (32 bytes)
//   Bytes  0-3 : magic "NBOD"
//   Bytes  4-7 : version = 2 (uint32_t)
//                  v1: per-frame layout = time(8) + x[n](4n) + y[n](4n) + z[n](4n)
//                  v2: per-frame layout = time(8) + x/y/z(12n) + agg_size[n](2n)
//                       agg_size is uint16_t: size of aggregate containing
//                       particle i (1 if singleton).
//   Bytes  8-11: n_particles (uint32_t)
//   Bytes 12-15: n_frames = 0 placeholder (uint32_t) — patched by finalize()
//   Bytes 16-23: dt (double)
//   Bytes 24-31: output_dt = dt * output_every_n_steps (double)
// ---------------------------------------------------------------------------
void OutputWriter::writeFramesHeader() {
    // Magic
    frames_file_.write("NBOD", 4);

    // Version
    uint32_t version = 2u;
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
// writeFrame — one frame (format v2):
//   time_T0        double
//   x[n], y[n], z[n]   float32 SoA
//   agg_size[n]    uint16_t (particle's aggregate size, 1 = singleton)
// ---------------------------------------------------------------------------
void OutputWriter::writeFrame(const ParticleData& cpu_particles, float sim_time_T0,
                               const std::vector<uint16_t>& agg_sizes_per_particle)
{
    int n = cpu_particles.n;
    if (static_cast<int>(agg_sizes_per_particle.size()) != n)
        throw std::runtime_error(
            "OutputWriter::writeFrame: agg_sizes_per_particle size mismatch");

    double t = static_cast<double>(sim_time_T0);
    writeRaw(frames_file_, t);

    frames_file_.write(reinterpret_cast<const char*>(cpu_particles.x), n * sizeof(float));
    frames_file_.write(reinterpret_cast<const char*>(cpu_particles.y), n * sizeof(float));
    frames_file_.write(reinterpret_cast<const char*>(cpu_particles.z), n * sizeof(float));

    // Aggregate sizes as uint16 (saves 50% over float32; max value 65535
    // far exceeds any realistic aggregate count).
    frames_file_.write(reinterpret_cast<const char*>(agg_sizes_per_particle.data()),
                       n * sizeof(uint16_t));
}

// ---------------------------------------------------------------------------
// writeStats — one CSV row per aggregate.
// If aggregates is empty, no row is written for this frame (early simulation
// frames before any particles cluster will produce zero rows).
// Note: finalize() must be called before process exit to guarantee all
// buffered CSV rows are flushed to disk.
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
// writePerturberStats — record the fixed outer perturber as a special row.
//
// Schema is the same as the per-aggregate row but:
//   agg_id      = -1   (sentinel for analysis filters)
//   n_particles =  0
//   mass_*      = perturber mass
//   cx/cy/cz    = perturber XYZ position
//   vx/vy/vz    = analytic Keplerian velocity (perpendicular to r in XZ plane)
//   dist_center = perturber orbital radius
// ---------------------------------------------------------------------------
void OutputWriter::writePerturberStats(int frame_idx, float sim_time_T0,
                                        int n_aggregates,
                                        float px, float py, float pz,
                                        float mass_msun, float omega)
{
    float sim_years = sim_time_T0 * T0_TO_YEARS;
    float r         = std::sqrt(px*px + pz*pz);
    // Tangent in XZ plane: phi-hat = (-z, 0, x)/r, velocity magnitude = omega*r
    float v_mag = omega * r;
    float vx = (r > 0.f) ? (-pz / r) * v_mag : 0.f;
    float vz = (r > 0.f) ? ( px / r) * v_mag : 0.f;
    float vy = 0.f;

    stats_file_
        << frame_idx        << ','
        << sim_time_T0      << ','
        << sim_years        << ','
        << n_aggregates     << ','
        << -1               << ','   // agg_id sentinel
        << 0                << ','   // n_particles
        << mass_msun        << ','
        << (mass_msun * MSUN_TO_MEARTH) << ','
        << px << ',' << py << ',' << pz << ','
        << vx << ',' << vy << ',' << vz << ','
        << r << '\n';
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
