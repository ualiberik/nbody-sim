#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <cstdint>
#include "OutputWriter.h"
#include "ParticleData.h"
#include "Config.h"
#include "Aggregate.h"

// ---------------------------------------------------------------------------
// Test 1: frames.bin header is correct after writing two frames
// ---------------------------------------------------------------------------
TEST_CASE("OutputWriter writes valid frames.bin header", "[output]") {
    Config cfg;
    cfg.n_particles          = 4;
    cfg.dt                   = 0.01f;
    cfg.output_every_n_steps = 1;
    cfg.output_dir           = "data/test_run_output";
    std::filesystem::create_directories(cfg.output_dir);

    ParticleData p = allocateParticlesCPU(4);
    p.x[0]=1.f; p.y[0]=0.f; p.z[0]=0.f;
    p.x[1]=2.f; p.y[1]=0.f; p.z[1]=0.f;
    p.x[2]=3.f; p.y[2]=0.f; p.z[2]=0.f;
    p.x[3]=4.f; p.y[3]=0.f; p.z[3]=0.f;

    {
        OutputWriter writer(cfg.output_dir, cfg);
        writer.writeFrame(p, 0.0f);
        writer.writeFrame(p, 1.0f);
        writer.finalize(2);
    }

    // --- verify binary header ---
    std::ifstream f(cfg.output_dir + "/frames.bin", std::ios::binary);
    REQUIRE(f.is_open());

    char magic[4];
    f.read(magic, 4);
    REQUIRE(magic[0] == 'N');
    REQUIRE(magic[1] == 'B');
    REQUIRE(magic[2] == 'O');
    REQUIRE(magic[3] == 'D');

    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    REQUIRE(version == 1u);

    uint32_t n_particles = 0;
    f.read(reinterpret_cast<char*>(&n_particles), 4);
    REQUIRE(n_particles == 4u);

    uint32_t n_frames = 0;
    f.read(reinterpret_cast<char*>(&n_frames), 4);
    REQUIRE(n_frames == 2u);

    double dt_d = 0.0;
    f.read(reinterpret_cast<char*>(&dt_d), 8);
    REQUIRE_THAT(dt_d, Catch::Matchers::WithinRel(0.01, 1e-6));

    double output_dt = 0.0;
    f.read(reinterpret_cast<char*>(&output_dt), 8);
    // output_every_n_steps = 1, so output_dt == dt
    REQUIRE_THAT(output_dt, Catch::Matchers::WithinRel(0.01, 1e-6));

    // --- verify first frame: sim_time then x array ---
    double t0 = -1.0;
    f.read(reinterpret_cast<char*>(&t0), 8);
    REQUIRE_THAT(t0, Catch::Matchers::WithinAbs(0.0, 1e-9));

    float x0 = 0.f;
    f.read(reinterpret_cast<char*>(&x0), 4);
    REQUIRE_THAT(x0, Catch::Matchers::WithinRel(1.0f, 1e-5f));

    f.close();
    freeParticlesCPU(p);
    std::filesystem::remove_all(cfg.output_dir);
}

// ---------------------------------------------------------------------------
// Test 2: stats.csv has correct header and one row per aggregate per frame
// ---------------------------------------------------------------------------
TEST_CASE("OutputWriter writes correct stats.csv header and rows", "[output]") {
    Config cfg;
    cfg.n_particles          = 2;
    cfg.dt                   = 0.05f;
    cfg.output_every_n_steps = 10;
    cfg.output_dir           = "data/test_run_stats";
    std::filesystem::create_directories(cfg.output_dir);

    ParticleData p = allocateParticlesCPU(2);

    // Build two aggregates
    Aggregate a0;
    a0.id = 0; a0.n_particles = 2; a0.mass_msun = 6.006e-7f;
    a0.cx = 1.5f; a0.cy = 0.f; a0.cz = 0.f;
    a0.vx = 0.f;  a0.vy = 1.f; a0.vz = 0.f;
    a0.dist_center = 1.5f;

    Aggregate a1;
    a1.id = 1; a1.n_particles = 1; a1.mass_msun = 3.003e-7f;
    a1.cx = 3.f; a1.cy = 0.f; a1.cz = 0.f;
    a1.vx = 0.f; a1.vy = 0.577f; a1.vz = 0.f;
    a1.dist_center = 3.f;

    std::vector<Aggregate> aggs = {a0, a1};

    {
        OutputWriter writer(cfg.output_dir, cfg);
        writer.writeFrame(p, 0.0f);
        writer.writeStats(aggs, 0, 0.0f);
        writer.finalize(1);
    }

    // Read stats.csv back
    std::ifstream csv(cfg.output_dir + "/stats.csv");
    REQUIRE(csv.is_open());

    std::string header;
    std::getline(csv, header);
    REQUIRE(header == "frame,sim_time_T0,sim_time_years,n_aggregates,"
                      "agg_id,n_particles,mass_Msun,mass_Mearth,"
                      "cx_AU,cy_AU,cz_AU,vx,vy,vz,dist_center_AU");

    // Two data rows (one per aggregate)
    std::string row0, row1, extra;
    REQUIRE(std::getline(csv, row0));
    REQUIRE(std::getline(csv, row1));
    REQUIRE(!std::getline(csv, extra));  // no third row

    // Row 0 should start with "0," (frame 0) and contain agg_id 0
    REQUIRE(row0.find("0,") == 0);          // frame == 0
    REQUIRE(row0.find(",0,") != std::string::npos);  // agg_id 0 somewhere

    // Row 1 should contain agg_id 1
    REQUIRE(row1.find(",1,") != std::string::npos);

    csv.close();
    freeParticlesCPU(p);
    std::filesystem::remove_all(cfg.output_dir);
}
