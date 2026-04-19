//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>
#include <fstream>
#include <optional>
#include <variant>
#include <vector>

#include "wave_cpu.h"

#ifdef AWAVE_CUDA
#include "wave_cuda.h"
using GpuWaveSimulation = CudaWaveSimulation;
#endif
#ifdef AWAVE_OMP
#include "wave_omp.h"
using GpuWaveSimulation = OmpWaveSimulation;
#endif
#ifdef AWAVE_SYCL
#include "wave_sycl.h"
using GpuWaveSimulation = SyclWaveSimulation;
#endif

namespace fs = std::filesystem;

using Clock = std::chrono::high_resolution_clock;
using Time = Clock::time_point;
using Dur = Clock::duration;

// For errors in parsing commandline so this can be caught separately
struct argparse_error : std::runtime_error {
    // Construct with a modern format string and args.
    template <typename... Args>
    explicit argparse_error(std::format_string<Args...> f, Args... args) : std::runtime_error(std::format(f, std::forward<Args>(args)...)) {
    }
};

struct RestartP {
    fs::path pth;
    int nsteps;
};
struct ParseResult {
    Params p;
    RunParams rp;
    fs::path outbase;
    std::optional<RestartP> restart;
    std::optional<fs::path> json;
};

ParseResult parse_args(int argc, char* argv[]) {
    // Defaults if not specified.
    shape_t shape{32, 32, 32};
    shape_t mpi_shape{1, 1, 1};
    double dx = 10.0; // metres
    double dt = 0.002; // seconds
    int nsteps = 100;
    bool set_steps = false;
    int out_period = 10;
    int nBoundaryLayers = 4;
    bool doIO = false;
    bool cpu_serial = false;
    bool skip_cpu = false;
    std::optional<fs::path> json = std::nullopt;
    std::optional<fs::path> restart = std::nullopt;
    fs::path outbase = "test";

    // Handle command line
    int nArgSeen = 0;
    for (int i = 1; i < argc; ++i) {
        auto a = std::string_view(argv[i]);

        if (a.starts_with('-')) {
            auto flag = a.substr(1);
            // Has an associated value
            ++i;
            if (i >= argc)
                throw argparse_error("No value for option {}", flag);
            auto val = std::string_view(argv[i]);

            auto read_fl = [&val](char const* flg, auto& var) {
                auto [ptr, ec] = std::from_chars(val.begin(), val.end(), var);
                if (ec != std::errc())
                    throw argparse_error("Error converting --{} to number", flg);
                if (ptr != val.end())
                    throw argparse_error("Did not consume all of --{} when converting", flg);
            };

            auto read_shape = [&flag, &val](shape_t& dest) {
                int dest_i = 0;
                for (std::size_t comma = 0; comma < std::string_view::npos; ++dest_i) {
                    if (dest_i == 3)
                        throw argparse_error("Too many values in {}", flag);

                    comma = val.find(',');
                    auto part = val.substr(0, std::min(comma, val.size()));
                    val = val.substr(comma + 1);
                    auto [ptr, ec] = std::from_chars(part.begin(), part.end(), dest[dest_i]);
                    if (ec != std::errc())
                        throw argparse_error("Error converting --{}[{}] to number", flag, dest_i);
                    if (ptr != part.end())
                        throw argparse_error("Did not consume all of --{}[{}] when converting", flag, dest_i);
                }
                if (dest_i != 3)
                    throw argparse_error("Not enough values in {}!", flag);
            };

            if (flag == "shape") {
                read_shape(shape);
            } else if (flag == "mpi") {
                read_shape(mpi_shape);
            } else if (flag == "dx") {
                read_fl("dx", dx);
            } else if (flag == "dt") {
                read_fl("dt", dt);
            } else if (flag == "nsteps") {
                read_fl("nsteps", nsteps);
                set_steps = true;
            } else if (flag == "out_period") {
                read_fl("out_period", out_period);
            } else if (flag == "io") {
                // No overload for bool
                int tmp = 0;
                read_fl("io", tmp);
                doIO = tmp;
            } else if (flag == "cpu_serial") {
                // No overload for bool
                int tmp = 0;
                read_fl("cpu_serial", tmp);
                cpu_serial = tmp;
            } else if (flag == "skip_cpu") {
                // No overload for bool
                int tmp = 0;
                read_fl("skip_cpu", tmp);
                skip_cpu = tmp;
            } else if (flag == "restart") {
                restart = std::make_optional<fs::path>(val);
            } else if (flag == "json") {
                json = std::make_optional<fs::path>(val);
            } else {
                throw argparse_error("Unknown flag {}", flag);
            }
        } else {
            if (nArgSeen)
                throw argparse_error("Unexpected positional argument {}", a);
            ++nArgSeen;
            outbase = a;
        }
    }
    return {
        Params{
            .dx=dx, .dt=dt, .shape=shape, 
            .nsteps=nsteps, .out_period=out_period,
            .nBoundaryLayers=nBoundaryLayers
        },
        RunParams{.mpi_shape=mpi_shape, .doIO=doIO, .cpu_serial=cpu_serial, .skip_cpu=skip_cpu},
        outbase,
        restart ? std::make_optional<RestartP>(*restart, set_steps ? nsteps : -1) : std::nullopt,
        json
    };
}


int main(int argc, char* argv[]) {
    auto mpi = MpiEnv(argc, argv);
    auto const rank = [] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }();

    try {
        // Parse command line args
        auto const [p, rp, outbase, restart, json] = parse_args(argc, argv);
        outroot(rank, "ASPP wave simulator starting\nProcess geometry: {}", rp.mpi_shape);
        // Set the different output files:
        auto mkcp = [&outbase] (auto infix) -> fs::path {
            auto stem = outbase.stem();
            auto newstem = stem.string();
            newstem += infix;
            newstem += ".vtkhdf";
            if (outbase.has_parent_path())
                return outbase.parent_path() / newstem;
            else
                return newstem;
        };
        auto cpu_cp = mkcp(".cpu");
        auto gpu_cp = mkcp(".gpu");

        CpuWaveSimulation cpu_state;
        if (restart) {
            cpu_state = CpuWaveSimulation::from_file(MPI_COMM_WORLD, rp, cpu_cp, restart->pth, restart->nsteps);
        } else {
            cpu_state = CpuWaveSimulation::from_params(MPI_COMM_WORLD, rp, cpu_cp, p);
        }
        // Set up the GPU implementation as copy of this
        auto gpu_state = GpuWaveSimulation::from_cpu_sim(gpu_cp, cpu_state);

        if (rp.cpu_serial) {
          cpu_state = CpuWaveSimulation();
          // set up again but force to be single process.
          auto srp = RunParams{.mpi_shape={1,1,1}, .doIO=(rank == 0 && rp.doIO), .cpu_serial=true};
          if (restart) {
            cpu_state = CpuWaveSimulation::from_file(MPI_COMM_SELF, srp, cpu_cp, restart->pth, restart->nsteps);
          } else {
            cpu_state = CpuWaveSimulation::from_params(MPI_COMM_SELF, srp, cpu_cp, p);
          }
        }

        // Timing stats
        auto n_chunks = ceildiv(p.nsteps - cpu_state.u.time(), p.out_period);
        auto cpu_time_s = std::vector<float>(n_chunks);
        auto gpu_time_s = std::vector<float>(n_chunks);

        // Runner/benchmarker lambda
        auto benchmarker = [&](
                WaveSimulation& state
        ) {
            std::vector<float> time_s(n_chunks);
            std::vector<float> sups(n_chunks);
            outroot(rank, "Starting run with {}, timing in {} chunks", state.ID(), n_chunks);
            auto& t = state.u.time();
            auto [nx, ny, nz] = state.decomp.global_shape;
            auto const nsites = nx*ny*nz;
            for (int i = 0; i < n_chunks; ++i) {
                auto len = std::min(t + p.out_period, p.nsteps) - t;
                // Get all processes in sync before starting the timer
                MPI_Barrier(state.decomp.comm);

                // Estimate the barrier delay
                auto barrier_delay = [&] {
                    Time const start = Clock::now();
                    MPI_Barrier(state.decomp.comm);
                    Time const stop = Clock::now();
                    return std::chrono::duration<float>(stop - start);
                }();
                Time const start = Clock::now();
                state.run(len);
                // Ensure all processes are finished, within the timing loop
                MPI_Barrier(state.decomp.comm);
                Time const stop = Clock::now();
                std::chrono::duration<float> dt{stop - start};

                // Subtract the barrier delay from the time recorded
                time_s[i] = dt.count() - barrier_delay.count();
                sups[i] = float(nsites * len) / time_s[i];
                // Exclude IO from timings
                state.append_u_fields();
                outroot(rank, "Chunk {}, length = {}, time = {} s", i, len, dt.count());
            }
            return std::make_pair(time_s, sups);
        };
        // Lamdba to check the results of GPU versions match the CPU reference
        auto checker = [&cpu_state, eps=1e-8](WaveSimulation const& sim) {
            auto& d = sim.decomp;
            outroot(d.rank, "Checking {} results...", sim.ID());
            int nerr = 0;
            auto const& ref_u = cpu_state.u.now();
            auto const& test_u = sim.u.now();
            auto& [L, M, N] = d.local_shape;
            shape_t ref_offset = sim.decomp.local_offset - cpu_state.decomp.local_offset;
            for (unsigned li = 0; li < L; ++li) {
                for (unsigned lj = 0; lj < M; ++lj) {
                    for (unsigned lk = 0; lk < N; ++lk) {
                        auto test_idx = shape_t{li+1 , lj+1, lk+1};
                        auto ref_idx = test_idx + ref_offset;
                        if (!approxEq(ref_u[ref_idx], test_u[test_idx], eps)) {
                            nerr += 1;
                            if (nerr < 50) {
                                auto global_idx = test_idx + sim.decomp.local_offset - 1;
                                out("Fields differ at {}", global_idx);
                            } else if (nerr == 50) {
                                out("Stopping reporting differences...");
                            }
                        }
                    }
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, &nerr, 1, MPI_INT, MPI_SUM, d.comm);
            outroot(d.rank, "Number of differences detected = {}", nerr);
            return nerr;
        };

	std::pair<std::vector<float>, std::vector<float>> cpu_stats;
	if (rp.skip_cpu) {
	  outroot(rank, "Skipping CPU run for speed");
	} else {
	  cpu_stats = benchmarker(cpu_state);
	}
        auto gpu_stats = benchmarker(gpu_state);
        int total_errs = rp.skip_cpu ? 0 : checker(gpu_state);

        auto print_stats = [] (auto const& stat_pair, char const* where) {
            std::vector<float> const& time = stat_pair.first;
            std::vector<float> const& sups = stat_pair.second;
            // Compute and print stats
            auto N = time.size();
            auto statter = [&N](std::vector<float> const& data) {
                auto min = std::numeric_limits<double>::infinity();
                double max = -min;
                double tsum = 0.0, tsumsq = 0.0;
                for (int i = 0; i < N; ++i) {
                    double const& t = data[i];
                    tsum += t;
                    tsumsq += t * t;
                    min = (t < min) ? t : min;
                    max = (t > max) ? t : max;
                }
                double mean = tsum / N;
                double tvar = (tsumsq - tsum * tsum / N) / (N - 1);
                double std = std::sqrt(tvar);
                out("min = {:.3e}, max = {:.3e}, mean = {:.3e}, std = {:.3e}",
                    min, max, mean, std);
            };

            out("Summary for {}", where);
            out("Run time / seconds");
            statter(time);
            out("Performance / (site updates per second)");
            statter(sups);
        };

        if (rank == 0) {
	    if (!rp.skip_cpu)
	        print_stats(cpu_stats, "CPU");
            print_stats(gpu_stats, gpu_state.ID());

            if (json) {
                // Horrible code to spit out the stats as dense JSON
                auto& jp = *json;
                std::ofstream js(*json);
                auto jstat = [&](auto const& stat_pair) {
                    auto& [time, sups] = stat_pair;
                    js << '{';
                    auto jvec = [&](auto& v) {
                        js << '[';
                        bool first = true;
                        for (auto& el: v) {
                            if (!first)
                                js << ',';
                            else
                                first = false;
                            js << el;
                        }
                        js << ']';
                    };
                    js << "\"time\":";
                    jvec(time);
                    js << ",\"sups\":";
                    jvec(sups);
                    js << '}';
                };
                js << "{\"ndiff\":" << total_errs;
		if (!rp.skip_cpu) {
		    js << ",\"CPU\":";
		    jstat(cpu_stats);
		}
                js << ",\"GPU\":";
                jstat(gpu_stats);
                js << '}';
            }
        }

        return total_errs;

    } catch (argparse_error& e) {
        if (rank == 0) {
            std::cerr << e.what() << std::endl;
            std::cerr << "Usage: awave [flags] [output_base]\n"
              "output_base defaults to 'test'\n"
              "\n"
              "Flag               | Default  | Description\n"
              "-------------------|----------|-------------\n"
              "-mpi int,int,int   |    1,1,1 | Layout of the grid of MPI processes\n"
              "-shape int,int,int | 32,32,32 | Shape of the domain to simulated=\n"
              "-dx float          |     10.0 | Space step in metres\n"
              "-dt float          |    0.002 | Time step in seconds\n"
              "-nsteps int        |      100 | Number of time steps\n"
              "-out_period int    |       10 | Number of time steps between outputs\n"
              "-restart path      |  nullopt | Restart from checkpoint (ignores parameters above)\n"
              "-io (0|1)          |    false | Actually write output\n"
              "-cpu_serial (0|1)  |    false | Force CPU run to be single MPI process\n"
              "-skip_cpu (0|1)    |    false | Skip the CPU run and just measure GPU performance\n"
              "-json path         |  nullopt | File to write JSON output to\n"
              ;
        }
        return 1;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
