//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_cpu.h"

#include <algorithm>
#include <iostream>

#include "init_sos.h"

static int get_true_rank(RunParams const& rp, MPI_Comm c) {
    if (rp.cpu_serial) {
      c = MPI_COMM_WORLD;
    }
    int r;
    MPI_Comm_rank(c, &r);
    return r;
}

// Restart from a checkpoint
CpuWaveSimulation CpuWaveSimulation::from_file(MPI_Comm c, RunParams const& rp, fs::path const& dest, fs::path const& src, int nsteps) {
    auto const truerank = get_true_rank(rp, c);
    outroot(truerank, "Initialising from checkpoint: {}", src.c_str());
    auto src_data = H5IO::read_only(src, c, rp.mpi_shape);
    CpuWaveSimulation ans;
    ans.checkpoint = dest;
    ans.params = src_data.get_params();
    if (nsteps > 0)
      ans.params.nsteps = nsteps;
    outroot(truerank, "Parameters:\n{}", ans.params);
    ans.decomp = Decomposition(c, rp.mpi_shape, ans.params.shape);
    outroot(truerank, "Loading fields...");
    ans.damp = src_data.get_damp();
    ans.sos = src_data.get_sos();
    ans.cs2 = array3d(ans.sos.shape());
    std::transform(
        ans.sos.data(), ans.sos.data() + ans.sos.size(),
        ans.cs2.data(),
        [](double const& c) {
            return c*c;
        }
    );
    ans.u = src_data.get_last_u();
    outroot(truerank, "Current time step: {}", ans.u.time());
    if (rp.doIO) {
        outroot(truerank, "Write initial conditions to {}", ans.checkpoint.c_str());
        ans.h5 = H5IO::from_params(ans.checkpoint, ans.params, ans.decomp);
        ans.h5.put_params(ans.params);
        ans.h5.put_damp(ans.damp);
        ans.h5.put_sos(ans.sos);
        ans.append_u_fields();
    } else {
        outroot(truerank, "IO off, skipping");
    }

    return ans;
}

CpuWaveSimulation CpuWaveSimulation::from_params(MPI_Comm c, RunParams const& rp, fs::path const& cp, Params const& p) {
    CpuWaveSimulation ans;
    auto& d = ans.decomp = Decomposition(c, rp.mpi_shape, p.shape);
    auto const truerank = get_true_rank(rp, c);

    outroot(truerank, "Initialising {} simulation from parameters...\n{}", ans.ID(), p);
    ans.checkpoint = cp;
    ans.params = p;

    auto is_local = [&](shape_t const& pos) {
        bool ans = true;
        for (int dim = 0; dim < 3; ++dim) {
            ans &= d.local_offset[dim] <= pos[dim];
            ans &= pos[dim] < (d.local_offset[dim] + d.local_shape[dim]);
        }
        return ans;
    };

    outroot(truerank, "Initial pressure field is pulse in middle of domain");
    ans.u = uField([&]{
        auto local_padded_shape = d.local_shape;
        std::for_each(local_padded_shape.begin(), local_padded_shape.end(), [](auto& _) {_ += 2;});
        auto u_init = array3d(local_padded_shape);
        std::fill_n(u_init.data(), u_init.size(), 0.0);

        auto set_maybe = [&](unsigned x, unsigned y, unsigned z, double val) {
            shape_t g = {x, y, z};
            if (is_local(g)) {
                u_init[g - d.local_offset + 1] = val;
            }
        };
        auto [hx, hy, hz] = p.shape / 2u;
        auto val = 50.0;
        set_maybe(hx, hy, hz, val);
        val /= -6.0;
        set_maybe(hx - 1, hy, hz, val);
        set_maybe(hx + 1, hy, hz, val);
        set_maybe(hx, hy - 1, hz, val);
        set_maybe(hx, hy + 1, hz, val);
        set_maybe(hx, hy, hz - 1, val);
        set_maybe(hx, hy, hz + 1, val);
        return u_init;
    }());

    auto [lx, ly, lz] = d.local_shape;
    auto [gx, gy, gz] = d.global_shape;
    // Speed of sound
    outroot(truerank, "Speed of sound is simple ocean depth model");
    ans.sos = array3d(d.local_shape);
    ans.cs2 = array3d(d.local_shape);
    for (unsigned lk = 0; lk < lz; ++lk) {
        auto depth = (d.local_offset[2] + lk) * ans.params.dx;
        auto cs = SpeedOfSoundProfile::convert(depth);
        for (unsigned li = 0; li < lx; ++li) {
            for (unsigned lj = 0; lj < ly; ++lj) {
                ans.sos(li, lj, lk) = cs;
                ans.cs2(li, lj, lk) = cs * cs;
            }
        }
    }

    outroot(truerank, "Damping field to avoid reflections in x & y directions (large quiet ocean)");
    ans.damp = array3d(d.local_shape);
    // Zero in the bulk
    std::fill_n(ans.damp.data(), ans.damp.size(), 0.0);
    auto nbl = ans.params.nBoundaryLayers;
    std::vector<double> ramp;
    ramp.reserve(nbl);
    for (int i = 0; i < nbl; ++i) {
        double r = nbl - i;
        ramp.push_back(9 * SpeedOfSoundProfile::MAX() * r * r / (2 * nbl * nbl * nbl * ans.params.dx));
    }
    // Damp only in x and y directions (silent, infinite ocean...)
    // Prep y cos multiply below for corners
    // global_i = 0..gx
    // local_i = full range
    for (unsigned li = 0; li < lx; ++li) {
        // global_j = 0..nbl
        for (unsigned gj = d.local_offset[1], lj = 0; gj < nbl && lj < ly; ++gj, ++lj)
            // But all local z
            std::fill_n(&ans.damp(li, lj, 0U), lz, 1.0);
        // gy-nbl..gy
        for (unsigned lj = gy - nbl - d.local_offset[1]; lj < ly; ++lj)
            std::fill_n(&ans.damp(li, lj, 0U), lz, 1.0);
    }
    // Set x
    for (unsigned gi = d.local_offset[0], li = 0; gi < nbl && li < lx; ++gi, ++li)
        std::fill_n(&ans.damp(li, 0u, 0u), ly * lz, ramp[gi]);
    for (unsigned li = gx - nbl - d.local_offset[0]; li < lx; ++li)
        std::fill_n(&ans.damp(li, 0u, 0u), ly * lz, ramp[gx - (li + d.local_offset[0]) - 1]);

    // y - multiplying hence set to 1 above
    for (unsigned li = 0; li < lx; ++li) {
        auto y_slice_scaler = [&](unsigned lj, double r) {
            double v = ans.damp(li, lj, 0u) * r;
            std::fill_n(&ans.damp(li, lj, 0u), lz, v);
        };
        for (unsigned gj = d.local_offset[1], lj = 0; gj < nbl && lj < ly; ++gj, ++lj)
            y_slice_scaler(lj, ramp[gj]);
        for (unsigned lj = gy - nbl - d.local_offset[1]; lj < ly; ++lj)
            y_slice_scaler(lj, ramp[gy - (lj + d.local_offset[1]) - 1]);
    }

    if (rp.doIO) {
        outroot(truerank, "Write initial conditions to {}", ans.checkpoint.c_str());
        ans.h5 = H5IO::from_params(ans.checkpoint, ans.params, ans.decomp);
        ans.h5.put_params(ans.params);
        ans.h5.put_damp(ans.damp);
        ans.h5.put_sos(ans.sos);
        ans.append_u_fields();
    } else {
        outroot(truerank, "IO off, skipping");
    }
    return ans;
}

void CpuWaveSimulation::append_u_fields() {
    if (h5) {
        h5.append_u(u);
    }
}

static void step(
    Decomposition const& decomp, Params const& params,
    view3d const cs2, view3d const damp,
    view3d const u_prev, view3d const u_now, view3d u_next
) {
    auto d2 = params.dx * params.dx;
    auto dt = params.dt;
    auto factor = dt*dt / d2;
    auto [nx, ny, nz] = decomp.local_shape;
    for (unsigned i = 0; i < nx; ++i) {
        auto ii = i + 1;
        for (unsigned j = 0; j < ny; ++j) {
            auto jj = j + 1;
            for (unsigned k = 0; k < nz; ++k) {
                auto kk = k + 1;
                // Simple approximation of Laplacian
                auto value = factor * cs2(i, j, k) * (
                        u_now(ii - 1, jj, kk) + u_now(ii + 1, jj, kk) +
                        u_now(ii, jj - 1, kk) + u_now(ii, jj + 1, kk) +
                        u_now(ii, jj, kk - 1) + u_now(ii, jj, kk + 1)
                        - 6.0 * u_now(ii, jj, kk)
                );
                // Deal with the damping field
                auto& d = damp(i, j, k);
                if (d == 0.0) {
                    u_next(ii, jj, kk) = 2.0 * u_now(ii, jj, kk) - u_prev(ii, jj, kk) + value;
                } else {
                    auto inv_denominator = 1.0 / (1.0 + d * dt);
                    auto numerator = 1.0 - d * dt;
                    value *= inv_denominator;
                    u_next(ii, jj, kk) = 2.0 * inv_denominator * u_now(ii, jj, kk) -
                                             numerator * inv_denominator * u_prev(ii, jj, kk) + value;
                }
            }
        }
    }
}

static void halo_exchange(Decomposition const& d, view3d field) {
    // We have up to six neighbours
    // Order them: +x, -x, +y, -y, +z, -z
    // In each case have a receive and send
    std::array<MPI_Request, 12> reqs;
    // This points to the next free request
    MPI_Request* next_req = reqs.data();

    // Shorter names
    auto& [px, py, pz] = d.mpi_shape;
    auto& [pi, pj, pk] = d.mpi_idx;
    // unpadded
    auto& [nx, ny, nz] = d.local_shape;
    // padded
    auto& [NX, NY, NZ] = field.shape();

    // In each case, want to send our last active face before the halo
    // and receive the neighbour's face into the halo.
    if (pi < px - 1) {
        // Have neighbour +x
        auto neigh = d.rank_layout(pi+1, pj, pk);
        MPI_Irecv(&field(nx+1u, 1u, 1u), 1, d.types->x, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(nx, 1u, 1u), 1, d.types->x, neigh, 0, d.comm, next_req++);
    }
    if (pi > 0) {
        // Have neighbour -x
        auto neigh = d.rank_layout(pi-1, pj, pk);
        MPI_Irecv(&field(0u, 1u, 1u), 1, d.types->x, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(1u, 1u, 1u), 1, d.types->x, neigh, 0, d.comm, next_req++);
    }

    if (pj < py - 1) {
        // Have neighbour +y
        auto neigh = d.rank_layout(pi, pj+1, pk);
        MPI_Irecv(&field(1u, ny+1u, 1u), 1, d.types->y, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(1u, ny, 1u), 1, d.types->y, neigh, 0, d.comm, next_req++);
    }
    if (pj > 0) {
        // Have neighbour -y
        auto neigh = d.rank_layout(pi, pj-1, pk);
        MPI_Irecv(&field(1u, 0u, 1u), 1, d.types->y, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(1u, 1u, 1u), 1, d.types->y, neigh, 0, d.comm, next_req++);
    }

    if (pk < pz - 1) {
        // Have neighbour +z
        auto neigh = d.rank_layout(pi, pj, pk+1);
        MPI_Irecv(&field(0u, 0u, nz+1u), 1, d.types->z, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(0u, 0u, nz), 1, d.types->z, neigh, 0, d.comm, next_req++);
    }
    if (pk > 0) {
        // Have neighbour -z
        auto neigh = d.rank_layout(pi, pj, pk-1);
        MPI_Irecv(&field(0u, 0u, 0u), 1, d.types->z, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(0u, 0u, 1u), 1, d.types->z, neigh, 0, d.comm, next_req++);
    }

    // Now wait on all the requests created
    auto nreqs = next_req - reqs.data();
    MPI_Waitall(nreqs, reqs.data(), MPI_STATUSES_IGNORE);
}


void CpuWaveSimulation::run(int n) {
    for (int i = 0; i < n; ++i) {
        halo_exchange(decomp, u.now());
        step(decomp, params, cs2, damp, u.prev(), u.now(), u.next());
        u.advance();
    }
}
