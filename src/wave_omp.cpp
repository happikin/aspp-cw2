//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_omp.h"

// This struct can hold any data you need to manage running on the device
//
// Allocate with std::make_unique when you create the simulation
// object below, in `from_cpu_sim`.
struct OmpImplementationData {
    // Add any data members you need here

    OmpImplementationData() {
    }
    ~OmpImplementationData() {
   }
};

OmpWaveSimulation::OmpWaveSimulation() = default;
OmpWaveSimulation::OmpWaveSimulation(OmpWaveSimulation&&)  noexcept = default;
OmpWaveSimulation& OmpWaveSimulation::operator=(OmpWaveSimulation&&) = default;
OmpWaveSimulation::~OmpWaveSimulation() = default;

OmpWaveSimulation OmpWaveSimulation::from_cpu_sim(const fs::path& cp, const WaveSimulation& source) {
    OmpWaveSimulation ans;
    auto rank = source.decomp.rank;
    outroot(rank, "Initialising {} simulation as copy of {}...", ans.ID(), source.ID());
    ans.params = source.params;
    ans.decomp = source.decomp;
    ans.u = source.u.clone();
    ans.sos = source.sos.clone();
    ans.cs2 = source.cs2.clone();
    ans.damp = source.damp.clone();

    ans.checkpoint = cp;

    if (source.h5) {
        ans.h5 = H5IO::from_params(cp, ans.params, ans.decomp);
        outroot(rank, "Write initial conditions to {}", ans.checkpoint.c_str());
        ans.h5.put_params(ans.params);
        ans.h5.put_damp(ans.damp);
        ans.h5.put_sos(ans.sos);
        ans.append_u_fields();
    } else {
        outroot(rank, "IO off, skipping");
    }
    // Perhaps you want to do some device set up now?
    // ans.impl = std::make_unique<OmpImplementationData>();

    return ans;
}

void OmpWaveSimulation::append_u_fields() {
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

void OmpWaveSimulation::run(int n) {
    for (int i = 0; i < n; ++i) {
        halo_exchange(decomp, u.now());
        step(decomp, params, cs2, damp, u.prev(), u.now(), u.next());
        u.advance();
    }
}
