// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_cuda.h"

#include <array>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

// Free helper macro to check for CUDA errors!
#define CUDA_CHECK(expr) do { \
    cudaError_t res = expr; \
    if (res != cudaSuccess) \
      throw std::runtime_error(std::format(__FILE__ ":{} CUDA error: {}", __LINE__, cudaGetErrorString(res))); \
  } while (0)


// This struct can hold any data you need to manage running on the device
//
// Allocate with std::make_unique when you create the simulation
// object below, in `from_cpu_sim`.
struct CudaImplementationData {
    enum Dir : int {XP = 0, XM = 1, YP = 2, YM = 3, ZP = 4, ZM = 5, NDIR = 6};

    int device = 0;
    std::size_t n_u = 0;
    std::size_t n_field = 0;
    unsigned nx = 0;
    unsigned ny = 0;
    unsigned nz = 0;
    unsigned NY = 0;
    unsigned NZ = 0;
    double* d_u_prev = nullptr;
    double* d_u_now = nullptr;
    double* d_u_next = nullptr;
    double* d_cs2 = nullptr;
    double* d_damp = nullptr;
    std::array<int, NDIR> neigh = {};
    std::array<int, NDIR> face_count = {};
    std::array<double*, NDIR> d_send = {};
    std::array<double*, NDIR> d_recv = {};
    std::array<double*, NDIR> h_send = {};
    std::array<double*, NDIR> h_recv = {};
    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
    cudaEvent_t halo_ready = nullptr;

    CudaImplementationData() {
        nvtx3::scoped_range r{"initialise"};
    }
    ~CudaImplementationData() {
        if (d_u_prev || d_u_now || d_u_next || d_cs2 || d_damp) {
            CUDA_CHECK(cudaSetDevice(device));
        }
        if (halo_ready) CUDA_CHECK(cudaEventDestroy(halo_ready));
        if (compute_stream) CUDA_CHECK(cudaStreamDestroy(compute_stream));
        if (comm_stream) CUDA_CHECK(cudaStreamDestroy(comm_stream));
        for (int d = 0; d < NDIR; ++d) {
            if (d_send[d]) CUDA_CHECK(cudaFree(d_send[d]));
            if (d_recv[d]) CUDA_CHECK(cudaFree(d_recv[d]));
            if (h_send[d]) CUDA_CHECK(cudaFreeHost(h_send[d]));
            if (h_recv[d]) CUDA_CHECK(cudaFreeHost(h_recv[d]));
        }
        if (d_u_prev) CUDA_CHECK(cudaFree(d_u_prev));
        if (d_u_now) CUDA_CHECK(cudaFree(d_u_now));
        if (d_u_next) CUDA_CHECK(cudaFree(d_u_next));
        if (d_cs2) CUDA_CHECK(cudaFree(d_cs2));
        if (d_damp) CUDA_CHECK(cudaFree(d_damp));
   }
};

static int local_rank(MPI_Comm comm) {
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local = 0;
    MPI_Comm_rank(local_comm, &local);
    MPI_Comm_free(&local_comm);
    return local;
}

__global__ static void step_interior_kernel(
    unsigned nx, unsigned ny, unsigned nz,
    unsigned NY, unsigned NZ,
    double factor, double dt,
    double const* __restrict__ cs2,
    double const* __restrict__ damp,
    double const* __restrict__ u_prev,
    double const* __restrict__ u_now,
    double* __restrict__ u_next
) {
    auto i = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
    auto j = unsigned(blockIdx.y * blockDim.y + threadIdx.y);
    auto k = unsigned(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny || k >= nz) return;
    if (!(i > 0 && i + 1 < nx && j > 0 && j + 1 < ny && k > 0 && k + 1 < nz)) return;

    auto const stride_x = std::size_t(NY) * NZ;
    auto const stride_y = std::size_t(NZ);

    auto const fi = std::size_t(i) * ny * nz + std::size_t(j) * nz + k;
    auto const ui = std::size_t(i + 1) * stride_x + std::size_t(j + 1) * stride_y + (k + 1);

    auto value = factor * cs2[fi] * (
        u_now[ui - stride_x] + u_now[ui + stride_x] +
        u_now[ui - stride_y] + u_now[ui + stride_y] +
        u_now[ui - 1] + u_now[ui + 1]
        - 6.0 * u_now[ui]
    );

    auto d = damp[fi];
    if (d == 0.0) {
        u_next[ui] = 2.0 * u_now[ui] - u_prev[ui] + value;
    } else {
        auto inv_denominator = 1.0 / (1.0 + d * dt);
        auto numerator = 1.0 - d * dt;
        value *= inv_denominator;
        u_next[ui] = 2.0 * inv_denominator * u_now[ui] -
                     numerator * inv_denominator * u_prev[ui] + value;
    }
}

__global__ static void step_boundary_kernel(
    unsigned nx, unsigned ny, unsigned nz,
    unsigned NY, unsigned NZ,
    double factor, double dt,
    double const* __restrict__ cs2,
    double const* __restrict__ damp,
    double const* __restrict__ u_prev,
    double const* __restrict__ u_now,
    double* __restrict__ u_next
) {
    auto i = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
    auto j = unsigned(blockIdx.y * blockDim.y + threadIdx.y);
    auto k = unsigned(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny || k >= nz) return;
    if (!(i == 0 || i + 1 == nx || j == 0 || j + 1 == ny || k == 0 || k + 1 == nz)) return;

    auto const stride_x = std::size_t(NY) * NZ;
    auto const stride_y = std::size_t(NZ);

    auto const fi = std::size_t(i) * ny * nz + std::size_t(j) * nz + k;
    auto const ui = std::size_t(i + 1) * stride_x + std::size_t(j + 1) * stride_y + (k + 1);

    auto value = factor * cs2[fi] * (
        u_now[ui - stride_x] + u_now[ui + stride_x] +
        u_now[ui - stride_y] + u_now[ui + stride_y] +
        u_now[ui - 1] + u_now[ui + 1]
        - 6.0 * u_now[ui]
    );

    auto d = damp[fi];
    if (d == 0.0) {
        u_next[ui] = 2.0 * u_now[ui] - u_prev[ui] + value;
    } else {
        auto inv_denominator = 1.0 / (1.0 + d * dt);
        auto numerator = 1.0 - d * dt;
        value *= inv_denominator;
        u_next[ui] = 2.0 * inv_denominator * u_now[ui] -
                     numerator * inv_denominator * u_prev[ui] + value;
    }
}

__global__ static void pack_x_face_kernel(
    unsigned ny, unsigned nz, unsigned NY, unsigned NZ, unsigned i_plane,
    double const* __restrict__ u_now, double* __restrict__ face
) {
    auto t = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
    auto n = ny * nz;
    if (t >= n) return;
    auto j = t / nz;
    auto k = t % nz;
    auto ui = (std::size_t(i_plane) * NY + (j + 1)) * NZ + (k + 1);
    face[t] = u_now[ui];
}

__global__ static void unpack_x_face_kernel(
    unsigned ny, unsigned nz, unsigned NY, unsigned NZ, unsigned i_plane,
    double const* __restrict__ face, double* __restrict__ u_now
) {
    auto t = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
    auto n = ny * nz;
    if (t >= n) return;
    auto j = t / nz;
    auto k = t % nz;
    auto ui = (std::size_t(i_plane) * NY + (j + 1)) * NZ + (k + 1);
    u_now[ui] = face[t];
}

__global__ static void pack_y_face_kernel(
    unsigned nx, unsigned nz, unsigned NY, unsigned NZ, unsigned j_plane,
    double const* __restrict__ u_now, double* __restrict__ face
) {
    auto t = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
    auto n = nx * nz;
    if (t >= n) return;
    auto i = t / nz;
    auto k = t % nz;
    auto ui = (std::size_t(i + 1) * NY + j_plane) * NZ + (k + 1);
    face[t] = u_now[ui];
}

__global__ static void unpack_y_face_kernel(
    unsigned nx, unsigned nz, unsigned NY, unsigned NZ, unsigned j_plane,
    double const* __restrict__ face, double* __restrict__ u_now
) {
    auto t = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
    auto n = nx * nz;
    if (t >= n) return;
    auto i = t / nz;
    auto k = t % nz;
    auto ui = (std::size_t(i + 1) * NY + j_plane) * NZ + (k + 1);
    u_now[ui] = face[t];
}

__global__ static void pack_z_face_kernel(
    unsigned nx, unsigned ny, unsigned NY, unsigned NZ, unsigned k_plane,
    double const* __restrict__ u_now, double* __restrict__ face
) {
    auto t = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
    auto n = nx * ny;
    if (t >= n) return;
    auto i = t / ny;
    auto j = t % ny;
    auto ui = (std::size_t(i + 1) * NY + (j + 1)) * NZ + k_plane;
    face[t] = u_now[ui];
}

__global__ static void unpack_z_face_kernel(
    unsigned nx, unsigned ny, unsigned NY, unsigned NZ, unsigned k_plane,
    double const* __restrict__ face, double* __restrict__ u_now
) {
    auto t = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
    auto n = nx * ny;
    if (t >= n) return;
    auto i = t / ny;
    auto j = t % ny;
    auto ui = (std::size_t(i + 1) * NY + (j + 1)) * NZ + k_plane;
    u_now[ui] = face[t];
}

static void pack_faces_to_host(Decomposition const&, CudaImplementationData& impl) {
    constexpr unsigned block = 256;
    auto launch_1d = [&](unsigned n, auto&& fn) {
        if (n == 0) return;
        dim3 grid(ceildiv(n, block));
        fn(grid);
    };

    if (impl.neigh[CudaImplementationData::XP] >= 0) {
        launch_1d(impl.ny * impl.nz, [&](dim3 grid) {
            pack_x_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.ny, impl.nz, impl.NY, impl.NZ, impl.nx, impl.d_u_now, impl.d_send[CudaImplementationData::XP]);
        });
    }
    if (impl.neigh[CudaImplementationData::XM] >= 0) {
        launch_1d(impl.ny * impl.nz, [&](dim3 grid) {
            pack_x_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.ny, impl.nz, impl.NY, impl.NZ, 1u, impl.d_u_now, impl.d_send[CudaImplementationData::XM]);
        });
    }
    if (impl.neigh[CudaImplementationData::YP] >= 0) {
        launch_1d(impl.nx * impl.nz, [&](dim3 grid) {
            pack_y_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.nx, impl.nz, impl.NY, impl.NZ, impl.ny, impl.d_u_now, impl.d_send[CudaImplementationData::YP]);
        });
    }
    if (impl.neigh[CudaImplementationData::YM] >= 0) {
        launch_1d(impl.nx * impl.nz, [&](dim3 grid) {
            pack_y_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.nx, impl.nz, impl.NY, impl.NZ, 1u, impl.d_u_now, impl.d_send[CudaImplementationData::YM]);
        });
    }
    if (impl.neigh[CudaImplementationData::ZP] >= 0) {
        launch_1d(impl.nx * impl.ny, [&](dim3 grid) {
            pack_z_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.nx, impl.ny, impl.NY, impl.NZ, impl.nz, impl.d_u_now, impl.d_send[CudaImplementationData::ZP]);
        });
    }
    if (impl.neigh[CudaImplementationData::ZM] >= 0) {
        launch_1d(impl.nx * impl.ny, [&](dim3 grid) {
            pack_z_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.nx, impl.ny, impl.NY, impl.NZ, 1u, impl.d_u_now, impl.d_send[CudaImplementationData::ZM]);
        });
    }

    CUDA_CHECK(cudaGetLastError());

    for (int dir = 0; dir < CudaImplementationData::NDIR; ++dir) {
        auto n = impl.face_count[dir];
        if (impl.neigh[dir] < 0 || n == 0) continue;
        CUDA_CHECK(cudaMemcpyAsync(
            impl.h_send[dir], impl.d_send[dir], std::size_t(n) * sizeof(double),
            cudaMemcpyDeviceToHost, impl.comm_stream
        ));
    }
}

static void unpack_faces_from_host(Decomposition const&, CudaImplementationData& impl) {
    for (int dir = 0; dir < CudaImplementationData::NDIR; ++dir) {
        auto n = impl.face_count[dir];
        if (impl.neigh[dir] < 0 || n == 0) continue;
        CUDA_CHECK(cudaMemcpyAsync(
            impl.d_recv[dir], impl.h_recv[dir], std::size_t(n) * sizeof(double),
            cudaMemcpyHostToDevice, impl.comm_stream
        ));
    }

    constexpr unsigned block = 256;
    auto launch_1d = [&](unsigned n, auto&& fn) {
        if (n == 0) return;
        dim3 grid(ceildiv(n, block));
        fn(grid);
    };

    if (impl.neigh[CudaImplementationData::XP] >= 0) {
        launch_1d(impl.ny * impl.nz, [&](dim3 grid) {
            unpack_x_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.ny, impl.nz, impl.NY, impl.NZ, impl.nx + 1u, impl.d_recv[CudaImplementationData::XP], impl.d_u_now);
        });
    }
    if (impl.neigh[CudaImplementationData::XM] >= 0) {
        launch_1d(impl.ny * impl.nz, [&](dim3 grid) {
            unpack_x_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.ny, impl.nz, impl.NY, impl.NZ, 0u, impl.d_recv[CudaImplementationData::XM], impl.d_u_now);
        });
    }
    if (impl.neigh[CudaImplementationData::YP] >= 0) {
        launch_1d(impl.nx * impl.nz, [&](dim3 grid) {
            unpack_y_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.nx, impl.nz, impl.NY, impl.NZ, impl.ny + 1u, impl.d_recv[CudaImplementationData::YP], impl.d_u_now);
        });
    }
    if (impl.neigh[CudaImplementationData::YM] >= 0) {
        launch_1d(impl.nx * impl.nz, [&](dim3 grid) {
            unpack_y_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.nx, impl.nz, impl.NY, impl.NZ, 0u, impl.d_recv[CudaImplementationData::YM], impl.d_u_now);
        });
    }
    if (impl.neigh[CudaImplementationData::ZP] >= 0) {
        launch_1d(impl.nx * impl.ny, [&](dim3 grid) {
            unpack_z_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.nx, impl.ny, impl.NY, impl.NZ, impl.nz + 1u, impl.d_recv[CudaImplementationData::ZP], impl.d_u_now);
        });
    }
    if (impl.neigh[CudaImplementationData::ZM] >= 0) {
        launch_1d(impl.nx * impl.ny, [&](dim3 grid) {
            unpack_z_face_kernel<<<grid, block, 0, impl.comm_stream>>>(impl.nx, impl.ny, impl.NY, impl.NZ, 0u, impl.d_recv[CudaImplementationData::ZM], impl.d_u_now);
        });
    }

    CUDA_CHECK(cudaGetLastError());
}

static void halo_exchange_faces(Decomposition const& d, CudaImplementationData& impl) {
    pack_faces_to_host(d, impl);
    CUDA_CHECK(cudaStreamSynchronize(impl.comm_stream));

    std::array<MPI_Request, 12> reqs;
    MPI_Request* next_req = reqs.data();
    for (int dir = 0; dir < CudaImplementationData::NDIR; ++dir) {
        if (impl.neigh[dir] < 0 || impl.face_count[dir] == 0) continue;
        MPI_Irecv(impl.h_recv[dir], impl.face_count[dir], MPI_DOUBLE, impl.neigh[dir], 0, d.comm, next_req++);
        MPI_Isend(impl.h_send[dir], impl.face_count[dir], MPI_DOUBLE, impl.neigh[dir], 0, d.comm, next_req++);
    }
    auto nreqs = int(next_req - reqs.data());
    if (nreqs > 0) {
        MPI_Waitall(nreqs, reqs.data(), MPI_STATUSES_IGNORE);
    }

    unpack_faces_from_host(d, impl);
    CUDA_CHECK(cudaEventRecord(impl.halo_ready, impl.comm_stream));
}

CudaWaveSimulation::CudaWaveSimulation() = default;
CudaWaveSimulation::CudaWaveSimulation(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation& CudaWaveSimulation::operator=(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation::~CudaWaveSimulation() = default;

CudaWaveSimulation CudaWaveSimulation::from_cpu_sim(const fs::path& cp, const WaveSimulation& source) {
    CudaWaveSimulation ans;
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
    ans.impl = std::make_unique<CudaImplementationData>();
    auto& impl = *ans.impl;

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) {
        throw std::runtime_error("No CUDA devices available");
    }
    impl.device = local_rank(ans.decomp.comm) % ndev;
    CUDA_CHECK(cudaSetDevice(impl.device));
    CUDA_CHECK(cudaStreamCreate(&impl.compute_stream));
    CUDA_CHECK(cudaStreamCreate(&impl.comm_stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&impl.halo_ready, cudaEventDisableTiming));

    impl.n_u = ans.u.now().size();
    impl.n_field = ans.cs2.size();
    impl.nx = unsigned(ans.decomp.local_shape[0]);
    impl.ny = unsigned(ans.decomp.local_shape[1]);
    impl.nz = unsigned(ans.decomp.local_shape[2]);
    impl.NY = unsigned(ans.u.now().shape()[1]);
    impl.NZ = unsigned(ans.u.now().shape()[2]);
    auto bytes_u = impl.n_u * sizeof(double);
    auto bytes_field = impl.n_field * sizeof(double);

    CUDA_CHECK(cudaMalloc(&impl.d_u_prev, bytes_u));
    CUDA_CHECK(cudaMalloc(&impl.d_u_now, bytes_u));
    CUDA_CHECK(cudaMalloc(&impl.d_u_next, bytes_u));
    CUDA_CHECK(cudaMalloc(&impl.d_cs2, bytes_field));
    CUDA_CHECK(cudaMalloc(&impl.d_damp, bytes_field));

    CUDA_CHECK(cudaMemcpy(impl.d_u_prev, ans.u.prev().data(), bytes_u, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl.d_u_now, ans.u.now().data(), bytes_u, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl.d_u_next, ans.u.next().data(), bytes_u, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl.d_cs2, ans.cs2.data(), bytes_field, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl.d_damp, ans.damp.data(), bytes_field, cudaMemcpyHostToDevice));

    auto& [px, py, pz] = ans.decomp.mpi_shape;
    auto& [pi, pj, pk] = ans.decomp.mpi_idx;

    impl.neigh[CudaImplementationData::XP] = (pi < px - 1) ? ans.decomp.rank_layout(pi + 1, pj, pk) : -1;
    impl.neigh[CudaImplementationData::XM] = (pi > 0) ? ans.decomp.rank_layout(pi - 1, pj, pk) : -1;
    impl.neigh[CudaImplementationData::YP] = (pj < py - 1) ? ans.decomp.rank_layout(pi, pj + 1, pk) : -1;
    impl.neigh[CudaImplementationData::YM] = (pj > 0) ? ans.decomp.rank_layout(pi, pj - 1, pk) : -1;
    impl.neigh[CudaImplementationData::ZP] = (pk < pz - 1) ? ans.decomp.rank_layout(pi, pj, pk + 1) : -1;
    impl.neigh[CudaImplementationData::ZM] = (pk > 0) ? ans.decomp.rank_layout(pi, pj, pk - 1) : -1;

    impl.face_count[CudaImplementationData::XP] = int(impl.ny * impl.nz);
    impl.face_count[CudaImplementationData::XM] = int(impl.ny * impl.nz);
    impl.face_count[CudaImplementationData::YP] = int(impl.nx * impl.nz);
    impl.face_count[CudaImplementationData::YM] = int(impl.nx * impl.nz);
    impl.face_count[CudaImplementationData::ZP] = int(impl.nx * impl.ny);
    impl.face_count[CudaImplementationData::ZM] = int(impl.nx * impl.ny);

    for (int dir = 0; dir < CudaImplementationData::NDIR; ++dir) {
        if (impl.neigh[dir] < 0 || impl.face_count[dir] == 0) continue;
        auto nbytes = std::size_t(impl.face_count[dir]) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&impl.d_send[dir], nbytes));
        CUDA_CHECK(cudaMalloc(&impl.d_recv[dir], nbytes));
        CUDA_CHECK(cudaHostAlloc(&impl.h_send[dir], nbytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&impl.h_recv[dir], nbytes, cudaHostAllocDefault));
    }

    return ans;
}

void CudaWaveSimulation::append_u_fields() {
    if (h5) {
        h5.append_u(u);
    }
}

void CudaWaveSimulation::run(int n) {
    nvtx3::scoped_range r{"run"};
    auto& impl = *this->impl;
    CUDA_CHECK(cudaSetDevice(impl.device));

    auto [nx, ny, nz] = decomp.local_shape;
    auto bytes_u = impl.n_u * sizeof(double);

    auto d2 = params.dx * params.dx;
    auto dt = params.dt;
    auto factor = dt * dt / d2;

    dim3 block(8, 8, 4);
    dim3 grid(ceildiv(nx, block.x), ceildiv(ny, block.y), ceildiv(nz, block.z));

    for (int i = 0; i < n; ++i) {
        step_interior_kernel<<<grid, block, 0, impl.compute_stream>>>(
            nx, ny, nz, impl.NY, impl.NZ, factor, dt,
            impl.d_cs2, impl.d_damp, impl.d_u_prev, impl.d_u_now, impl.d_u_next
        );
        CUDA_CHECK(cudaGetLastError());

        halo_exchange_faces(decomp, impl);
        CUDA_CHECK(cudaStreamWaitEvent(impl.compute_stream, impl.halo_ready, 0));

        step_boundary_kernel<<<grid, block, 0, impl.compute_stream>>>(
            nx, ny, nz, impl.NY, impl.NZ, factor, dt,
            impl.d_cs2, impl.d_damp, impl.d_u_prev, impl.d_u_now, impl.d_u_next
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(impl.compute_stream));

        std::swap(impl.d_u_prev, impl.d_u_now);
        std::swap(impl.d_u_now, impl.d_u_next);
        u.advance();
    }
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(u.prev().data(), impl.d_u_prev, bytes_u, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(u.now().data(), impl.d_u_now, bytes_u, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
