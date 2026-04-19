//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "parallel.h"

#include <format>

#include "util.h"

template <typename... Args>
auto runerr(std::format_string<Args...> f, Args... args) {
    return std::runtime_error(std::format(f, std::forward<Args>(args)...));
}

// Construct from command line arguments
// Requires MPI_THREAD_SERIALIZED to be available.
MpiEnv::MpiEnv(int& argc, char**& argv) {
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
        owns_mpi = true;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &flag);
        if (flag < MPI_THREAD_SERIALIZED) {
            throw std::runtime_error("MPI_THREAD_SERIALIZED not available");
        }
    }
}

// Finalise MPI iff this instance owns MPI
MpiEnv::~MpiEnv() {
    if (owns_mpi) {
        MPI_Finalize();
    }
}

Decomposition::RankArray3d::RankArray3d(RankArray3d const& src) : base(src.clone()) {
}

auto Decomposition::RankArray3d::operator=(RankArray3d const& src) -> RankArray3d& {
    if (this != &src) {
        auto tmp = RankArray3d(src);
        swap(tmp, *this);
    }
    return *this;
}

Decomposition::StridedTypes::~StridedTypes() {
    MPI_Type_free(&x);
    MPI_Type_free(&y);
    MPI_Type_free(&z);
}

Decomposition::Decomposition(MPI_Comm c, shape_t mpiShape, shape_t domShape) : comm{c}, mpi_shape{mpiShape}, rank_layout{mpiShape}, global_shape{domShape} {
    MPI_Comm_size(c, &size);
    MPI_Comm_rank(c, &rank);

    auto n_shape = std::reduce(mpiShape.begin(), mpiShape.end(), 1, std::multiplies<int>{});
    if (n_shape != size)
        throw runerr("MPI communicator has size ({}) that is incompatible with process shape ({})", size, mpiShape);

    // Fill the rank_layout
    std::iota(rank_layout.data(), rank_layout.data() + rank_layout.size(), 0);
    // Work out the block index from the rank and shape
    mpi_idx[2] = rank % mpi_shape[2];
    auto tmp = rank / mpi_shape[2];
    mpi_idx[1] = tmp % mpi_shape[1];
    mpi_idx[0] = tmp / mpi_shape[1];
    // Check we did it right...
    if (rank != (mpi_idx[0]*mpi_shape[1] + mpi_idx[1]) * mpi_shape[2] + mpi_idx[2])
        throw runerr("rank ({}) does not match 3d index ({}) and process shape ({})", rank, mpi_idx, mpi_shape);
    // ...and that the rank_layout is consistent
    if (rank != rank_layout[mpi_idx])
        throw runerr("rank ({}) does not match rank_layout[mpi_idx] ({})", rank, rank_layout[mpi_idx]);

    block_size = ceildiv(global_shape, mpi_shape);
    local_offset = block_size * mpi_idx;
    local_shape = min(block_size, global_shape - local_offset);

    // Check by summing along each dimension
    for (int dim=0; dim<3; ++dim) {
        std::size_t tot = 0;
        if (mpi_idx[dim] > 0) {
            auto neigh = mpi_idx;
            neigh[dim] -= 1;
            MPI_Recv(&tot, 1, MPI_UINT64_T, rank_layout[neigh], 0, c, MPI_STATUS_IGNORE);
        }
        tot += local_shape[dim];
        if (mpi_idx[dim] < mpi_shape[dim] -1) {
            auto neigh = mpi_idx;
            neigh[dim] += 1;
            MPI_Send(&tot, 1, MPI_UINT64_T, rank_layout[neigh], 0, c);
        } else {
            if (tot != global_shape[dim])
                throw runerr("Subdomain sizes don't add up in dimension {}", dim);
        }
    }

    // Create the datatypes
    // unpadded shape
    auto& [nx, ny, nz] = local_shape;
    // padded
    auto NX = nx + 2;
    auto NY = ny + 2;
    auto NZ = nz + 2;

    types = std::make_shared<StridedTypes>();
    // X: this would be easy and contiguous except for the halo
    // Pointer should be first value you want to send
    MPI_Type_vector(ny, nz, NZ, MPI_DOUBLE, &types->x);
    MPI_Type_commit(&types->x);
    // Y: without halos, this would be lots of columns, still is with the halos
    // Pointer as above
    MPI_Type_vector(nx, nz, NY*NZ, MPI_DOUBLE, &types->y);
    MPI_Type_commit(&types->y);
    // Z: this is scattered values
    // Pointer should point to element (0,0,j) if you want to send the
    // active values in plane j (i.e. data[1:-1, 1:-1, j] in numpy array
    // slice syntax.
    constexpr int ND = 3;
    int const big[ND] = {int(NX), int(NY), int(NZ)};
    int const sub[ND] = {int(nx), int(ny), 1};
    int const off[ND] = {1, 1, 0};
    MPI_Type_create_subarray(ND, big, sub, off, MPI_ORDER_C, MPI_DOUBLE, &types->z);
    MPI_Type_commit(&types->z);
}
