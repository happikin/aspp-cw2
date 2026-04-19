// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_PARALLEL_H
#define AWAVE_PARALLEL_H

#include <mpi.h>

#include "ndarray.h"
#include "params.h"

// Simple MPI RAII wrapper
class MpiEnv {
    bool owns_mpi = false;

public:
    MpiEnv() = default;
    // No copying
    MpiEnv(MpiEnv const&) = delete;
    MpiEnv& operator=(MpiEnv const&) = delete;

    // Construct from command line arguments
    // Requires MPI_THREAD_SERIALIZED to be available.
    MpiEnv(int& argc, char**& argv);

    // Finalise MPI iff this instance owns MPI
    ~MpiEnv();
};


// Describe how the problem is broken up over the MPI processes.
struct Decomposition {
    // MPI datatypes
    struct StridedTypes {
        MPI_Datatype x, y, z;
      ~StridedTypes();
    };

    // Exactly as the base, but copies with clone()
    class RankArray3d : public nd::array<int, 3> {
    public:
        using base = nd::array<int, 3>;
        using base::base;

        RankArray3d(RankArray3d const&);
        RankArray3d& operator=(RankArray3d const&);
    };

    // Communicator...
    MPI_Comm comm = MPI_COMM_NULL;
    // ...and it's size and shape
    int size;
    int rank;
    // Organisation of processes (note product of this must equal size)
    shape_t mpi_shape;
    // The ranks of all processes
    RankArray3d rank_layout;

    // Shape of the whole domain
    shape_t global_shape;
    // The size of the "default" block size
    shape_t block_size;
    // Shape of this process's sub domain
    shape_t local_shape;
    // Starting index of this process's sub domain
    shape_t local_offset;

    // 3d index of this process's subdomain.
    // need rank_layout[mpi_idx] == rank
    shape_t mpi_idx;

    // Pointer to all the MPI types used
    std::shared_ptr<StridedTypes> types;

    Decomposition() = default;
    Decomposition(MPI_Comm c, shape_t mpiShape, shape_t domShape);
};

#endif
