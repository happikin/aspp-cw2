// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
// This contains a minimal wrapper around the HDF5 API (namespace
// h5) and functionality to use this for our problem.
#ifndef AWAVE_H5IO_H
#define AWAVE_H5IO_H

#include <filesystem>
#include <memory>

#include "params.h"
#include "parallel.h"

namespace fs = std::filesystem;
class uField;

// Going to use the PIMPL idiom (pointer to implementation)
namespace h5 {
    struct Impl;
}

// Control writing data to VTK compatible HDF5 file.
// Should be able to open in Paraview to inspect.
class H5IO {
    std::unique_ptr<h5::Impl> m_impl;

public:
    /// Create the object by opening an existing checkpoint file
    static H5IO from_params(fs::path const& path, Params const& params, Decomposition const& d);
    /// Create one for reading an existing cp
    static H5IO read_only(const fs::path& path, MPI_Comm c, shape_t mpiShape);

    // Since hiding implementation, we declare here and default in .cpp
    H5IO();
    H5IO(H5IO&&) noexcept;
    H5IO& operator=(H5IO&&) noexcept;
    ~H5IO();

    operator bool() const {
        return m_impl.get() != nullptr;
    }

    void put_params(Params const& p);
    Params get_params();

    void put_damp(array3d const&);
    void put_sos(array3d const&);
    array3d get_damp();
    array3d get_sos();

    void append_u(uField const& u);
    uField get_last_u();
};
#endif
