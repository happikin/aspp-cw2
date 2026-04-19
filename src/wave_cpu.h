// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_WAVE_CPU_H
#define AWAVE_WAVE_CPU_H

#include <filesystem>

#include "ndarray.h"
#include "h5io.h"
#include "params.h"
#include "ufield.h"

namespace fs = std::filesystem;

/// Abstract base class for simulations.
///
/// Holds data required to run the wave propagation simulation on CPU
/// This would normally be a class (i.e. default private data members),
/// but to simplify we are making all members public
///
/// Subclasses must override the virtual member functions.
struct WaveSimulation {
    Params params;

    // Hold parallel information
    Decomposition decomp;

    // NOTE all fields hold process-local fields

    // The time-dependent pressure (offset) field
    uField u;
    // Speed of sound (not needed for computation)
    array3d sos;
    // Speed of sound squared (actually needed)
    array3d cs2;
    // Field to damp at the boundaries
    array3d damp;

    // Path for output
    fs::path checkpoint;
    // HDF5 output management object
    H5IO h5;

    virtual char const* ID() const = 0;
    // Write the current and previous u fields to the output
    virtual void append_u_fields() = 0;
    // Run `n` steps`
    virtual void run(int n) = 0;
};

struct CpuWaveSimulation : WaveSimulation {
    inline const char * ID() const override {
        return "CPU";
    }
    // Initialise the object from a checkpoint file.
    static CpuWaveSimulation from_file(MPI_Comm c, RunParams const& rp, fs::path const& dest, fs::path const& src, int nsteps);
    // Initialise to a simple case based on the command line parameters
    static CpuWaveSimulation from_params(MPI_Comm c, RunParams const& rp, fs::path const& cp, Params const& p);

    void append_u_fields() override;
    void run(int n) override;
};
#endif
