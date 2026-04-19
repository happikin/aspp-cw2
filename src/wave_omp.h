// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_WAVE_OMP_H
#define AWAVE_WAVE_OMP_H

#include "wave_cpu.h"

struct OmpImplementationData;

/// All data required to run the wave propagation simulation on GPU
struct OmpWaveSimulation : WaveSimulation {
    inline const char * ID() const override {
        return "OpenMP";
    }

    // Pointer to any data needed for OpenMP implementation
    std::unique_ptr<OmpImplementationData> impl;

    // Initialise the object as a copy of a CPU state
    static OmpWaveSimulation from_cpu_sim(fs::path const& cp, WaveSimulation const& source);

    // Because we want to hide the definition of the implementation
    // struct, we need to declare these special members. They will be
    // defaulted in the implementation.
    OmpWaveSimulation();
    OmpWaveSimulation(OmpWaveSimulation&&) noexcept;
    OmpWaveSimulation& operator=(OmpWaveSimulation&&);
    ~OmpWaveSimulation();

    // Write the current and previous u fields to the output
    void append_u_fields() override;
    // Run `n` steps`
    void run(int n) override;
};

#endif
