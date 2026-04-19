// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_WAVE_SYCL_H
#define AWAVE_WAVE_SYCL_H

#include "wave_cpu.h"

struct SyclImplementationData;

/// All data required to run the wave propagation simulation on GPU
struct SyclWaveSimulation : WaveSimulation {
    inline const char * ID() const override {
        return "OpenMP";
    }

    // Pointer to any data needed for OpenMP implementation
    std::unique_ptr<SyclImplementationData> impl;

    // Initialise the object as a copy of a CPU state
    static SyclWaveSimulation from_cpu_sim(fs::path const& cp, WaveSimulation const& source);

    // Because we want to hide the definition of the implementation
    // struct, we need to declare these special members. They will be
    // defaulted in the implementation.
    SyclWaveSimulation();
    SyclWaveSimulation(SyclWaveSimulation&&) noexcept;
    SyclWaveSimulation& operator=(SyclWaveSimulation&&);
    ~SyclWaveSimulation();

    // Write the current and previous u fields to the output
    void append_u_fields() override;
    // Run `n` steps`
    void run(int n) override;
};

#endif
