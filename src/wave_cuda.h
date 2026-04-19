// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_WAVE_CUDA_H
#define AWAVE_WAVE_CUDA_H

#include "wave_cpu.h"

struct CudaImplementationData;

/// All data required to run the wave propagation simulation on GPU
struct CudaWaveSimulation : WaveSimulation {
    inline const char * ID() const override {
        return "CUDA";
    }
    // Pointer to any data needed for CUDA implementation
    std::unique_ptr<CudaImplementationData> impl;

    // Initialise the object as a copy of a CPU state
    static CudaWaveSimulation from_cpu_sim(fs::path const& cp, WaveSimulation const& source);

    // Because we want to hide the definition of the implementation
    // struct, we need to declare these special members. They will be
    // defaulted in the implementation.
    CudaWaveSimulation();
    CudaWaveSimulation(CudaWaveSimulation&&) noexcept;
    CudaWaveSimulation& operator=(CudaWaveSimulation&&) noexcept;
    ~CudaWaveSimulation();

    // Write the current and previous u fields to the output
    void append_u_fields() override;
    // Run `n` steps`
    void run(int n) override;
};

#endif
