cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(ASPP)

# Defer language set up so we can ensure the right compiler is picked
project(awave LANGUAGES NONE)

set(AWAVE_MODE "" CACHE STRING "GPU programming model: OpenMP, CUDA, SYCL")

# The following if-tree sets up the compiler based on the programming
# model and sets the gpu_impl variable to the required source file,
# any libraries needed, and the AWAVE_${MODEL} definition.
if (AWAVE_MODE STREQUAL "OpenMP")
    # Use the NVIDIA HPC compiler
    set(CMAKE_CXX_COMPILER nvc++)
    enable_language(CXX)
    get_nvhpc_flags()
    # NVC++ likes to target the host ISA very specifically, which is Intel
    # for ASPP VM but AMD for GPU nodes
    set(CMAKE_CXX_FLAGS -march=x86-64-v3)
    # Set OpenMP to GPU mode
    find_package(OpenMP REQUIRED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        INTERFACE_COMPILE_OPTIONS "-mp=gpu;${NVHPC_GPU_FLAGS}"
        INTERFACE_LINK_OPTIONS "-mp=gpu")

    set(gpu_impl wave_omp.cpp)
    set(gpu_libs OpenMP::OpenMP_CXX)
    add_compile_definitions(AWAVE_OMP)

elseif(AWAVE_MODE STREQUAL "CUDA")
    # NVCC will only use GGC <= 12 which has poor C++20 support; use NV's compiler instead
    set(CMAKE_CXX_COMPILER nvc++)
    enable_language(CXX)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    # NVC++ likes to target the host ISA very specifically, which is Intel
    # for ASPP VM but AMD for GPU nodes
    set(CMAKE_CXX_FLAGS -march=x86-64-v3)
    # Be relaxed about constexpr device functions and use host-side compiler
    set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -ccbin ${CMAKE_CXX_COMPILER} -march=x86-64-v3")

    set(gpu_impl wave_cuda.cu)
    set(gpu_libs "")
    # pass the mode to the code
    add_compile_definitions(AWAVE_CUDA)

elseif(AWAVE_MODE STREQUAL "SYCL")
    set(CMAKE_CXX_COMPILER acpp)
    enable_language(CXX)
    # Adaptive C++ complains if you don't set an explict optimisation level.
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(gpu_impl wave_sycl.cpp)
    set(gpu_libs "")
    add_compile_definitions(AWAVE_SYCL)

else()
    message(FATAL_ERROR "Invalid AWAVE_MODE '${AWAVE_MODE}', must be one of OpenMP, CUDA, SYCL")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get MPI
find_package(MPI REQUIRED)

# Use HDF for IO
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(awave
        h5io.cpp
        parallel.cpp
        init_sos.cpp
        ufield.cpp
        wave_cpu.cpp
        ${gpu_impl}
        main.cpp
)

target_link_libraries(awave MPI::MPI_CXX hdf5::hdf5 ${gpu_libs})
