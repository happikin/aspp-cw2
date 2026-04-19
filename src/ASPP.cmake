# From: https://www.kitware.com/cmake-and-the-default-build-type/
# Set a default build type if none was specified
set(_default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${_default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${_default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Set default CUDA arch to Ampere real and virtual
if (NOT CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "Setting CUDA target architectures to A100 and H100")
  set(CMAKE_CUDA_ARCHITECTURES "80;90" CACHE
    STRING "CUDA architectures (;-list)" FORCE)
endif()


# Based on the CMAKE CUDA ARCH, set the -gpu flags for the NVHPC compilers
function(get_nvhpc_flags)
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES PREPEND "cc" OUTPUT_VARIABLE flags)
  list(JOIN flags "," flags)
  set(flags "-gpu=${flags}")
  set(NVHPC_GPU_FLAGS "${flags}" CACHE STRING "NVHPC compiler GPU flags (;-list)")
endfunction()

# Create the interface library target to compile and link a CUDA Fortran program
# Call this, then do target_link_libraries(your-exe PUBLIC cuda_fortran)
function(enable_cuda_fortran)
  add_library(cuda_fortran INTERFACE)
  get_nvhpc_flags()
  target_compile_options(cuda_fortran INTERFACE "-cuda;${NVHPC_GPU_FLAGS}")
  target_link_options(cuda_fortran INTERFACE "-cuda;${NVHPC_GPU_FLAGS}")
endfunction()

# CMake debug helper: print a variable
macro(D var)
  message(STATUS "${var} = '${${var}}'")
endmacro()
