#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build-CUDA"

echo "[buildscript] Removing ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"

echo "[buildscript] Configuring CMake for CUDA"
cmake -S src -B "${BUILD_DIR}" -DAWAVE_MODE=CUDA -DCMAKE_BUILD_TYPE=Release

echo "[buildscript] Building"
cmake --build "${BUILD_DIR}" -j

echo "[buildscript] Done"
