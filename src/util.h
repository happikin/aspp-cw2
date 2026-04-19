// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_UTIL_H
#define AWAVE_UTIL_H

#include <format>
#include <iostream>

#include "ndarray.h"

using array3d = nd::array<double, 3>;
using shape_t = array3d::index_type;

// How many blocks of size B to reach n?
auto ceildiv(std::integral auto n, std::integral auto b) {
    return n ? (n - 1) / b + 1 : 0;
}
template <std::integral T, std::size_t N>
auto ceildiv(nd::vec<T, N> const& n, nd::vec<T, N> const& b) {
    return (n - 1) / b + 1;
}

template <typename... Args>
void out(std::format_string<Args...> fstr, Args&&... args) {
    std::cout << std::format(fstr, std::forward<Args>(args)...) << '\n';
}

template <typename... Args>
void outroot(int rank, std::format_string<Args...> fstr, Args&&... args) {
  if (rank == 0)
    std::cout << std::format(fstr, std::forward<Args>(args)...) << '\n';
}

template <typename... Args>
void err(std::format_string<Args...> fstr, Args&&... args) {
    std::cerr << std::format(fstr, std::forward<Args>(args)...) << '\n';
}

template <std::floating_point T>
bool approxEq(T a, T b, T epsilon) {
    constexpr T min_norm = std::numeric_limits<T>::min();
    T const absA = std::abs(a);
    T const absB = std::abs(b);
    T const diff = std::abs(a - b);

    // handle exact equality
    if (a == b) {
        return true;
    } else if (a == 0 || b == 0 || (absA + absB < min_norm)) {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return diff < (epsilon * min_norm);
    } else { // use relative error
        return diff / std::min((absA + absB), std::numeric_limits<T>::max()) < epsilon;
    }
}

#endif
