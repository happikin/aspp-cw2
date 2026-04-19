// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_UFIELD_H
#define AWAVE_UFIELD_H

#include <array>
#include "ndarray.h"
#include "util.h"

/// Hold the current time step's u field as well as the next and previous.
class uField {
    int m_time;
    // Going to do a ring buffer
    std::array<array3d, 3> fields;
    std::size_t current;
public:
    uField() = default;

    explicit uField(array3d&& initial, int t = 0) ;

    uField clone() const;

    [[nodiscard]] auto& time() const {
        return m_time;
    }

    auto& next() {
        return fields[(current + 1) % 3];
    }
    auto& next() const {
        return fields[(current + 1) % 3];
    }
    auto& now() {
        return fields[current];
    }
    auto& now() const {
        return fields[current];
    }
    auto& prev() {
        return fields[(current + 2) % 3];
    }
    auto& prev() const {
        return fields[(current + 2) % 3];
    }

    void advance();
};
#endif