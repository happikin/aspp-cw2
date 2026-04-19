// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_PARAMS_H
#define AWAVE_PARAMS_H

#include <format>

#include "ndarray.h"

// An array owns it's data (i.e. is responsible for deleting it).
using array3d = nd::array<double, 3>;
// A view, like a span, does not own data, it just, well, views it.
using view3d = array3d::view_type;
// Type to describe the shapes/indices of arrays/views
using shape_t = array3d::index_type;

// Describe the simulation
struct Params {
    double dx;
    double dt;
    shape_t shape;
    int nsteps;
    int out_period;
    int nBoundaryLayers;
};

// Describe how we're running
struct RunParams {
    shape_t mpi_shape;
    bool doIO;
    bool cpu_serial;
    bool skip_cpu;
};

// Format shape
template<>
struct std::formatter<shape_t, char> {
    using elem_t = typename shape_t::value_type;
    std::formatter<elem_t, char> elem_formatter;

    // Deal with the format string: delegate to element
    template<class Ctx>
    constexpr Ctx::iterator parse(Ctx& ctx) {
        return elem_formatter.parse(ctx);
    }
    // Actually do the formatting
    template<class Ctx>
    Ctx::iterator format(shape_t const& sh, Ctx& ctx) const {
        typename Ctx::iterator o = ctx.out();
        *o = '[';
	int n = 0;
	for (auto& e: sh) {
	  if (n++)
	    *o = ", ";
	  o = elem_formatter.format(e, ctx);
	}
        *o = ']';
        return o;
    }
};

// Allowing formatting of parameters
template<>
struct std::formatter<Params, char> {
    // Deal with the format string, don't allow anything for simplicity
    template<class Ctx>
    constexpr Ctx::iterator parse(Ctx& ctx) {
        auto it = ctx.begin();
        auto end = ctx.end();

        // Ensure no unexpected characters after the colon (if any)
        if (it != end && *it != '}') {
            throw std::format_error("Invalid format specifier for Params");
        }
        return it;
    }
    // Actually do the formatting
    template<class Ctx>
    Ctx::iterator format(Params const& p, Ctx& ctx) const {
        return std::format_to(
            ctx.out(),
            "Grid shape: {}\n"
            "Grid spacing: {} m\n"
            "Time step: {} s\n"
            "Number of steps: {}\n"
            "Output period: {}",
            p.shape,
            p.dx, p.dt, p.nsteps, p.out_period);
    }
};
#endif
