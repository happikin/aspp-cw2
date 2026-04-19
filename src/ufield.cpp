//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//

#include "ufield.h"

uField::uField(array3d&& initial, int t) : m_time(t) {
    current = 1;
    fields[1] = std::move(initial);
    fields[0] = fields[1].clone();
    fields[2] = fields[1].clone();
}

uField uField::clone() const {
    uField ans;
    ans.m_time = m_time;
    ans.current = current;
    for (int i = 0; i < 3; ++i)
        ans.fields[i] = fields[i].clone();
    return ans;
}

void uField::advance() {
    current = (current + 1) % 3;
    m_time += 1;
}