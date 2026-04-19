//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "init_sos.h"

#include <algorithm>
#include <cmath>

const SpeedOfSoundProfile& SpeedOfSoundProfile::Get() {
    static SpeedOfSoundProfile instance;
    return instance;
}

// Data from wikipedia
SpeedOfSoundProfile::SpeedOfSoundProfile() :
        m_depth({0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0,
                 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1750.0, 2000.0}),
        m_speed({1540.4, 1540.5, 1540.7, 1534.4, 1523.3, 1519.6, 1518.5, 1517.9, 1517.3, 1516.6, 1516.5, 1516.2, 1516.4,
                 1517.2, 1518.2, 1519.5, 1521.0, 1522.6, 1524.1, 1525.7, 1527.3, 1529.0, 1530.7, 1532.4, 1536.7,
                 1541.0}) {
    m_max = *std::max_element(m_speed.begin(), m_speed.end());
}

double SpeedOfSoundProfile::conv(double depth) const {
    auto gt = std::upper_bound(m_depth.begin(), m_depth.end(), depth);
    if (gt == m_depth.end())
        return m_speed.back();
    if (gt == m_depth.begin())
        return m_speed.front();
    int hi = gt - m_depth.begin();
    int lo = hi - 1;
    return std::lerp(m_speed[lo], m_speed[hi], (depth - m_depth[lo])/(m_depth[hi] - m_depth[lo]));
}
