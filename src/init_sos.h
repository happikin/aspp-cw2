// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//

#ifndef AWAVE_INIT_SOS_H
#define AWAVE_INIT_SOS_H
#include <vector>

class SpeedOfSoundProfile {
    std::vector<double> m_depth;
    std::vector<double> m_speed;
    double m_max;

    static SpeedOfSoundProfile const& Get();
    SpeedOfSoundProfile();
    double conv(double depth) const;
public:
    inline static double MAX() {
        return Get().m_max;
    }
    inline static double convert(double depth) {
        return Get().conv(depth);
    }
};
#endif //AWAVE_INIT_SOS_H
