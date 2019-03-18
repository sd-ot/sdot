#pragma once

#include <chrono>

/***/
class Time {
public:
    using TP = std::chrono::high_resolution_clock::time_point;

    static TP get_time() {
        return std::chrono::high_resolution_clock::now();
    }

    static double delta( TP beg, TP end ) {
        auto delta = end - beg;
        return std::chrono::duration_cast<std::chrono::microseconds>( delta ).count() / 1e6;
    }

    static double elapsed_since( TP start ) {
        return delta( start, std::chrono::high_resolution_clock::now() );
    }
};
