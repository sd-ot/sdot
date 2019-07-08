#pragma once

#include <mutex>
#include <deque>
#include <map>

namespace sdot {

/***/
class Stat {
public:
    struct StatData {
        std::deque<double> values;
        double step = 0;
    };

    ~Stat();

    void add( const std::string &id, double value ) {
        mutex.lock();
        stats[ id ].values.push_back( value );
        mutex.unlock();
    }

    void add_for_dist( const std::string &id, double value, double step = 1.0 ) {
        mutex.lock();
        stats[ id ].values.push_back( value );
        stats[ id ].step = step;
        mutex.unlock();
    }

    std::mutex mutex;
    std::size_t num_phase = 0;
    std::map<std::string,StatData> stats;
};

extern Stat stat;

}
