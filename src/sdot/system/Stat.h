#pragma once

#include <deque>
#include <map>

/***/
class Stat {
public:
    struct StatData {
        std::deque<double> values;
    };

    ~Stat();

    void add( const std::string &id, double value ) {
        stats[ id ].values.push_back( value );
    }

    std::map<std::string,StatData> stats;
};

extern Stat stat;
