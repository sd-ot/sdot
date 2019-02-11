#pragma once

#include <chrono>
#include <deque>
#include <map>

/***/
class Tick {
public:
    using TP = std::chrono::high_resolution_clock::time_point;

    struct TaskData {
        std::deque<double> timings; ///< in seconds
        TP                 start;
    };

    ~Tick();

    static TP get_time() {
        return std::chrono::high_resolution_clock::now();
    }

    static double delta( TP beg, TP end ) {
        auto delta = end, beg;
        return std::chrono::duration_cast<std::chrono::microseconds>( delta ).count() / 1e6;
    }

    static double elapsed_since( TP start ) {
        return delta( start, std::chrono::high_resolution_clock::now() );
    }

    void start( const std::string &id ) {
        tasks[ id ].start = get_time();
    }

    void stop( const std::string &id ) {
        TaskData &td = tasks[ id ];
        td.timings.emplace_back( elapsed_since( td.start ) );
    }

    Tick &operator<<( const std::string &id ) {
        start( id );
        return *this;
    }

    Tick &operator>>( const std::string &id ) {
        stop( id );
        return *this;
    }

    void display_timings( const std::string &sub_task );

    std::map<std::string,TaskData> tasks;
};

extern Tick tick;

/** RAII tick */
struct AutoTick {
    AutoTick( const std::string &id ) : id( id ) {
        tick.start( id );
    }
    ~AutoTick() {
        tick.stop( id );
    }

    const std::string id;
};
