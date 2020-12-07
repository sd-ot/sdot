#pragma once

#include "Task.h"
#include <map>

namespace parex {

/**
*/
class SchedulerFront {
public:
    using Map            = std::map<double,std::vector<Task *>>;

    void  write_to_stream( std::ostream &os ) const;
    void  insert         ( Task *task );
    bool  empty          () const;
    Task* pop            ();

    Map   map;
};

} // namespace parex
