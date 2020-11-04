#pragma once

#include "Value.h"

namespace parex {

class Scheduler {
public:
    /**/               Scheduler ();

    Scheduler&         operator<<( const Value &value );
    void               run       ();

    std::vector<Value> targets;
};

extern Scheduler scheduler;

} // namespace parex


