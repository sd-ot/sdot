#pragma once

#include "KernelCode.h"
#include "Value.h"

namespace parex {

class Scheduler {
public:
    /**/                 Scheduler    ();
    /**/                ~Scheduler    ();

    Scheduler&           operator<<   ( const TaskRef &task_ref );
    Scheduler&           operator<<   ( const Value &value );
    Scheduler&           operator<<   ( Task *task );

    void                 run          ();

    bool                 log          = false;

    KernelCode           kernel_code; ///<

private:
    void                 exec_task    ( Task *task );

    std::vector<TaskRef> targets;     ///<
};

extern Scheduler scheduler;

} // namespace parex


