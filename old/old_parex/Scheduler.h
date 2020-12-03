#pragma once

#include "Task.h"

/**
*/
class Scheduler {
public:
    /***/                    Scheduler    ();

    void                     append_target( const RcPtr<Task> &target );
    void                     run          ();

    bool                     log;
private:
    void                     exec_task    ( const RcPtr<Task> &task );

    std::vector<RcPtr<Task>> targets;
};

extern Scheduler scheduler;
