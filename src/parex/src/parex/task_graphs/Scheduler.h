#pragma once

#include "../utility/Rc.h"
#include "Task.h"

/**
*/
class Scheduler {
public:
    /***/                 Scheduler();

    void                  append   ( const Rc<Task> &target );
    void                  run      ( const Rc<Task> &target );
    void                  run      ();

    bool                  log;
private:
    void                  exec_task( const Rc<Task> &task );

    std::vector<Rc<Task>> targets;
};

extern Scheduler scheduler;
