#pragma once

#include <utility>
#include "Task.h"

namespace parex {

/**
  Essentially a wrapper around a `Task *`

  Can create
*/
class TaskRef {
public:
    /**/        TaskRef   ( Task *t = nullptr, std::size_t nout = 0 ) : task( t ), nout( nout ) { inc_ref( t ); }
    /**/        TaskRef   ( const TaskRef &that ) : task( that.task ) { inc_ref( task ); }
    /**/        TaskRef   ( TaskRef &&that ) : task( std::exchange( that.task, nullptr ) ) {}

    /**/       ~TaskRef   () { dec_ref( task ); }

    TaskRef&    operator= ( const TaskRef &that ) { inc_ref( that.task ); dec_ref( task ); task = that.task; return *this; }
    TaskRef&    operator= ( TaskRef &&that ) { task = std::exchange( that.task, nullptr ); return *this; }

    static void inc_ref   ( Task *task ) { if ( task ) ++task->cpt_use; }
    static void dec_ref   ( Task *task ) { if ( task && ! --task->cpt_use ) delete task; }

    Task*       task;
    std::size_t nout = 0; ///< num output
};

} // namespace parex

