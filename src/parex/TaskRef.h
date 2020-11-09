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
    /**/        TaskRef        ( Task *t = nullptr, std::size_t nout = 0 ) : task( t ), nout( nout ) { inc_ref( t ); }
    /**/        TaskRef        ( const TaskRef &that ) : task( that.task ), nout( that.nout ) { inc_ref( that.task ); }
    /**/        TaskRef        ( TaskRef &&that ) : task( std::exchange( that.task, nullptr ) ), nout( that.nout ) {}

    /**/       ~TaskRef        () { dec_ref( task ); }

    TaskRef&    operator=      ( const TaskRef &that ) { inc_ref( that.task ); dec_ref( task ); task = that.task; nout = that.nout; return *this; }
    TaskRef&    operator=      ( TaskRef &&that ) { dec_ref( task ); task = std::exchange( that.task, nullptr ); nout = that.nout; return *this; }

    void        write_to_stream( std::ostream &os ) const { if ( task ) task->write_to_stream( os ); else os << "null"; os << "[" << nout << "]"; }

    static void inc_ref        ( Task *task ) { if ( task ) ++task->ref_count; }
    static void dec_ref        ( Task *task ) { if ( task && ! --task->ref_count ) delete task; }

    Task*       task;
    std::size_t nout           = 0; ///< num output
};

// defined here because we need to be able to used it in kernels without additionnal .o files
inline bool Task::move_arg( std::size_t num_arg, std::size_t num_out ) {
    if ( num_arg < children.size() && children[ num_arg ].task->ref_count <= 1 ) {
        if ( outputs.size() <= num_out )
            outputs.resize( num_out + 1 );
        outputs[ num_out ] = std::move( children[ num_arg ].task->outputs[ children[ num_arg ].nout ] );
        return true;
    }
    return false;
}

} // namespace parex

