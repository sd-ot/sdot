#pragma once

#include "support/ERROR.h"
#include <algorithm>
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

    bool        operator==      ( const TaskRef &that ) const { return task == that.task && nout == that.nout; }

    void        write_to_stream( std::ostream &os ) const { if ( task ) task->write_to_stream( os ); else os << "null"; os << "[" << nout << "]"; }

    static void inc_ref        ( Task *task ) { if ( task ) ++task->ref_count; }
    static void dec_ref        ( Task *task ) { if ( task && ! --task->ref_count ) delete task; }

    Task*       task;
    std::size_t nout           = 0; ///< num output
};

// defined here because we need to be able to used it in kernels without additionnal .o files
inline Task::~Task() {
    // erase ref of `this` in children
    auto ei = [&]( const Task *t ) { return t == this; };
    for( const TaskRef &child : children )
        if ( child.task )
            child.task->parents.erase( std::remove_if( child.task->parents.begin(),  child.task->parents.end(), ei ), child.task->parents.end() );

    //
    #ifdef PAREX_IN_KERNEL
    ERROR( "for now, tasks can't be destroyed inside a kernel" );
    #else
    for( Output &output : outputs )
        output.destroy();
    #endif // PAREX_IN_KERNEL
}

inline bool Task::move_arg( std::size_t num_arg, std::size_t num_out ) {
    return move_arg( std::vector<std::size_t>{ num_arg }, std::vector<std::size_t>{ num_out } );
}

inline bool Task::move_arg( const std::vector<std::size_t> &num_args, const std::vector<std::size_t> &num_outs ) {
    // if amongst the childrent task, there's one that is referenced elsewhere, we can't preempt it
    for( std::size_t num_arg : num_args )
        if ( children[ num_arg ].task->ref_count > children[ num_arg ].task->parents.size() )
            return false;

    // check that args have only `this` is a parent. If it's not the case, check if the parent is interested by another output
    for( std::size_t num_arg : num_args )
        for( Task *p : children[ num_arg ].task->parents )
            if ( p != this )
                for( const TaskRef &c : p->children )
                    if ( c == children[ num_arg ] )
                        return false;

    // if ok, move inputs to outputs
    for( std::size_t i = 0; i < num_args.size(); ++i ) {
        if ( outputs.size() <= num_outs[ i ] )
            outputs.resize( num_outs[ i ] + 1 );
        outputs[ num_outs[ i ] ] = std::move( children[ num_args[ i ] ].task->outputs[ children[ num_args[ i ] ].nout ] );
    }
    return true;
}

inline bool Task::move_arg( const std::vector<std::size_t> &num_arg ) {
    return move_arg( num_arg, num_arg );
}

inline TaskRef Task::call_r( const Kernel &kernel, std::vector<TaskRef> &&inputs, bool append_parent_task ) {
    Task *res = new Task;

    res->children = std::move( inputs );
    res->kernel = kernel;

    for( TaskRef &ch : res->children )
        ch.task->parents.push_back( res );

    if ( append_parent_task )
        return call_r( Kernel::with_task_as_arg( "move" ), { res }, false );

    return res;
}

inline Task *Task::call( const Kernel &kernel, const std::vector<TaskRef *> &outputs, std::vector<TaskRef> &&inputs ) {
    Task *res = new Task;

    res->children = std::move( inputs );
    res->kernel = kernel;

    for( TaskRef &ch : res->children )
        ch.task->parents.push_back( res );

    for( std::size_t n = 0; n < outputs.size(); ++n )
        *outputs[ n ] = { res, n };

    return res;
}

inline void Task::insert_before_parents( const TaskRef &t ) {
    std::set<Task *> seen_parents;
    for( Task *p : parents ) {
        if ( p == t.task || seen_parents.insert( p ).second == false )
            continue;

        for( TaskRef &ch : p->children ) {
            if ( ch.task == this ) {
                t.task->parents.push_back( p );
                ch = t;
            }
        }
    }
    parents = { t.task };
}

} // namespace parex

