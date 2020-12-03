#include "generic_ostream_output.h"
#include "Scheduler.h"
#include <iostream>
#include <map>

namespace {
    bool all_children_are_computed( Task *task ) {
        for( const RcPtr<Task> &child : task->children )
            if ( ! child->computed )
                return false;
        return true;
    }

    void get_front_rec( Task *task, std::map<int,std::vector<RcPtr<Task>>> &front ) {
        // in_front
        if ( task->in_front || task->computed )
            return;

        if ( all_children_are_computed( task ) ) {
            front[ - task->priority ].push_back( task );
            task->in_front = true;
            return;
        }

        // in_schedule
        if ( task->scheduled )
            return;
        task->scheduled = true;

        for( const RcPtr<Task> &child : task->children )
            get_front_rec( child.ptr(), front );
    }
}

Scheduler scheduler;

Scheduler::Scheduler() {
    log = false;
}

void Scheduler::append_target( const RcPtr<Task> &target ) {
    targets.push_back( target );
}

void Scheduler::run() {
    std::map<int,std::vector<RcPtr<Task>>> front;
    for( const RcPtr<Task> &target : targets )
        get_front_rec( target.ptr(), front );

    //
    while ( ! front.empty() ) {
        // find the next task to execute
        std::map<int,std::vector<RcPtr<Task>>>::iterator first_in_front = front.begin();
        std::vector<RcPtr<Task>> &vf = first_in_front->second;
        RcPtr<Task> task = vf.back();
        vf.pop_back();

        if ( vf.empty() )
            front.erase( front.begin() );

        // exec
        if ( log ) std::cout << *task->kernel << std::endl;
        task->kernel->exec( task.ptr() );
        task->computed = true;

        // parent task that can be executed
        for( Task *parent : task->parents )
            get_front_rec( parent, front );

        // free the tasks that are not going to be used anymore
        for( RcPtr<Task> &ch : task->children )
            ch = nullptr;
    }

    targets.clear();
}
