#include "../utility/generic_ostream_output.h"
#include "../utility/TODO.h"
#include "../utility/P.h"
#include "SchedulerFront.h"
#include "Scheduler.h"
#include <iostream>
#include <map>

namespace parex {

Scheduler scheduler;

Scheduler::Scheduler() {
    log = false;
}

void Scheduler::append( const Rc<Task> &target ) {
    targets.push_back( target );
}

void Scheduler::run( const Rc<Task> &target ) {
    append( target );
    run();
}

void Scheduler::run() {
    SchedulerFront front;
    for( const Rc<Task> &target : targets )
        target->get_front_rec( front );

    // find the next task to execute
    while ( Task *task = front.pop() ) {
        if ( log ) { task->write_to_stream( std::cout ); std::cout << std::endl; }

        // preparation
        task->computed = true;
        task->prepare();
        if ( ! task->computed ) {
            task->in_front = false;
            task->scheduled = false;
            task->get_front_rec( front );
            Task::display_dot( targets );
            continue;
        }

        // exec
        task->exec();

        // parent task that can be executed
        for( Task *parent : task->parents )
            parent->get_front_rec( front );
    }

    targets.clear();
}

} // namespace parex
