#include "generic_ostream_output.h"
#include "ComputableTask.h"
#include "Scheduler.h"
#include <iostream>
#include <map>

Scheduler scheduler;

Scheduler::Scheduler() {
    log = false;
}

void Scheduler::append( const Rc<Task> &target ) {
    targets.push_back( target );
}

void Scheduler::run() {
    std::map<int,std::vector<ComputableTask *>> front;
    for( const Rc<Task> &target : targets )
        target->get_front_rec( front );

    //
    while ( ! front.empty() ) {
        // find the next task to execute
        std::map<int,std::vector<ComputableTask *>>::iterator first_in_front = front.begin();
        std::vector<ComputableTask *> &vf = first_in_front->second;
        ComputableTask *task = vf.back();
        vf.pop_back();

        if ( vf.empty() )
            front.erase( front.begin() );

        // exec
        if ( log ) { task->write_to_stream( std::cout ); std::cout << std::endl; }
        task->computed = true;
        task->exec();

        // parent task that can be executed
        for( ComputableTask *parent : task->parents )
            parent->get_front_rec( front );
    }

    targets.clear();
}
