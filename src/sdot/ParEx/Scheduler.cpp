#include "KernelCode.h"
#include "Scheduler.h"

#include "../support/P.h"

namespace parex {

Scheduler scheduler;

Scheduler::Scheduler() {
}

Scheduler &Scheduler::operator<<( const TaskRef &task_ref ) {
    return operator<<( task_ref.task );
}

Scheduler &Scheduler::operator<<( const Value &value ) {
    return operator<<( value.ref );
}

Scheduler &Scheduler::operator<<( Task *task ) {
    targets.push_back( task );
    return *this;
}

void Scheduler::run() {
    // front = all the task that can be executed
    std::vector<Task *> front;
    ++Task::curr_op_id;
    for( const TaskRef &value : targets )
        value.task->get_front_rec( front );

    for( Task *task : front )
        task->in_front = true;

    //
    while ( ! front.empty() ) {
        Task *task = front.back();
        front.pop_back();

        task->computed = true;
        exec_task( task );

        for( Task *parent : task->parents ) {
            if ( parent->children_are_computed() && ! parent->in_front ) {
                front.push_back( parent );
                parent->in_front = true;
            }
        }
    }

}

void Scheduler::exec_task( Task *task ) {
    std::vector<std::string> input_type;
    std::vector<void *> input_data;

    if ( std::size_t ni = task->children.size() ) {
        input_type.resize( ni );
        input_data.resize( ni );
        for( std::size_t i = 0; i < ni; ++i ) {
            input_type[ i ] = task->children[ i ].task->output_type;
            input_data[ i ] = task->children[ i ].task->output_data;
        }
    }

    auto func = kernel_code.func( *task->kernel, input_type );
    func( task, input_data.data() );
}

} // namespace parex
