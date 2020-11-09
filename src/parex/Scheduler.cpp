#include "support/generic_ostream_output.h"
#include "KernelCode.h"
#include "Scheduler.h"

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
    task->is_target_in_scheduler = true;
    targets.push_back( task );
    return *this;
}

void Scheduler::run() {
    // front = all the task that can be executed
    std::vector<Task *> front;
    ++Task::curr_op_id;
    for( const TaskRef &value : targets )
        value.task->get_front_rec( front );

    //
    while ( ! front.empty() ) {
        // find the next task to execute
        Task *task = front.back();
        front.pop_back();

        // exec
        if ( log ) std::cout << *task << std::endl;
        task->computed = true;
        exec_task( task );

        // parent task that can be executed
        for( Task *parent : task->parents ) {
            if ( parent->children_are_computed() && ! parent->in_front ) {
                front.push_back( parent );
                parent->in_front = true;
            }
        }

        // free the tasks that are not going to be used anymore
        for( TaskRef &ch : task->children ) {
            if ( --ch.task->ref_count == 0 )
                delete ch.task;
            ch.task = nullptr;
        }
        if ( task->is_target_in_scheduler ) {
            for( TaskRef &t : targets ) {
                if ( t.task == task ) {
                    if ( --task->ref_count == 0 )
                        delete task;
                    t.task = nullptr;
                }
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
            input_type[ i ] = task->children[ i ].task->outputs[ task->children[ i ].nout ].type;
            input_data[ i ] = task->children[ i ].task->outputs[ task->children[ i ].nout ].data;
        }
    }

    auto func = kernel_code.func( *task->kernel, input_type );
    func( task, input_data.data() );
}

} // namespace parex
