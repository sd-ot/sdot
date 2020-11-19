#include "support/generic_ostream_output.h"
#include "KernelCode.h"
#include "Scheduler.h"

namespace parex {

Scheduler scheduler;

Scheduler::Scheduler() {
}

Scheduler::~Scheduler() {
    run();
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
    //    std::vector<Task *> tasks;
    //    for( const TaskRef &value : targets )
    //        tasks.push_back( value.task );
    //    Task::display_graphviz( tasks );
    //    P( targets.size() );

    // front = all the task that can be executed
    std::map<int,std::vector<TaskRef>> front;
    ++Task::curr_op_id;
    for( const TaskRef &value : targets )
        value.task->get_front_rec( front );

    //
    while ( ! front.empty() ) {
        // find the next task to execute
        std::map<int,std::vector<TaskRef>>::iterator first_in_front = front.begin();
        std::vector<TaskRef> &vf = first_in_front->second;
        TaskRef task_ref = vf.back();
        vf.pop_back();

        if ( vf.empty() )
            front.erase( front.begin() );

        //        if ( task_ref.task->kernel.name.find( "mk_item" ) != std::string::npos )
        //            Task::display_graphviz( { task_ref.task } );

        // exec
        if ( log ) std::cout << task_ref.task->kernel.name << std::endl;
        task_ref.task->computed = true;
        exec_task( task_ref.task );

        // parent task that can be executed
        for( Task *parent : task_ref.task->parents )
            parent->get_front_rec( front );

        // free the tasks that are not going to be used anymore
        for( TaskRef &ch : task_ref.task->children )
            ch = nullptr;
    }

    targets.clear();
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

    auto func = kernel_code.func( task->kernel, input_type );
    func( task, input_data.data() );
}

} // namespace parex
