#include <algorithm>
#include "Kernel.h"

namespace parex {

std::size_t Task::curr_op_id = 0;

Task::~Task() {
    delete kernel;

    auto ei = [&]( const Task *t ) { return t == this; };
    for( const TaskRef &child : children )
        child.task->parents.erase( std::remove_if( child.task->parents.begin(),  child.task->parents.end(), ei ), child.task->parents.end() );
}

Task *Task::call( Kernel *kernel, std::vector<TaskRef> &&children ) {
    Task *res = new Task;

    res->children = std::move( children );
    res->kernel = kernel;

    for( TaskRef &ch : res->children )
        ch.task->parents.push_back( res );

    return res;
}

void Task::get_front_rec( std::vector<Task *> &front ) {
    if ( computed )
        return;

    if ( children_are_computed() ) {
        front.push_back( this );
        return;
    }

    for( const TaskRef &child : children )
        child.task->get_front_rec( front );
}

bool Task::children_are_computed() const {
    for( const TaskRef &child : children )
        if ( ! child.task->computed )
            return false;
    return true;
}

} // namespace parex
