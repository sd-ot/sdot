#include "Task.h"

namespace parex {

std::size_t Task::curr_op_id = 0;

Task::Task() {
    computed = false;
    cpt_use = 1;
    op_id = 0;
}

void Task::get_front_rec( std::vector<Task *> &front ) {
    if ( computed )
        return;

    if ( children_are_computed() ) {
        front.push_back( this );
        return;
    }

    for( const Value &input : inputs )
        input.task->get_front_rec( front );
}

bool Task::children_are_computed() const {
    for( const Value &input : inputs )
        if ( ! input.task->computed )
            return false;
    return true;
}

} // namespace parex
