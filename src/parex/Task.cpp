#include <algorithm>
#include "Kernel.h"

namespace parex {

std::size_t Task::curr_op_id = 0;

Task::~Task() {
    // kernel is owned
    delete kernel;

    // erase ref of `this` in children
    auto ei = [&]( const Task *t ) { return t == this; };
    for( const TaskRef &child : children )
        if ( child.task )
            child.task->parents.erase( std::remove_if( child.task->parents.begin(),  child.task->parents.end(), ei ), child.task->parents.end() );

    //
    for( Output &output : outputs )
        output.destroy();
}

void Task::write_to_stream( std::ostream &os ) const {
    if ( kernel )
        os << kernel->name;
    else {
        os << "src(";
        for( std::size_t i = 0; i < outputs.size(); ++i )
            os << ( i ? "," : "" ) << outputs[ i ].type;
        os << ")";
    }
}

TaskRef Task::call_r( Kernel *kernel, std::vector<TaskRef> &&inputs ) {
    Task *res = new Task;

    res->children = std::move( inputs );
    res->kernel = kernel;

    for( TaskRef &ch : res->children )
        ch.task->parents.push_back( res );

    return res;
}

Task *Task::call( Kernel *kernel, const std::vector<TaskRef *> &outputs, std::vector<TaskRef> &&inputs ) {
    Task *res = new Task;

    res->children = std::move( inputs );
    res->kernel = kernel;

    for( TaskRef &ch : res->children )
        ch.task->parents.push_back( res );

    for( std::size_t n = 0; n < outputs.size(); ++n )
        *outputs[ n ] = { res, n };

    return res;
}

void Task::get_front_rec( std::vector<Task *> &front ) {
    if ( computed || in_front )
        return;

    if ( children_are_computed() ) {
        front.push_back( this );
        in_front = true;
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
