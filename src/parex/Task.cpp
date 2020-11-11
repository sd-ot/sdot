#include "support/DotOut.h"
#include "TaskRef.h"

#include <algorithm>
#include <fstream>

namespace parex {

std::size_t Task::curr_op_id = 0;

Task::~Task() {
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
        os << kernel.name;
    else {
        os << "src(";
        for( std::size_t i = 0; i < outputs.size(); ++i )
            os << ( i ? "," : "" ) << outputs[ i ].type;
        os << ")";
    }
}

Task *Task::ref_type( const std::string type ) {
    Task *res = new Task;
    res->outputs.emplace_back( "parex::S<" + type + ">", nullptr, false );
    res->computed = true;
    return res;
}

Task *Task::ref_num( int value ) {
    Task *res = new Task;
    res->outputs.emplace_back( "parex::N<" + std::to_string( value ) + ">", nullptr, false );
    res->computed = true;
    return res;
}

TaskRef Task::call_r( const Kernel &kernel, std::vector<TaskRef> &&inputs ) {
    Task *res = new Task;

    res->children = std::move( inputs );
    res->kernel = kernel;

    for( TaskRef &ch : res->children )
        ch.task->parents.push_back( res );

    return res;
}

Task *Task::call( const Kernel &kernel, const std::vector<TaskRef *> &outputs, std::vector<TaskRef> &&inputs ) {
    Task *res = new Task;

    res->children = std::move( inputs );
    res->kernel = kernel;

    for( TaskRef &ch : res->children )
        ch.task->parents.push_back( res );

    for( std::size_t n = 0; n < outputs.size(); ++n )
        *outputs[ n ] = { res, n };

    return res;
}

void Task::display_graphviz( const std::vector<Task *> &tasks, std::string f, const char *prg ) {
    std::ofstream os( f );

    os << "digraph LexemMaker {\n";
    std::set<Task *> seen;
    for( Task *t : tasks ) {
        t->for_each_rec( [&]( Task *task ) {
            os << "  node_" << task << " [label=\"";
            dot_out( os, *task );
            os << "," << task->computed;
            os << "\"];\n";

            for( const TaskRef &tr : task->children )
                os << "  node_" << task << " -> node_" << tr.task << ";\n";

            for( const Task *tr : task->parents )
                os << "  node_" << tr << " -> node_" << task << " [color=red];\n";

        }, seen );
    }
    os << "}\n";

    os.close();

    exec_dot( f, prg );
}

void Task::for_each_rec( const std::function<void (Task *)> &f, std::set<Task *> &seen ) {
    if ( seen.count( this ) )
        return;
    seen.insert( this );

    for( const TaskRef &tr : children )
        tr.task->for_each_rec( f, seen );

    f( this );
}

void Task::get_front_rec( std::vector<TaskRef> &front ) {
    if ( computed || in_front || op_id == curr_op_id )
        return;
    op_id = curr_op_id;

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
