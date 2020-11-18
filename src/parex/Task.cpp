#include "support/DotOut.h"
#include "TaskRef.h"

#include <fstream>

namespace parex {

std::size_t Task::curr_op_id = 0;

void Task::write_to_stream( std::ostream &os ) const {
    if ( kernel )
        os << kernel.name;
    else {
        os << "src";
        //        os << "src(";
        //        for( std::size_t i = 0; i < outputs.size(); ++i )
        //            os << ( i ? "," : "" ) << outputs[ i ].type;
        //        os << ")";
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

void Task::display_graphviz( const std::vector<Task *> &tasks, bool display_src_nodes, std::string f, const char *prg ) {
    std::ofstream os( f );

    os << "digraph LexemMaker {\n";
    std::set<Task *> seen;
    for( Task *t : tasks ) {
        t->for_each_rec( [&]( Task *task ) {
            if ( display_src_nodes == false && task->kernel.name.empty() )
                return;

            os << "  node_" << task << " [label=\"";
            dot_out( os, *task );
            os << "," << task->computed;
            os << "\"];\n";

            for( const TaskRef &tr : task->children )
                if ( tr.task && ( display_src_nodes || ! tr.task->kernel.name.empty() ) )
                    os << "  node_" << task << " -> node_" << tr.task << " [label=" << tr.nout << "];\n";

            //            for( const Task *tr : task->parents )
            //                os << "  node_" << tr << " -> node_" << task << " [color=red];\n";

        }, seen, /*go to parents*/ true );
    }
    os << "}\n";

    os.close();

    exec_dot( f, prg );
}

void Task::for_each_rec( const std::function<void (Task *)> &f, std::set<Task *> &seen, bool go_to_parents ) {
    if ( seen.count( this ) )
        return;
    seen.insert( this );

    for( const TaskRef &tr : children )
        if ( tr.task )
            tr.task->for_each_rec( f, seen, go_to_parents );

    for( Task *p : parents )
        if ( p )
            p->for_each_rec( f, seen, go_to_parents );

    f( this );
}

void Task::get_front_rec( std::map<int,std::vector<TaskRef>> &front ) {
    // in_front
    if ( in_front || computed )
        return;

    if ( children_are_computed() ) {
        front[ - kernel.priority ].push_back( this );
        in_front = true;
        return;
    }

    // in_schedule
    if ( in_schedule )
        return;
    in_schedule = true;

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
