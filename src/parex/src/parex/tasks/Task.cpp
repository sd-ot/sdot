#include "../data/TypeFactory.h"
#include "../utility/DotOut.h"
#include "../utility/TODO.h"
#include "SchedulerFront.h"
#include "Task.h"
#include <fstream>
#include <set>

namespace parex {

Task::Task( std::string &&name, std::vector<Rc<Task>> &&children, double priority ) : children( std::move( children ) ), priority( priority ), name( std::move( name ) ) {
    scheduled = false;
    computed  = false;
    in_front  = false;

    for( Rc<Task> &ch : this->children )
        ch->parents.push_back( this );
}

Task::~Task() {
    for( Rc<Task> &ch : this->children )
        if ( ch )
            ch->remove_from_parents( this );
}

Task *Task::new_src( Type *type, void *data, bool own ) {
    Task *res = new Task( "Src[" + type->name + "]" );
    res->output.type = type;
    res->output.data = data;
    res->output.own = own;
    res->computed  = true;
    return res;
}

void Task::remove_from_parents( const Task *parent_to_remove ) {
    for( std::size_t i = 0; i < parents.size(); ++i ) {
        if ( parents[ i ] == parent_to_remove ) {
            std::swap( parents[ i ], parents.back() );
            parents.pop_back();
            return;
        }
    }
}

bool Task::all_ch_computed() const {
    for( const Rc<Task> &child : children )
        if ( child && ! child->computed )
            return false;
    return true;
}

void Task::write_to_stream( std::ostream &os ) const {
    os << name;
}

void Task::get_front_rec( SchedulerFront &front ) {
    // in_front
    if ( in_front || computed )
        return;

    if ( all_ch_computed() ) {
        front.insert( this );
        in_front = true;
        return;
    }

    // in_schedule
    if ( scheduled )
        return;
    scheduled = true;

    for( const Rc<Task> &child : children )
        child->get_front_rec( front );
}

void Task::prepare() {
    check_input_same_memory();
}

void Task::exec() {
}

void Task::for_each_rec( const std::function<void (Task *)> &f, std::set<Task *> &seen, bool go_to_parents ) {
    if ( seen.count( this ) )
        return;
    seen.insert( this );

    for( const Rc<Task> &tr : children )
        if ( tr )
            tr->for_each_rec( f, seen, go_to_parents );

    for( Task *p : parents )
        p->for_each_rec( f, seen, go_to_parents );

    f( this );
}

void Task::display_dot( const std::vector<Rc<Task>> &tasks, std::string f, const char *prg ) {
    std::ofstream os( f );

    os << "digraph LexemMaker {\n";
    std::set<Task *> seen;
    for( const Rc<Task> &t : tasks ) {
        t->for_each_rec( [&]( Task *task ) {
            os << "  node_" << task << " [label=\"";
            dot_out( os, *task );
            os << "," << task->computed;
            os << "\"];\n";

            for( const Rc<Task> &tr : task->children )
                if ( tr )
                    os << "  node_" << task << " -> node_" << tr.ptr() << ";\n";

            //            for( const Task *tr : task->parents )
            //                os << "  node_" << tr << " -> node_" << task << " [color=red];\n";

        }, seen, /*go to parents*/ true );
    }
    os << "}\n";

    os.close();

    exec_dot( f, prg );
}

void Task::check_input_same_memory() {
    VecUnique<hardware_information::Memory *> memories;
    for( const Rc<Task> &ch : children )
        if ( ch->output.type )
            ch->output.type->get_memories( memories, ch->output.data );
}

void Task::insert_child( std::size_t num_child, const Rc<Task> &new_child ) {
    if ( children[ num_child ] )
        children[ num_child ]->remove_from_parents( this );
    if ( new_child )
        new_child->parents.push_back( this );
    children[ num_child ] = new_child;
}

Rc<Task> Task::move_child( std::size_t num_child ) {
    children[ num_child ]->remove_from_parents( this );
    return std::move( children[ num_child ] );
}

Type *Task::type_factory_virtual( const std::string &name ) {
    TypeFactory &tf = type_factory_virtual();
    return tf( name );
}

TypeFactory &Task::type_factory_virtual() {
    return type_factory();
}

Type *Task::type_factory( const std::string &name ) {
    TypeFactory &tf = type_factory();
    return tf( name );
}

TypeFactory &Task::type_factory() {
    static TypeFactory res;
    return res;
}

} // namespace parex
